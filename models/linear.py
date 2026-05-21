import numpy as np
import sklearn.decomposition
import sklearn.linear_model
import sklearn.preprocessing
import skglm
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler

# Hyperparameter grids (match RegularizedLinearModel defaults across hand-built PCA/macro pipelines)
_RL_LASSO_ELASTIC_ALPHAS = np.logspace(-5, 1, 30)
_RL_RIDGE_ALPHAS = np.logspace(-5, 5, 30)
_RL_ELASTICNET_L1_RATIOS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
_RL_MAX_ITER = 20000


class RegularizedLinearModel:
    """
    Unified linear model wrapper for Lasso, Ridge, and ElasticNet.

    Feature matrix is standardized with ``StandardScaler`` fit on the **current training
    sample only** each ``fit()`` (no future OOS rows). Hyperparameters are chosen on an
    85/15 **temporal** split using a **temporary** scaler fit only on that sub-train fold,
    then the instance ``scaler_features`` is ``fit`` on the **full** training matrix for
    the final model—mirroring ``LassoRawMacroFwdDirectModel`` / ``RidgeRawMacroFwdDirectModel``.

    When ``include_fred=True``, one scaler is applied to the stacked ``[macro | forward]``
    block (not the two-stage macro-then-block scaling of the raw-macro helpers).

    Features:
    - model_type: "lasso", "ridge", or "elasticnet"; ``include_fred`` / ``forward_max_cols``, etc.
    - tune_every: retune hyperparameters only every N ``fit()`` calls when enabled.
    """

    def __init__(
        self,
        model_type="lasso",
        *,
        alphas=None,
        l1_ratios=None,
        series=None,
        include_fred=None,
        forward_max_cols=None,
        optimize_hyperparams=True,
        tune_every=1,
        max_iter=_RL_MAX_ITER,
        random_state=42,
    ):
        model_type = str(model_type).lower()
        if model_type not in {"lasso", "ridge", "elasticnet"}:
            raise ValueError("model_type must be one of: 'lasso', 'ridge', 'elasticnet'.")

        if tune_every < 1:
            raise ValueError("tune_every must be >= 1.")

        if include_fred is not None and series is not None:
            raise ValueError("Use either include_fred or series, not both.")

        if forward_max_cols is not None:
            k = int(forward_max_cols)
            if k < 1:
                raise ValueError("forward_max_cols must be >= 1 when set.")
            self.forward_max_cols = k
        else:
            self.forward_max_cols = None

        self.model_type = model_type
        self.alphas = (
            np.array(alphas)
            if alphas is not None
            else (
                _RL_LASSO_ELASTIC_ALPHAS.copy()
                if model_type in {"lasso", "elasticnet"}
                else _RL_RIDGE_ALPHAS.copy()
            )
        )
        self.l1_ratios = (
            np.array(l1_ratios) if l1_ratios is not None else _RL_ELASTICNET_L1_RATIOS.copy()
        )
        self.series = series
        self.include_fred = include_fred
        self.optimize_hyperparams = optimize_hyperparams
        self.tune_every = int(tune_every)
        self.max_iter = max_iter
        self.random_state = random_state

        self.model = None
        self.scaler_features = sklearn.preprocessing.StandardScaler()
        self.best_alpha_ = None
        self.best_l1_ratio_ = None
        self._fit_count = 0

    def _subset_features(self, X):
        if self.include_fred is not None:
            if not hasattr(X, "columns"):
                raise ValueError("X must be a DataFrame with MultiIndex column blocks when include_fred is set.")
            if not hasattr(X.columns, "levels"):
                raise ValueError(
                    "Expected MultiIndex columns with top-level blocks including 'forward' and optionally 'fred'."
                )

            level0 = X.columns.get_level_values(0)
            if "forward" not in level0:
                raise ValueError("Expected 'forward' block in level-0 columns.")

            fwd_block = X["forward"]
            if self.forward_max_cols is not None:
                fwd_block = fwd_block.iloc[:, : self.forward_max_cols]
            X_forward = fwd_block.values
            if self.include_fred:
                if "fred" not in level0:
                    raise ValueError("Expected 'fred' block in level-0 columns when include_fred=True.")
                X_fred = X["fred"].values
                return np.hstack([X_fred, X_forward])
            return X_forward

        if self.series is None:
            X_sub = X
        else:
            if isinstance(self.series, str):
                X_sub = X[[self.series]]
            else:
                X_sub = X[self.series]

        return X_sub.values if hasattr(X_sub, "values") else np.asarray(X_sub)

    def _needs_tuning(self):
        if not self.optimize_hyperparams:
            return False
        if self.best_alpha_ is None:
            return True
        return self._fit_count % self.tune_every == 0

    def _build_model(self, alpha, l1_ratio=None):
        if self.model_type == "lasso":
            return sklearn.linear_model.Lasso(alpha=alpha, max_iter=self.max_iter)
        if self.model_type == "ridge":
            return sklearn.linear_model.Ridge(alpha=alpha)
        return sklearn.linear_model.ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def _default_params(self):
        alpha = float(np.median(self.alphas))
        if self.model_type == "elasticnet":
            l1_ratio = float(np.median(self.l1_ratios))
            return alpha, l1_ratio
        return alpha, None

    def _tune_params(self, X_vals, y_vals):
        n = len(y_vals)
        split = int(n * 0.85)

        # Need enough observations in validation set to tune robustly.
        if split < 10 or (n - split) < 3:
            return self._default_params()

        X_subtrain, X_val = X_vals[:split], X_vals[split:]
        y_subtrain, y_val = y_vals[:split], y_vals[split:]

        scaler_tune = sklearn.preprocessing.StandardScaler()
        X_sub_sc = scaler_tune.fit_transform(X_subtrain)
        X_val_sc = scaler_tune.transform(X_val)

        best_alpha, best_l1_ratio = self._default_params()
        best_mse = np.inf

        if self.model_type == "elasticnet":
            for alpha in self.alphas:
                for l1_ratio in self.l1_ratios:
                    m = self._build_model(alpha=float(alpha), l1_ratio=float(l1_ratio))
                    m.fit(X_sub_sc, y_subtrain)
                    mse = np.mean((y_val - m.predict(X_val_sc)) ** 2)
                    if mse < best_mse:
                        best_mse = mse
                        best_alpha = float(alpha)
                        best_l1_ratio = float(l1_ratio)
            return best_alpha, best_l1_ratio

        for alpha in self.alphas:
            m = self._build_model(alpha=float(alpha))
            m.fit(X_sub_sc, y_subtrain)
            mse = np.mean((y_val - m.predict(X_val_sc)) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)

        return best_alpha, None

    def fit(self, X, y):
        X_vals = self._subset_features(X)
        y_vals = np.asarray(y).ravel()

        if self._needs_tuning():
            self.best_alpha_, self.best_l1_ratio_ = self._tune_params(X_vals, y_vals)
        elif self.best_alpha_ is None:
            self.best_alpha_, self.best_l1_ratio_ = self._default_params()

        X_model = self.scaler_features.fit_transform(X_vals)
        self.model = self._build_model(alpha=self.best_alpha_, l1_ratio=self.best_l1_ratio_)
        self.model.fit(X_model, y_vals)
        self._fit_count += 1

    def predict(self, X):
        X_vals = self._subset_features(X)
        X_model = self.scaler_features.transform(X_vals)
        return self.model.predict(X_model)


### Lasso and Ridge models with raw macro features
class LassoRawMacroFwdDirectModel:
    """
    Lasso on:
      - full scaled macro dataset
      - forward regressors directly (optional first ``forward_max_cols`` columns)

    No macro PCA / LN compression.
    Proper 85/15 temporal tuning split.
    """

    def __init__(
        self,
        alphas=None,
        macro_series='fred',
        forward_series='forward',
        forward_max_cols=None,
    ):
        self.alphas = alphas if alphas is not None else _RL_LASSO_ELASTIC_ALPHAS.copy()
        self.macro_series = macro_series
        self.forward_series = forward_series
        self.forward_max_cols = forward_max_cols

        self.scaler_macro = sklearn.preprocessing.StandardScaler()
        self.scaler_features = sklearn.preprocessing.StandardScaler()

        self.best_alpha_ = None
        self.model = None

    def _forward_values(self, X):
        fwd = X[self.forward_series]
        if self.forward_max_cols is not None:
            fwd = fwd.iloc[:, : self.forward_max_cols]
        return fwd.values

    def _fit_feature_pipeline(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.fit_transform(macro)

        forward = self._forward_values(X)
        return np.concatenate([macro_scaled, forward], axis=1)

    def _transform_features(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.transform(macro)

        forward = self._forward_values(X)
        return np.concatenate([macro_scaled, forward], axis=1)

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y)
        n = len(y)
        split = int(n * 0.85)

        if len(y) < 10 or len(y[split:]) < 3:
            features = self._fit_feature_pipeline(X)
            features_scaled = self.scaler_features.fit_transform(features)
            self.best_alpha_ = np.median(self.alphas)
            self.model = sklearn.linear_model.Lasso(alpha=self.best_alpha_, max_iter=_RL_MAX_ITER)
            self.model.fit(features_scaled, y)
            return

        X_train = X.iloc[:split]
        X_val_int = X.iloc[split:]
        y_train = y[:split]
        y_val_int = y[split:]

        features_train = self._fit_feature_pipeline(X_train)
        features_val = self._transform_features(X_val_int)

        scaler_tune = sklearn.preprocessing.StandardScaler()
        X_train_scaled = scaler_tune.fit_transform(features_train)
        X_val_scaled = scaler_tune.transform(features_val)

        best_alpha = self.alphas[0]
        best_mse = np.inf

        for alpha in self.alphas:
            m = sklearn.linear_model.Lasso(alpha=alpha, max_iter=_RL_MAX_ITER)
            m.fit(X_train_scaled, y_train)
            mse = np.mean((y_val_int - m.predict(X_val_scaled)) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        self.best_alpha_ = best_alpha

        features_full = self._fit_feature_pipeline(X)
        features_full_scaled = self.scaler_features.fit_transform(features_full)

        self.model = sklearn.linear_model.Lasso(alpha=self.best_alpha_, max_iter=_RL_MAX_ITER)
        self.model.fit(features_full_scaled, y)

    def predict(self, X):
        features = self._transform_features(X)
        features_scaled = self.scaler_features.transform(features)
        return self.model.predict(features_scaled)


class RidgeRawMacroFwdDirectModel:
    """
    Ridge on:
      - full scaled macro dataset
      - forward regressors directly (optional first ``forward_max_cols`` columns)

    No macro PCA / LN compression.
    Proper 85/15 temporal tuning split.
    """

    def __init__(
        self,
        alphas=None,
        macro_series='fred',
        forward_series='forward',
        forward_max_cols=None,
    ):
        self.alphas = alphas if alphas is not None else _RL_RIDGE_ALPHAS.copy()
        self.macro_series = macro_series
        self.forward_series = forward_series
        self.forward_max_cols = forward_max_cols

        self.scaler_macro = sklearn.preprocessing.StandardScaler()
        self.scaler_features = sklearn.preprocessing.StandardScaler()

        self.best_alpha_ = None
        self.model = None

    def _forward_values(self, X):
        fwd = X[self.forward_series]
        if self.forward_max_cols is not None:
            fwd = fwd.iloc[:, : self.forward_max_cols]
        return fwd.values

    def _fit_feature_pipeline(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.fit_transform(macro)

        forward = self._forward_values(X)
        return np.concatenate([macro_scaled, forward], axis=1)

    def _transform_features(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.transform(macro)

        forward = self._forward_values(X)
        return np.concatenate([macro_scaled, forward], axis=1)

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y)
        n = len(y)
        split = int(n * 0.85)

        if len(y) < 10 or len(y[split:]) < 3:
            features = self._fit_feature_pipeline(X)
            features_scaled = self.scaler_features.fit_transform(features)
            self.best_alpha_ = np.median(self.alphas)
            self.model = sklearn.linear_model.Ridge(alpha=self.best_alpha_)
            self.model.fit(features_scaled, y)
            return

        X_train = X.iloc[:split]
        X_val_int = X.iloc[split:]
        y_train = y[:split]
        y_val_int = y[split:]

        features_train = self._fit_feature_pipeline(X_train)
        features_val = self._transform_features(X_val_int)

        scaler_tune = sklearn.preprocessing.StandardScaler()
        X_train_scaled = scaler_tune.fit_transform(features_train)
        X_val_scaled = scaler_tune.transform(features_val)

        best_alpha = self.alphas[0]
        best_mse = np.inf

        for alpha in self.alphas:
            m = sklearn.linear_model.Ridge(alpha=alpha)
            m.fit(X_train_scaled, y_train)
            mse = np.mean((y_val_int - m.predict(X_val_scaled)) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        self.best_alpha_ = best_alpha

        features_full = self._fit_feature_pipeline(X)
        features_full_scaled = self.scaler_features.fit_transform(features_full)

        self.model = sklearn.linear_model.Ridge(alpha=self.best_alpha_)
        self.model.fit(features_full_scaled, y)

    def predict(self, X):
        features = self._transform_features(X)
        features_scaled = self.scaler_features.transform(features)
        return self.model.predict(features_scaled)


class ElasticNetRawMacroFwdDirectModel:
    """
    ElasticNet on:
      - full scaled macro dataset
      - forward regressors directly (optional first ``forward_max_cols`` columns)

    Same feature construction as LassoRawMacroFwdDirectModel.
    Alpha / l1_ratio grids match `RegularizedLinearModel` with model_type elasticnet.
    """

    def __init__(
        self,
        alphas=None,
        l1_ratios=None,
        macro_series="fred",
        forward_series="forward",
        forward_max_cols=None,
        max_iter=_RL_MAX_ITER,
        random_state=42,
    ):
        self.alphas = alphas if alphas is not None else _RL_LASSO_ELASTIC_ALPHAS.copy()
        self.l1_ratios = l1_ratios if l1_ratios is not None else _RL_ELASTICNET_L1_RATIOS.copy()
        self.macro_series = macro_series
        self.forward_series = forward_series
        self.forward_max_cols = forward_max_cols
        self.max_iter = max_iter
        self.random_state = random_state

        self.scaler_macro = sklearn.preprocessing.StandardScaler()
        self.scaler_features = sklearn.preprocessing.StandardScaler()

        self.best_alpha_ = None
        self.best_l1_ratio_ = None
        self.model = None

    def _forward_values(self, X):
        fwd = X[self.forward_series]
        if self.forward_max_cols is not None:
            fwd = fwd.iloc[:, : self.forward_max_cols]
        return fwd.values

    def _fit_feature_pipeline(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.fit_transform(macro)

        forward = self._forward_values(X)
        return np.concatenate([macro_scaled, forward], axis=1)

    def _transform_features(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.transform(macro)

        forward = self._forward_values(X)
        return np.concatenate([macro_scaled, forward], axis=1)

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y)
        n = len(y)
        split = int(n * 0.85)

        if len(y) < 10 or len(y[split:]) < 3:
            features = self._fit_feature_pipeline(X)
            features_scaled = self.scaler_features.fit_transform(features)
            self.best_alpha_ = float(np.median(self.alphas))
            self.best_l1_ratio_ = float(np.median(self.l1_ratios))
            self.model = sklearn.linear_model.ElasticNet(
                alpha=self.best_alpha_,
                l1_ratio=self.best_l1_ratio_,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            self.model.fit(features_scaled, y)
            return

        X_train = X.iloc[:split]
        X_val_int = X.iloc[split:]
        y_train = y[:split]
        y_val_int = y[split:]

        features_train = self._fit_feature_pipeline(X_train)
        features_val = self._transform_features(X_val_int)

        scaler_tune = sklearn.preprocessing.StandardScaler()
        X_train_scaled = scaler_tune.fit_transform(features_train)
        X_val_scaled = scaler_tune.transform(features_val)

        best_alpha = float(self.alphas[0])
        best_l1_ratio = float(self.l1_ratios[0])
        best_mse = np.inf

        for alpha in self.alphas:
            for l1_ratio in self.l1_ratios:
                m = sklearn.linear_model.ElasticNet(
                    alpha=float(alpha),
                    l1_ratio=float(l1_ratio),
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                )
                m.fit(X_train_scaled, y_train)
                mse = np.mean((y_val_int - m.predict(X_val_scaled)) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = float(alpha)
                    best_l1_ratio = float(l1_ratio)

        self.best_alpha_ = best_alpha
        self.best_l1_ratio_ = best_l1_ratio

        features_full = self._fit_feature_pipeline(X)
        features_full_scaled = self.scaler_features.fit_transform(features_full)

        self.model = sklearn.linear_model.ElasticNet(
            alpha=self.best_alpha_,
            l1_ratio=self.best_l1_ratio_,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model.fit(features_full_scaled, y)

    def predict(self, X):
        features = self._transform_features(X)
        features_scaled = self.scaler_features.transform(features)
        return self.model.predict(features_scaled)


class LassoRawMacroCPModel:
    """
    Lasso on:
      - full scaled macro dataset
      - CP factor

    No macro PCA / LN compression.
    Proper 85/15 temporal tuning split.
    """

    def __init__(self, xr_full, cp_cols, alphas=None,
                 macro_series='fred', forward_series='forward', n_cp_forwards=5):
        self.xr_full = xr_full
        self.cp_cols = cp_cols
        self.alphas = alphas if alphas is not None else _RL_LASSO_ELASTIC_ALPHAS.copy()
        self.macro_series = macro_series
        self.forward_series = forward_series
        self.n_cp_forwards = n_cp_forwards

        self.scaler_macro = sklearn.preprocessing.StandardScaler()
        self.cp_model = sklearn.linear_model.LinearRegression()
        self.scaler_features = sklearn.preprocessing.StandardScaler()

        self.best_alpha_ = None
        self.model = None

    def _fit_feature_pipeline(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.fit_transform(macro)

        forward = X[self.forward_series].iloc[:, :self.n_cp_forwards]
        cp_target = self.xr_full.loc[X.index, self.cp_cols].mean(axis=1)
        self.cp_model.fit(forward, cp_target)
        cp_factor = self.cp_model.predict(forward).reshape(-1, 1)

        return np.concatenate([macro_scaled, cp_factor], axis=1)

    def _transform_features(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.transform(macro)

        forward = X[self.forward_series].iloc[:, :self.n_cp_forwards]
        cp_factor = self.cp_model.predict(forward).reshape(-1, 1)

        return np.concatenate([macro_scaled, cp_factor], axis=1)

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y)
        n = len(y)
        split = int(n * 0.85)

        if len(y) < 10 or len(y[split:]) < 3:
            features = self._fit_feature_pipeline(X)
            features_scaled = self.scaler_features.fit_transform(features)
            self.best_alpha_ = np.median(self.alphas)
            self.model = sklearn.linear_model.Lasso(alpha=self.best_alpha_, max_iter=_RL_MAX_ITER)
            self.model.fit(features_scaled, y)
            return

        X_train = X.iloc[:split]
        X_val_int = X.iloc[split:]
        y_train = y[:split]
        y_val_int = y[split:]

        features_train = self._fit_feature_pipeline(X_train)
        features_val = self._transform_features(X_val_int)

        scaler_tune = sklearn.preprocessing.StandardScaler()
        X_train_scaled = scaler_tune.fit_transform(features_train)
        X_val_scaled = scaler_tune.transform(features_val)

        best_alpha = self.alphas[0]
        best_mse = np.inf

        for alpha in self.alphas:
            m = sklearn.linear_model.Lasso(alpha=alpha, max_iter=_RL_MAX_ITER)
            m.fit(X_train_scaled, y_train)
            mse = np.mean((y_val_int - m.predict(X_val_scaled)) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        self.best_alpha_ = best_alpha

        features_full = self._fit_feature_pipeline(X)
        features_full_scaled = self.scaler_features.fit_transform(features_full)

        self.model = sklearn.linear_model.Lasso(alpha=self.best_alpha_, max_iter=_RL_MAX_ITER)
        self.model.fit(features_full_scaled, y)

    def predict(self, X):
        features = self._transform_features(X)
        features_scaled = self.scaler_features.transform(features)
        return self.model.predict(features_scaled)


class RidgeRawMacroCPModel:
    """
    Ridge on:
      - full scaled macro dataset
      - CP factor

    No macro PCA / LN compression.
    Proper 85/15 temporal tuning split.
    """

    def __init__(self, xr_full, cp_cols, alphas=None,
                 macro_series='fred', forward_series='forward', n_cp_forwards=5):
        self.xr_full = xr_full
        self.cp_cols = cp_cols
        self.alphas = alphas if alphas is not None else _RL_RIDGE_ALPHAS.copy()
        self.macro_series = macro_series
        self.forward_series = forward_series
        self.n_cp_forwards = n_cp_forwards

        self.scaler_macro = sklearn.preprocessing.StandardScaler()
        self.cp_model = sklearn.linear_model.LinearRegression()
        self.scaler_features = sklearn.preprocessing.StandardScaler()

        self.best_alpha_ = None
        self.model = None

    def _fit_feature_pipeline(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.fit_transform(macro)

        forward = X[self.forward_series].iloc[:, :self.n_cp_forwards]
        cp_target = self.xr_full.loc[X.index, self.cp_cols].mean(axis=1)
        self.cp_model.fit(forward, cp_target)
        cp_factor = self.cp_model.predict(forward).reshape(-1, 1)

        return np.concatenate([macro_scaled, cp_factor], axis=1)

    def _transform_features(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.transform(macro)

        forward = X[self.forward_series].iloc[:, :self.n_cp_forwards]
        cp_factor = self.cp_model.predict(forward).reshape(-1, 1)

        return np.concatenate([macro_scaled, cp_factor], axis=1)

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y)
        n = len(y)
        split = int(n * 0.85)

        if len(y) < 10 or len(y[split:]) < 3:
            features = self._fit_feature_pipeline(X)
            features_scaled = self.scaler_features.fit_transform(features)
            self.best_alpha_ = np.median(self.alphas)
            self.model = sklearn.linear_model.Ridge(alpha=self.best_alpha_)
            self.model.fit(features_scaled, y)
            return

        X_train = X.iloc[:split]
        X_val_int = X.iloc[split:]
        y_train = y[:split]
        y_val_int = y[split:]

        features_train = self._fit_feature_pipeline(X_train)
        features_val = self._transform_features(X_val_int)

        scaler_tune = sklearn.preprocessing.StandardScaler()
        X_train_scaled = scaler_tune.fit_transform(features_train)
        X_val_scaled = scaler_tune.transform(features_val)

        best_alpha = self.alphas[0]
        best_mse = np.inf

        for alpha in self.alphas:
            m = sklearn.linear_model.Ridge(alpha=alpha)
            m.fit(X_train_scaled, y_train)
            mse = np.mean((y_val_int - m.predict(X_val_scaled)) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        self.best_alpha_ = best_alpha

        features_full = self._fit_feature_pipeline(X)
        features_full_scaled = self.scaler_features.fit_transform(features_full)

        self.model = sklearn.linear_model.Ridge(alpha=self.best_alpha_)
        self.model.fit(features_full_scaled, y)

    def predict(self, X):
        features = self._transform_features(X)
        features_scaled = self.scaler_features.transform(features)
        return self.model.predict(features_scaled)


class ElasticNetRawMacroCPModel:
    """
    ElasticNet on:
      - full scaled macro dataset
      - CP factor

    Same construction as LassoRawMacroCPModel; grids match RegularizedLinearModel.elasticnet.
    """

    def __init__(
        self,
        xr_full,
        cp_cols,
        alphas=None,
        l1_ratios=None,
        macro_series="fred",
        forward_series="forward",
        n_cp_forwards=5,
        max_iter=_RL_MAX_ITER,
        random_state=42,
    ):
        self.xr_full = xr_full
        self.cp_cols = cp_cols
        self.alphas = alphas if alphas is not None else _RL_LASSO_ELASTIC_ALPHAS.copy()
        self.l1_ratios = l1_ratios if l1_ratios is not None else _RL_ELASTICNET_L1_RATIOS.copy()
        self.macro_series = macro_series
        self.forward_series = forward_series
        self.n_cp_forwards = n_cp_forwards
        self.max_iter = max_iter
        self.random_state = random_state

        self.scaler_macro = sklearn.preprocessing.StandardScaler()
        self.cp_model = sklearn.linear_model.LinearRegression()
        self.scaler_features = sklearn.preprocessing.StandardScaler()

        self.best_alpha_ = None
        self.best_l1_ratio_ = None
        self.model = None

    def _fit_feature_pipeline(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.fit_transform(macro)

        forward = X[self.forward_series].iloc[:, : self.n_cp_forwards]
        cp_target = self.xr_full.loc[X.index, self.cp_cols].mean(axis=1)
        self.cp_model.fit(forward, cp_target)
        cp_factor = self.cp_model.predict(forward).reshape(-1, 1)

        return np.concatenate([macro_scaled, cp_factor], axis=1)

    def _transform_features(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.transform(macro)

        forward = X[self.forward_series].iloc[:, : self.n_cp_forwards]
        cp_factor = self.cp_model.predict(forward).reshape(-1, 1)

        return np.concatenate([macro_scaled, cp_factor], axis=1)

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y)
        n = len(y)
        split = int(n * 0.85)

        if len(y) < 10 or len(y[split:]) < 3:
            features = self._fit_feature_pipeline(X)
            features_scaled = self.scaler_features.fit_transform(features)
            self.best_alpha_ = float(np.median(self.alphas))
            self.best_l1_ratio_ = float(np.median(self.l1_ratios))
            self.model = sklearn.linear_model.ElasticNet(
                alpha=self.best_alpha_,
                l1_ratio=self.best_l1_ratio_,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            self.model.fit(features_scaled, y)
            return

        X_train = X.iloc[:split]
        X_val_int = X.iloc[split:]
        y_train = y[:split]
        y_val_int = y[split:]

        features_train = self._fit_feature_pipeline(X_train)
        features_val = self._transform_features(X_val_int)

        scaler_tune = sklearn.preprocessing.StandardScaler()
        X_train_scaled = scaler_tune.fit_transform(features_train)
        X_val_scaled = scaler_tune.transform(features_val)

        best_alpha = float(self.alphas[0])
        best_l1_ratio = float(self.l1_ratios[0])
        best_mse = np.inf

        for alpha in self.alphas:
            for l1_ratio in self.l1_ratios:
                m = sklearn.linear_model.ElasticNet(
                    alpha=float(alpha),
                    l1_ratio=float(l1_ratio),
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                )
                m.fit(X_train_scaled, y_train)
                mse = np.mean((y_val_int - m.predict(X_val_scaled)) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = float(alpha)
                    best_l1_ratio = float(l1_ratio)

        self.best_alpha_ = best_alpha
        self.best_l1_ratio_ = best_l1_ratio

        features_full = self._fit_feature_pipeline(X)
        features_full_scaled = self.scaler_features.fit_transform(features_full)

        self.model = sklearn.linear_model.ElasticNet(
            alpha=self.best_alpha_,
            l1_ratio=self.best_l1_ratio_,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model.fit(features_full_scaled, y)

    def predict(self, X):
        features = self._transform_features(X)
        features_scaled = self.scaler_features.transform(features)
        return self.model.predict(features_scaled)


### With LN macro features: F1, F1^3, F3, F4, F8
class LassoMacroFwdDirectModel:
    """
    Panel B, model 3:
    Lasso using
      - LN macro features: F1, F1^3, F3, F4, F8
      - all forward regressors directly

    Hyperparameter tuning is done with a proper 85/15 temporal split,
    with no leakage from validation into feature construction.
    """

    def __init__(self, alphas=None, macro_series='fred', forward_series='forward'):
        self.alphas = alphas if alphas is not None else _RL_LASSO_ELASTIC_ALPHAS.copy()
        self.macro_series = macro_series
        self.forward_series = forward_series

        self.scaler_macro = sklearn.preprocessing.StandardScaler()
        self.pca = sklearn.decomposition.PCA(n_components=8)

        self.scaler_features = sklearn.preprocessing.StandardScaler()
        self.best_alpha_ = None
        self.model = None

    def _fit_feature_pipeline(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.fit_transform(macro)
        factors = self.pca.fit_transform(macro_scaled)

        F1 = factors[:, [0]]
        F3 = factors[:, [2]]
        F4 = factors[:, [3]]
        F8 = factors[:, [7]]
        ln_features = np.concatenate([F1, F1**3, F3, F4, F8], axis=1)

        forward = X[self.forward_series].values
        return np.concatenate([ln_features, forward], axis=1)

    def _transform_features(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.transform(macro)
        factors = self.pca.transform(macro_scaled)

        F1 = factors[:, [0]]
        F3 = factors[:, [2]]
        F4 = factors[:, [3]]
        F8 = factors[:, [7]]
        ln_features = np.concatenate([F1, F1**3, F3, F4, F8], axis=1)

        forward = X[self.forward_series].values
        return np.concatenate([ln_features, forward], axis=1)

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y)
        n = len(y)
        split = int(n * 0.85)

        if len(y) < 10 or len(y[split:]) < 3:
            features = self._fit_feature_pipeline(X)
            features_scaled = self.scaler_features.fit_transform(features)
            self.best_alpha_ = np.median(self.alphas)
            self.model = sklearn.linear_model.Lasso(alpha=self.best_alpha_, max_iter=_RL_MAX_ITER)
            self.model.fit(features_scaled, y)
            return

        X_train = X.iloc[:split]
        X_val_int = X.iloc[split:]
        y_train = y[:split]
        y_val_int = y[split:]

        features_train = self._fit_feature_pipeline(X_train)
        features_val = self._transform_features(X_val_int)

        scaler_tune = sklearn.preprocessing.StandardScaler()
        X_train_scaled = scaler_tune.fit_transform(features_train)
        X_val_scaled = scaler_tune.transform(features_val)

        best_alpha = self.alphas[0]
        best_mse = np.inf

        for alpha in self.alphas:
            m = sklearn.linear_model.Lasso(alpha=alpha, max_iter=_RL_MAX_ITER)
            m.fit(X_train_scaled, y_train)
            mse = np.mean((y_val_int - m.predict(X_val_scaled)) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        self.best_alpha_ = best_alpha

        features_full = self._fit_feature_pipeline(X)
        features_full_scaled = self.scaler_features.fit_transform(features_full)

        self.model = sklearn.linear_model.Lasso(alpha=self.best_alpha_, max_iter=_RL_MAX_ITER)
        self.model.fit(features_full_scaled, y)

    def predict(self, X):
        features = self._transform_features(X)
        features_scaled = self.scaler_features.transform(features)
        return self.model.predict(features_scaled)


class RidgeMacroFwdDirectModel:
    """
    Panel B, model 4:
    Ridge using
      - LN macro features: F1, F1^3, F3, F4, F8
      - all forward regressors directly

    Hyperparameter tuning is done with a proper 85/15 temporal split,
    with no leakage from validation into feature construction.
    """

    def __init__(self, alphas=None, macro_series='fred', forward_series='forward'):
        self.alphas = alphas if alphas is not None else _RL_RIDGE_ALPHAS.copy()
        self.macro_series = macro_series
        self.forward_series = forward_series

        self.scaler_macro = sklearn.preprocessing.StandardScaler()
        self.pca = sklearn.decomposition.PCA(n_components=8)

        self.scaler_features = sklearn.preprocessing.StandardScaler()
        self.best_alpha_ = None
        self.model = None

    def _fit_feature_pipeline(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.fit_transform(macro)
        factors = self.pca.fit_transform(macro_scaled)

        F1 = factors[:, [0]]
        F3 = factors[:, [2]]
        F4 = factors[:, [3]]
        F8 = factors[:, [7]]
        ln_features = np.concatenate([F1, F1**3, F3, F4, F8], axis=1)

        forward = X[self.forward_series].values
        return np.concatenate([ln_features, forward], axis=1)

    def _transform_features(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.transform(macro)
        factors = self.pca.transform(macro_scaled)

        F1 = factors[:, [0]]
        F3 = factors[:, [2]]
        F4 = factors[:, [3]]
        F8 = factors[:, [7]]
        ln_features = np.concatenate([F1, F1**3, F3, F4, F8], axis=1)

        forward = X[self.forward_series].values
        return np.concatenate([ln_features, forward], axis=1)

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y)
        n = len(y)
        split = int(n * 0.85)

        if len(y) < 10 or len(y[split:]) < 3:
            features = self._fit_feature_pipeline(X)
            features_scaled = self.scaler_features.fit_transform(features)
            self.best_alpha_ = np.median(self.alphas)
            self.model = sklearn.linear_model.Ridge(alpha=self.best_alpha_)
            self.model.fit(features_scaled, y)
            return

        X_train = X.iloc[:split]
        X_val_int = X.iloc[split:]
        y_train = y[:split]
        y_val_int = y[split:]

        features_train = self._fit_feature_pipeline(X_train)
        features_val = self._transform_features(X_val_int)

        scaler_tune = sklearn.preprocessing.StandardScaler()
        X_train_scaled = scaler_tune.fit_transform(features_train)
        X_val_scaled = scaler_tune.transform(features_val)

        best_alpha = self.alphas[0]
        best_mse = np.inf

        for alpha in self.alphas:
            m = sklearn.linear_model.Ridge(alpha=alpha)
            m.fit(X_train_scaled, y_train)
            mse = np.mean((y_val_int - m.predict(X_val_scaled)) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        self.best_alpha_ = best_alpha

        features_full = self._fit_feature_pipeline(X)
        features_full_scaled = self.scaler_features.fit_transform(features_full)

        self.model = sklearn.linear_model.Ridge(alpha=self.best_alpha_)
        self.model.fit(features_full_scaled, y)

    def predict(self, X):
        features = self._transform_features(X)
        features_scaled = self.scaler_features.transform(features)
        return self.model.predict(features_scaled)


class LassoMacroCPModel:
    """
    Panel B, model 5:
    Lasso using
      - LN macro features: F1, F1^3, F3, F4, F8
      - CP factor

    CP factor is constructed from the first n_cp_forwards forward columns.
    Hyperparameter tuning is done with a proper 85/15 temporal split,
    with no leakage from validation into feature construction.
    """

    def __init__(self, xr_full, cp_cols, alphas=None,
                 macro_series='fred', forward_series='forward', n_cp_forwards=5):
        self.xr_full = xr_full
        self.cp_cols = cp_cols
        self.alphas = alphas if alphas is not None else _RL_LASSO_ELASTIC_ALPHAS.copy()
        self.macro_series = macro_series
        self.forward_series = forward_series
        self.n_cp_forwards = n_cp_forwards

        self.scaler_macro = sklearn.preprocessing.StandardScaler()
        self.pca = sklearn.decomposition.PCA(n_components=8)
        self.cp_model = sklearn.linear_model.LinearRegression()

        self.scaler_features = sklearn.preprocessing.StandardScaler()
        self.best_alpha_ = None
        self.model = None

    def _fit_feature_pipeline(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.fit_transform(macro)
        factors = self.pca.fit_transform(macro_scaled)

        F1 = factors[:, [0]]
        F3 = factors[:, [2]]
        F4 = factors[:, [3]]
        F8 = factors[:, [7]]
        ln_features = np.concatenate([F1, F1**3, F3, F4, F8], axis=1)

        forward = X[self.forward_series].iloc[:, :self.n_cp_forwards]
        cp_target = self.xr_full.loc[X.index, self.cp_cols].mean(axis=1)
        self.cp_model.fit(forward, cp_target)
        cp_factor = self.cp_model.predict(forward).reshape(-1, 1)

        return np.concatenate([ln_features, cp_factor], axis=1)

    def _transform_features(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.transform(macro)
        factors = self.pca.transform(macro_scaled)

        F1 = factors[:, [0]]
        F3 = factors[:, [2]]
        F4 = factors[:, [3]]
        F8 = factors[:, [7]]
        ln_features = np.concatenate([F1, F1**3, F3, F4, F8], axis=1)

        forward = X[self.forward_series].iloc[:, :self.n_cp_forwards]
        cp_factor = self.cp_model.predict(forward).reshape(-1, 1)

        return np.concatenate([ln_features, cp_factor], axis=1)

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y)
        n = len(y)
        split = int(n * 0.85)

        if len(y) < 10 or len(y[split:]) < 3:
            features = self._fit_feature_pipeline(X)
            features_scaled = self.scaler_features.fit_transform(features)
            self.best_alpha_ = np.median(self.alphas)
            self.model = sklearn.linear_model.Lasso(alpha=self.best_alpha_, max_iter=_RL_MAX_ITER)
            self.model.fit(features_scaled, y)
            return

        X_train = X.iloc[:split]
        X_val_int = X.iloc[split:]
        y_train = y[:split]
        y_val_int = y[split:]

        # Fit feature pipeline on subtrain only
        features_train = self._fit_feature_pipeline(X_train)
        features_val = self._transform_features(X_val_int)

        scaler_tune = sklearn.preprocessing.StandardScaler()
        X_train_scaled = scaler_tune.fit_transform(features_train)
        X_val_scaled = scaler_tune.transform(features_val)

        best_alpha = self.alphas[0]
        best_mse = np.inf

        for alpha in self.alphas:
            m = sklearn.linear_model.Lasso(alpha=alpha, max_iter=_RL_MAX_ITER)
            m.fit(X_train_scaled, y_train)
            mse = np.mean((y_val_int - m.predict(X_val_scaled)) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        self.best_alpha_ = best_alpha

        # Refit everything on full in-sample window
        features_full = self._fit_feature_pipeline(X)
        features_full_scaled = self.scaler_features.fit_transform(features_full)

        self.model = sklearn.linear_model.Lasso(alpha=self.best_alpha_, max_iter=_RL_MAX_ITER)
        self.model.fit(features_full_scaled, y)

    def predict(self, X):
        features = self._transform_features(X)
        features_scaled = self.scaler_features.transform(features)
        return self.model.predict(features_scaled)


class RidgeMacroCPModel:
    """
    Panel B, model 6:
    Ridge using
      - LN macro features: F1, F1^3, F3, F4, F8
      - CP factor

    CP factor is constructed from the first n_cp_forwards forward columns.
    Hyperparameter tuning is done with a proper 85/15 temporal split,
    with no leakage from validation into feature construction.
    """

    def __init__(self, xr_full, cp_cols, alphas=None,
                 macro_series='fred', forward_series='forward', n_cp_forwards=5):
        self.xr_full = xr_full
        self.cp_cols = cp_cols
        self.alphas = alphas if alphas is not None else _RL_RIDGE_ALPHAS.copy()
        self.macro_series = macro_series
        self.forward_series = forward_series
        self.n_cp_forwards = n_cp_forwards

        self.scaler_macro = sklearn.preprocessing.StandardScaler()
        self.pca = sklearn.decomposition.PCA(n_components=8)
        self.cp_model = sklearn.linear_model.LinearRegression()

        self.scaler_features = sklearn.preprocessing.StandardScaler()
        self.best_alpha_ = None
        self.model = None

    def _fit_feature_pipeline(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.fit_transform(macro)
        factors = self.pca.fit_transform(macro_scaled)

        F1 = factors[:, [0]]
        F3 = factors[:, [2]]
        F4 = factors[:, [3]]
        F8 = factors[:, [7]]
        ln_features = np.concatenate([F1, F1**3, F3, F4, F8], axis=1)

        forward = X[self.forward_series].iloc[:, :self.n_cp_forwards]
        cp_target = self.xr_full.loc[X.index, self.cp_cols].mean(axis=1)
        self.cp_model.fit(forward, cp_target)
        cp_factor = self.cp_model.predict(forward).reshape(-1, 1)

        return np.concatenate([ln_features, cp_factor], axis=1)

    def _transform_features(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.transform(macro)
        factors = self.pca.transform(macro_scaled)

        F1 = factors[:, [0]]
        F3 = factors[:, [2]]
        F4 = factors[:, [3]]
        F8 = factors[:, [7]]
        ln_features = np.concatenate([F1, F1**3, F3, F4, F8], axis=1)

        forward = X[self.forward_series].iloc[:, :self.n_cp_forwards]
        cp_factor = self.cp_model.predict(forward).reshape(-1, 1)

        return np.concatenate([ln_features, cp_factor], axis=1)

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y)
        n = len(y)
        split = int(n * 0.85)

        if len(y) < 10 or len(y[split:]) < 3:
            features = self._fit_feature_pipeline(X)
            features_scaled = self.scaler_features.fit_transform(features)
            self.best_alpha_ = np.median(self.alphas)
            self.model = sklearn.linear_model.Ridge(alpha=self.best_alpha_)
            self.model.fit(features_scaled, y)
            return

        X_train = X.iloc[:split]
        X_val_int = X.iloc[split:]
        y_train = y[:split]
        y_val_int = y[split:]

        # Fit feature pipeline on subtrain only
        features_train = self._fit_feature_pipeline(X_train)
        features_val = self._transform_features(X_val_int)

        scaler_tune = sklearn.preprocessing.StandardScaler()
        X_train_scaled = scaler_tune.fit_transform(features_train)
        X_val_scaled = scaler_tune.transform(features_val)

        best_alpha = self.alphas[0]
        best_mse = np.inf

        for alpha in self.alphas:
            m = sklearn.linear_model.Ridge(alpha=alpha)
            m.fit(X_train_scaled, y_train)
            mse = np.mean((y_val_int - m.predict(X_val_scaled)) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        self.best_alpha_ = best_alpha

        # Refit everything on full in-sample window
        features_full = self._fit_feature_pipeline(X)
        features_full_scaled = self.scaler_features.fit_transform(features_full)

        self.model = sklearn.linear_model.Ridge(alpha=self.best_alpha_)
        self.model.fit(features_full_scaled, y)

    def predict(self, X):
        features = self._transform_features(X)
        features_scaled = self.scaler_features.transform(features)
        return self.model.predict(features_scaled)
