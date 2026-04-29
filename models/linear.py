import numpy as np
import sklearn.linear_model
import skglm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit

class LassoModel:
    def __init__(self, alphas=None, series=None):
        self.alphas = alphas if alphas is not None else np.logspace(-5, 1, 30)
        self.series = series
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        # 1. Handle feature selection (subsetting columns)
        X_sub = X[self.series] if self.series else X
        
        n = len(y)
        split = int(n * 0.85)
        X_vals = X_sub.values
        X_subtrain, X_v = X_vals[:split], X_vals[split:]
        y_subtrain, y_v = y[:split], y[split:]

        best_alpha = self.alphas[0]
        best_mse = np.inf
        
        # Optimization: only tune if we have enough validation data
        if len(y_v) >= 3:
            for alpha in self.alphas:
                m = sklearn.linear_model.Lasso(alpha=alpha, max_iter=10000)
                m.fit(X_subtrain, y_subtrain)
                mse = np.mean((y_v - m.predict(X_v)) ** 2)
                if mse < best_mse:
                    best_mse, best_alpha = mse, alpha
        else:
            best_alpha = np.median(self.alphas)

        # 4. FINAL REFIT
        # Refit on subtrain + internal val
        X_final = np.vstack([X_subtrain, X_v])
        y_final = np.concatenate([y_subtrain, y_v])
        
        self.model = sklearn.linear_model.Lasso(alpha=best_alpha, max_iter=10000)
        self.model.fit(X_final, y_final)

    def predict(self, X):
        X_sub = X[self.series].values if self.series else X.values
        return self.model.predict(X_sub)



class LassoModelScaled:
    def __init__(self, alphas=None, series=None):
        self.alphas = alphas if alphas is not None else np.logspace(-5, 1, 30)
        self.series = series
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.model = None
        self.best_alpha = None

    def fit(self, X, y, X_val=None, y_val=None):
        X_sub = X[self.series] if self.series is not None else X
        X_vals = X_sub.values if hasattr(X_sub, "values") else np.asarray(X_sub)
        y = np.asarray(y)

        n = len(y)
        split = int(n * 0.85)

        X_train, X_v = X_vals[:split], X_vals[split:]
        y_train, y_v = y[:split], y[split:]

        # Scale using training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_v_scaled = self.scaler.transform(X_v) if len(X_v) > 0 else X_v

        best_alpha = self.alphas[0]
        best_mse = np.inf

        if len(y_v) >= 3:
            for alpha in self.alphas:
                m = sklearn.linear_model.Lasso(alpha=alpha, max_iter=10000)
                m.fit(X_train_scaled, y_train)
                mse = np.mean((y_v - m.predict(X_v_scaled)) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = alpha
        else:
            best_alpha = np.median(self.alphas)

        self.best_alpha = best_alpha

        # Refit on full in-sample data
        X_full_scaled = self.scaler.fit_transform(X_vals)
        self.model = sklearn.linear_model.Lasso(alpha=best_alpha, max_iter=10000)
        self.model.fit(X_full_scaled, y)

    def predict(self, X):
        X_sub = X[self.series] if self.series is not None else X
        X_vals = X_sub.values if hasattr(X_sub, "values") else np.asarray(X_sub)
        X_scaled = self.scaler.transform(X_vals)
        return self.model.predict(X_scaled)


class RidgeModel:
    """
    Ridge with internal time-series-safe hyperparameter tuning.
    
    Same 85/15 temporal split approach as LassoModel.
    """
    
    def __init__(self, alphas=None, series='yields'):
        if alphas is None:
            self.alphas = np.logspace(-5, 5, 30)  # 1e-5 to 1e5
        else:
            self.alphas = alphas
        self.series = series
        self.best_alpha_ = None
        self.model = None
    
    def fit(self, X, y):
        X_sub = X[[self.series]].values if self.series else X.values
        n = len(y)
        
        # 85/15 temporal split — no shuffling
        split = int(n * 0.85)
        
        # Need enough data in both splits
        if split < 10 or (n - split) < 3:
            self.best_alpha_ = np.median(self.alphas)
            self.model = sklearn.linear_model.Ridge(alpha=self.best_alpha_)
            self.model.fit(X_sub, y)
            return
        
        X_subtrain, X_val = X_sub[:split], X_sub[split:]
        y_subtrain, y_val = y[:split], y[split:]
        
        # Grid search
        best_alpha = self.alphas[0]
        best_mse = np.inf
        
        for alpha in self.alphas:
            m = sklearn.linear_model.Ridge(alpha=alpha)
            m.fit(X_subtrain, y_subtrain)
            preds = m.predict(X_val)
            mse = np.mean((y_val - preds) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        
        self.best_alpha_ = best_alpha

        # Refit on full training set with best alpha
        self.model = sklearn.linear_model.Ridge(alpha=self.best_alpha_)
        self.model.fit(X_sub, y)
    
    def predict(self, X):
        X_sub = X[[self.series]].values if self.series else X.values
        return self.model.predict(X_sub)



class RidgeModelScaled:
    """
    Ridge with internal time-series-safe hyperparameter tuning
    and feature standardization.

    Uses an 85/15 temporal split of the in-sample period:
    - first 85% for training
    - last 15% for validation

    Scaling is fit on the training split during tuning, then
    refit on the full in-sample data after the best alpha is chosen.
    """

    def __init__(self, alphas=None, series='yields'):
        self.alphas = np.logspace(-5, 5, 30) if alphas is None else alphas
        self.series = series
        self.best_alpha_ = None
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.model = None

    def fit(self, X, y):
        X_sub = X[self.series].values if self.series else X.values
        y = np.asarray(y)
        n = len(y)

        split = int(n * 0.85)

        # Fallback if there is not enough data for tuning
        if split < 10 or (n - split) < 3:
            self.best_alpha_ = np.median(self.alphas)
            X_scaled = self.scaler.fit_transform(X_sub)
            self.model = sklearn.linear_model.Ridge(alpha=self.best_alpha_)
            self.model.fit(X_scaled, y)
            return

        X_train, X_val = X_sub[:split], X_sub[split:]
        y_train, y_val = y[:split], y[split:]

        # Scale using training data only during tuning
        scaler_tune = sklearn.preprocessing.StandardScaler()
        X_train_scaled = scaler_tune.fit_transform(X_train)
        X_val_scaled = scaler_tune.transform(X_val)

        best_alpha = self.alphas[0]
        best_mse = np.inf

        for alpha in self.alphas:
            m = sklearn.linear_model.Ridge(alpha=alpha)
            m.fit(X_train_scaled, y_train)
            preds = m.predict(X_val_scaled)
            mse = np.mean((y_val - preds) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        self.best_alpha_ = best_alpha

        # Refit scaler and model on full in-sample data
        X_scaled = self.scaler.fit_transform(X_sub)
        self.model = sklearn.linear_model.Ridge(alpha=self.best_alpha_)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_sub = X[self.series].values if self.series else X.values
        X_scaled = self.scaler.transform(X_sub)
        return self.model.predict(X_scaled)


class GroupLassoModel:
    """
    Group Lasso with internal 85/15 temporal validation split for alpha tuning.
    
    Uses StandardScaler on features. The `groups` parameter is a per-feature
    integer array (e.g., [0,0,0,1,1,2,2,2,...]) that maps each column to its group.
    Internally converted to group_sizes list for skglm.GroupLasso.
    """
    
    def __init__(self, alphas=None, groups=None):
        if alphas is None:
            self.alphas = np.logspace(-4, 1, 30)
        else:
            self.alphas = alphas
        self.groups = groups  # per-feature integer label array
        self.model = None
        self.scaler = None
        self.best_alpha_ = None

    def _get_group_sizes(self):
        """Convert per-feature group labels to ordered group sizes list for skglm."""
        _, counts = np.unique(self.groups, return_counts=True)
        return counts.tolist()

    def fit(self, X, y):
        X_vals = X.values if hasattr(X, 'values') else np.array(X)
        y_vals = np.array(y).ravel()
        
        group_sizes = self._get_group_sizes()
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_vals)
        
        n = len(y_vals)
        split = int(n * 0.85)
        
        # Fallback if not enough data for a proper split
        if split < 10 or (n - split) < 3:
            self.best_alpha_ = np.median(self.alphas)
            self.model = skglm.GroupLasso(alpha=self.best_alpha_, groups=group_sizes)
            self.model.fit(X_scaled, y_vals)
            return
        
        X_subtrain, X_val = X_scaled[:split], X_scaled[split:]
        y_subtrain, y_val = y_vals[:split], y_vals[split:]
        
        # Grid search over alpha
        best_alpha = self.alphas[0]
        best_mse = np.inf
        
        for alpha in self.alphas:
            try:
                m = skglm.GroupLasso(alpha=alpha, groups=group_sizes)
                m.fit(X_subtrain, y_subtrain)
                preds = m.predict(X_val)
                mse = np.mean((y_val - preds) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = alpha
            except Exception:
                # Some alpha values may cause convergence issues; skip them
                continue
        
        self.best_alpha_ = best_alpha
        
        # Refit on full training set (subtrain + val) with best alpha
        self.model = skglm.GroupLasso(alpha=self.best_alpha_, groups=group_sizes)
        self.model.fit(X_scaled, y_vals)
    
    def predict(self, X):
        X_vals = X.values if hasattr(X, 'values') else np.array(X)
        X_scaled = self.scaler.transform(X_vals)
        return self.model.predict(X_scaled)


class BianchiElasticNet:

    """
    Faithful reimplementation of Bianchi's ElasticNet_Exog_Plain,
    adapted to work with our expanding_window API.

    Key design choices matching Bianchi exactly:
      - StandardScaler on training data, applied to test
      - PredefinedSplit: last 15% of training as single validation fold
      - ElasticNetCV with l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9]
      - max_iter=5000, random_state=42, n_jobs=-1
      - No refit on full training set after CV (ElasticNetCV uses its
        internal refit on the CV-selected alpha/l1_ratio)
    """

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
            All features (macro + yields concatenated) for the training window.
        y : np.array, shape (n_samples,)
            Target (single maturity excess return).
        """
        X_train = np.array(X) if not isinstance(X, np.ndarray) else X.copy()
        y_train = np.array(y).ravel()

        # Scale inputs for training
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Construct validation sample as last 15% of training sample
        N_train = int(np.round(X_train_scaled.shape[0] * 0.85))
        N_val = X_train_scaled.shape[0] - N_train
        test_fold = np.concatenate((
            np.full(N_train, -1),   # -1 = always in training
            np.full(N_val, 0)       #  0 = validation fold
        ))
        ps = PredefinedSplit(test_fold.tolist())

        # Fit ElasticNetCV — exactly as Bianchi
        self.model = sklearn.linear_model.ElasticNetCV(
            cv=ps,
            max_iter=20000,
            n_jobs=-1,
            l1_ratio=[.1, .3, .5, .7, .9],
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)

    def predict(self, X):
        """
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (1, n_features) or (n, n_features)
        
        Returns
        -------
        np.array of predictions
        """
        X_test = np.array(X) if not isinstance(X, np.ndarray) else X.copy()
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)



import numpy as np
import sklearn.decomposition
import sklearn.linear_model
import sklearn.preprocessing


import numpy as np
import sklearn.decomposition
import sklearn.linear_model
import sklearn.preprocessing


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
        self.alphas = alphas if alphas is not None else np.logspace(-5, 1, 30)
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
            self.model = sklearn.linear_model.Lasso(alpha=self.best_alpha_, max_iter=10000)
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
            m = sklearn.linear_model.Lasso(alpha=alpha, max_iter=10000)
            m.fit(X_train_scaled, y_train)
            mse = np.mean((y_val_int - m.predict(X_val_scaled)) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        self.best_alpha_ = best_alpha

        features_full = self._fit_feature_pipeline(X)
        features_full_scaled = self.scaler_features.fit_transform(features_full)

        self.model = sklearn.linear_model.Lasso(alpha=self.best_alpha_, max_iter=10000)
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
        self.alphas = alphas if alphas is not None else np.logspace(-5, 5, 30)
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

import numpy as np
import sklearn.decomposition
import sklearn.linear_model
import sklearn.preprocessing


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
        self.alphas = alphas if alphas is not None else np.logspace(-5, 1, 30)
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
            self.model = sklearn.linear_model.Lasso(alpha=self.best_alpha_, max_iter=10000)
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
            m = sklearn.linear_model.Lasso(alpha=alpha, max_iter=10000)
            m.fit(X_train_scaled, y_train)
            mse = np.mean((y_val_int - m.predict(X_val_scaled)) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        self.best_alpha_ = best_alpha

        # Refit everything on full in-sample window
        features_full = self._fit_feature_pipeline(X)
        features_full_scaled = self.scaler_features.fit_transform(features_full)

        self.model = sklearn.linear_model.Lasso(alpha=self.best_alpha_, max_iter=10000)
        self.model.fit(features_full_scaled, y)

    def predict(self, X):
        features = self._transform_features(X)
        features_scaled = self.scaler_features.transform(features)
        return self.model.predict(features_scaled)

import numpy as np
import sklearn.decomposition
import sklearn.linear_model
import sklearn.preprocessing


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
        self.alphas = alphas if alphas is not None else np.logspace(-5, 5, 30)
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




import numpy as np
import sklearn.linear_model
import sklearn.preprocessing


class LassoRawMacroFwdDirectModel:
    """
    Lasso on:
      - full scaled macro dataset
      - all forward regressors directly

    No macro PCA / LN compression.
    Proper 85/15 temporal tuning split.
    """

    def __init__(self, alphas=None, macro_series='fred', forward_series='forward'):
        self.alphas = alphas if alphas is not None else np.logspace(-5, 1, 30)
        self.macro_series = macro_series
        self.forward_series = forward_series

        self.scaler_macro = sklearn.preprocessing.StandardScaler()
        self.scaler_features = sklearn.preprocessing.StandardScaler()

        self.best_alpha_ = None
        self.model = None

    def _fit_feature_pipeline(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.fit_transform(macro)

        forward = X[self.forward_series].values
        return np.concatenate([macro_scaled, forward], axis=1)

    def _transform_features(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.transform(macro)

        forward = X[self.forward_series].values
        return np.concatenate([macro_scaled, forward], axis=1)

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y)
        n = len(y)
        split = int(n * 0.85)

        if len(y) < 10 or len(y[split:]) < 3:
            features = self._fit_feature_pipeline(X)
            features_scaled = self.scaler_features.fit_transform(features)
            self.best_alpha_ = np.median(self.alphas)
            self.model = sklearn.linear_model.Lasso(alpha=self.best_alpha_, max_iter=10000)
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
            m = sklearn.linear_model.Lasso(alpha=alpha, max_iter=10000)
            m.fit(X_train_scaled, y_train)
            mse = np.mean((y_val_int - m.predict(X_val_scaled)) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        self.best_alpha_ = best_alpha

        features_full = self._fit_feature_pipeline(X)
        features_full_scaled = self.scaler_features.fit_transform(features_full)

        self.model = sklearn.linear_model.Lasso(alpha=self.best_alpha_, max_iter=10000)
        self.model.fit(features_full_scaled, y)

    def predict(self, X):
        features = self._transform_features(X)
        features_scaled = self.scaler_features.transform(features)
        return self.model.predict(features_scaled)


class RidgeRawMacroFwdDirectModel:
    """
    Ridge on:
      - full scaled macro dataset
      - all forward regressors directly

    No macro PCA / LN compression.
    Proper 85/15 temporal tuning split.
    """

    def __init__(self, alphas=None, macro_series='fred', forward_series='forward'):
        self.alphas = alphas if alphas is not None else np.logspace(-5, 5, 30)
        self.macro_series = macro_series
        self.forward_series = forward_series

        self.scaler_macro = sklearn.preprocessing.StandardScaler()
        self.scaler_features = sklearn.preprocessing.StandardScaler()

        self.best_alpha_ = None
        self.model = None

    def _fit_feature_pipeline(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.fit_transform(macro)

        forward = X[self.forward_series].values
        return np.concatenate([macro_scaled, forward], axis=1)

    def _transform_features(self, X):
        macro = X[self.macro_series]
        macro_scaled = self.scaler_macro.transform(macro)

        forward = X[self.forward_series].values
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
        self.alphas = alphas if alphas is not None else np.logspace(-5, 1, 30)
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
            self.model = sklearn.linear_model.Lasso(alpha=self.best_alpha_, max_iter=10000)
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
            m = sklearn.linear_model.Lasso(alpha=alpha, max_iter=10000)
            m.fit(X_train_scaled, y_train)
            mse = np.mean((y_val_int - m.predict(X_val_scaled)) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        self.best_alpha_ = best_alpha

        features_full = self._fit_feature_pipeline(X)
        features_full_scaled = self.scaler_features.fit_transform(features_full)

        self.model = sklearn.linear_model.Lasso(alpha=self.best_alpha_, max_iter=10000)
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
        self.alphas = alphas if alphas is not None else np.logspace(-5, 5, 30)
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