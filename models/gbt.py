import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except ModuleNotFoundError:
    xgb = None

try:
    import lightgbm as lgb
except ModuleNotFoundError:
    lgb = None


__all__ = [
    "XGB_ARCH_GRID",
    "LGBM_ARCH_GRID",
    "XGBoostModel",
    "LightGBMModel",
]


# ── Suggested architecture grids (for CV in notebooks) ──────────────────────
#
# These are *plausible* tree specifications in the spirit of Bianchi–Büchner–
# Tamoni (shallow → medium → deeper trees, more estimators + stronger
# regularization). They provide a small search space for expanding-window
# tuning without adding a large amount of runtime.

XGB_ARCH_GRID = [
    {
        "name": "xgb_shallow",
        "max_depth": 2,
        "n_estimators": 200,
        "learning_rate": 0.10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "name": "xgb_medium",
        "max_depth": 3,
        "n_estimators": 400,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "name": "xgb_deep",
        "max_depth": 4,
        "n_estimators": 600,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.0,
        "reg_lambda": 2.0,
    },
]


LGBM_ARCH_GRID = [
    {
        "name": "lgbm_shallow",
        "num_leaves": 15,
        "max_depth": 3,
        "n_estimators": 200,
        "learning_rate": 0.10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_data_in_leaf": 20,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    },
    {
        "name": "lgbm_medium",
        "num_leaves": 31,
        "max_depth": -1,
        "n_estimators": 400,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_data_in_leaf": 30,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    },
    {
        "name": "lgbm_deep",
        "num_leaves": 63,
        "max_depth": -1,
        "n_estimators": 600,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_data_in_leaf": 50,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
]


def _strip_candidate_name(spec):
    return {k: v for k, v in spec.items() if k != "name"}


def _dedupe_candidates(specs):
    deduped = []
    seen = set()
    for spec in specs:
        key = tuple(sorted(spec.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return deduped


class _MedianImputer:
    """Small numpy-based median imputer to keep preprocessing leakage-safe."""

    def __init__(self):
        self.fill_values_ = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        observed_mask = ~np.isnan(X_arr)
        has_observed = observed_mask.any(axis=0)
        if not np.all(has_observed):
            missing_cols = np.where(~has_observed)[0].tolist()
            raise ValueError(
                "Median imputation requires at least one observed value per feature "
                f"column in the training window. Empty columns: {missing_cols}"
            )

        fill_values = np.empty(X_arr.shape[1], dtype=np.float64)
        fill_values[has_observed] = np.nanmedian(X_arr[:, has_observed], axis=0)
        self.fill_values_ = fill_values
        return self

    def transform(self, X):
        if self.fill_values_ is None:
            raise ValueError("Imputer has not been fit yet.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if not np.isnan(X_arr).any():
            return X_arr

        X_out = X_arr.copy()
        nan_rows, nan_cols = np.where(np.isnan(X_out))
        X_out[nan_rows, nan_cols] = self.fill_values_[nan_cols]
        return X_out


class _BaseGBTModel:
    _estimator_name = ""
    _package_name = ""
    _estimator_cls = None
    _default_arch_grid = ()

    def __init__(self, features=None, arch_grid=None, tune_every=60, impute_strategy=None):
        if impute_strategy not in {None, "median"}:
            raise ValueError("impute_strategy must be None or 'median'.")

        self.features = features
        self.arch_grid = arch_grid
        self.tune_every = tune_every
        self.impute_strategy = impute_strategy

        self.model = None
        self.best_params_ = None
        self._fit_calls = 0
        self._imputers = {}
        self._scalers = {}
        self._pcas = {}
        self._last_val_loss = np.nan

        # Populated by subclasses.
        self.params = {}
        self.fixed_params = {}

    def _ensure_estimator_available(self):
        if self._estimator_cls is not None:
            return
        raise ImportError(
            f"{self._estimator_name} is not installed. "
            f"Install `{self._package_name}` to use this model."
        )

    def _should_tune(self):
        if self.best_params_ is None:
            return True
        if self.tune_every is None or self.tune_every <= 1:
            return True
        return (self._fit_calls % self.tune_every) == 0

    def _resolve_candidate_specs(self):
        candidates = [self.params.copy()]
        arch_grid = self._default_arch_grid if self.arch_grid is None else self.arch_grid
        candidates.extend(_strip_candidate_name(spec) for spec in arch_grid)
        return _dedupe_candidates(candidates)

    def _normalize_target(self, y):
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 2:
            if y_arr.shape[1] != 1:
                raise ValueError(
                    f"{self._estimator_name} wrappers currently support a single target series."
                )
            y_arr = y_arr[:, 0]
        return y_arr.ravel()

    def _extract_group_values(self, X, group):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame when `features` is specified.")

        try:
            block = X[group]
        except KeyError as exc:
            raise ValueError(f"Expected '{group}' block in X columns.") from exc

        arr = block.values if hasattr(block, "values") else np.asarray(block)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr.astype(np.float64)

    def _fit_imputer(self, X_group, key, state):
        if self.impute_strategy is None:
            return X_group

        imputer = _MedianImputer().fit(X_group)
        state["imputers"][key] = imputer
        return imputer.transform(X_group)

    def _transform_with_imputer(self, X_group, key, state):
        if self.impute_strategy is None:
            return X_group

        imputer = state["imputers"].get(key)
        if imputer is None:
            raise ValueError(f"Missing imputer state for feature block '{key}'.")
        return imputer.transform(X_group)

    def _fit_feature_state(self, X):
        if self.features is None:
            state = {"imputers": {}, "scalers": {}, "pcas": {}}
            if isinstance(X, pd.DataFrame):
                X_arr = X.values.astype(np.float64)
            else:
                X_arr = np.asarray(X, dtype=np.float64)
            X_arr = self._fit_imputer(X_arr, None, state)
            return X_arr, state

        blocks = []
        state = {"imputers": {}, "scalers": {}, "pcas": {}}

        for group, cfg in self.features.items():
            X_group = self._extract_group_values(X, group)
            X_group = self._fit_imputer(X_group, group, state)
            method = cfg.get("method", "raw")

            if method == "pca":
                n_comp = min(int(cfg["n_components"]), X_group.shape[0], X_group.shape[1])
                if n_comp < 1:
                    raise ValueError(f"Group '{group}' has too few observations for PCA.")

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_group)
                pca = PCA(n_components=n_comp)
                X_out = pca.fit_transform(X_scaled)

                state["scalers"][group] = scaler
                state["pcas"][group] = pca
                blocks.append(X_out)
            elif method == "raw":
                blocks.append(X_group)
            else:
                raise ValueError(f"Unknown method '{method}' for group '{group}'")

        if not blocks:
            raise ValueError("No feature blocks were selected for the model.")

        return np.hstack(blocks), state

    def _transform_features(self, X, state):
        if self.features is None:
            if isinstance(X, pd.DataFrame):
                X_arr = X.values.astype(np.float64)
            else:
                X_arr = np.asarray(X, dtype=np.float64)
            return self._transform_with_imputer(X_arr, None, state)

        blocks = []
        for group, cfg in self.features.items():
            X_group = self._extract_group_values(X, group)
            X_group = self._transform_with_imputer(X_group, group, state)
            method = cfg.get("method", "raw")

            if method == "pca":
                scaler = state["scalers"][group]
                pca = state["pcas"][group]
                X_scaled = scaler.transform(X_group)
                blocks.append(pca.transform(X_scaled))
            elif method == "raw":
                blocks.append(X_group)
            else:
                raise ValueError(f"Unknown method '{method}' for group '{group}'")

        if not blocks:
            raise ValueError("No feature blocks were selected for the model.")

        return np.hstack(blocks)

    def _make_estimator(self, params):
        self._ensure_estimator_available()
        estimator_params = self.fixed_params.copy()
        estimator_params.update(self.params)
        estimator_params.update(params)
        return self._estimator_cls(**estimator_params)

    def fit(self, X, y):
        y_arr = self._normalize_target(y)
        n_obs = len(y_arr)
        split = int(n_obs * 0.85)
        candidate_specs = self._resolve_candidate_specs()

        should_tune = self._should_tune() and split >= 10 and (n_obs - split) >= 3

        if should_tune:
            X_subtrain = X.iloc[:split] if hasattr(X, "iloc") else X[:split]
            X_val = X.iloc[split:] if hasattr(X, "iloc") else X[split:]
            y_subtrain = y_arr[:split]
            y_val = y_arr[split:]

            X_subtrain_np, tune_state = self._fit_feature_state(X_subtrain)
            X_val_np = self._transform_features(X_val, tune_state)

            best_mse = np.inf
            best_params = candidate_specs[0]

            for params in candidate_specs:
                candidate = self._make_estimator(params)
                candidate.fit(X_subtrain_np, y_subtrain)
                preds = np.asarray(candidate.predict(X_val_np), dtype=np.float64).ravel()
                mse = np.mean((y_val - preds) ** 2)

                if mse < best_mse:
                    best_mse = mse
                    best_params = params

            self.best_params_ = best_params.copy()
            self._last_val_loss = float(best_mse)
        else:
            if self.best_params_ is None:
                self.best_params_ = candidate_specs[0].copy()
            self._last_val_loss = np.nan

        X_full_np, feature_state = self._fit_feature_state(X)
        self._imputers = feature_state["imputers"]
        self._scalers = feature_state["scalers"]
        self._pcas = feature_state["pcas"]

        self.model = self._make_estimator(self.best_params_)
        self.model.fit(X_full_np, y_arr)
        self._fit_calls += 1

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been fit yet.")

        feature_state = {"imputers": self._imputers, "scalers": self._scalers, "pcas": self._pcas}
        X_np = self._transform_features(X, feature_state)
        return self.model.predict(X_np)


class XGBoostModel(_BaseGBTModel):
    """XGBoost regressor with leakage-safe block feature engineering."""

    _estimator_name = "XGBoost"
    _package_name = "xgboost"
    _estimator_cls = xgb.XGBRegressor if xgb is not None else None
    _default_arch_grid = XGB_ARCH_GRID

    def __init__(
        self,
        features=None,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        arch_grid=None,
        tune_every=60,
        impute_strategy=None,
    ):
        super().__init__(
            features=features,
            arch_grid=arch_grid,
            tune_every=tune_every,
            impute_strategy=impute_strategy,
        )
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
        }
        self.fixed_params = {
            "random_state": random_state,
            "objective": "reg:squarederror",
        }


class LightGBMModel(_BaseGBTModel):
    """LightGBM regressor with leakage-safe block feature engineering."""

    _estimator_name = "LightGBM"
    _package_name = "lightgbm"
    _estimator_cls = lgb.LGBMRegressor if lgb is not None else None
    _default_arch_grid = LGBM_ARCH_GRID

    def __init__(
        self,
        features=None,
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.05,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        min_data_in_leaf=30,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        arch_grid=None,
        tune_every=60,
        impute_strategy=None,
    ):
        super().__init__(
            features=features,
            arch_grid=arch_grid,
            tune_every=tune_every,
            impute_strategy=impute_strategy,
        )
        self.params = {
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_data_in_leaf": min_data_in_leaf,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
        }
        self.fixed_params = {
            "random_state": random_state,
            "objective": "regression",
        }
