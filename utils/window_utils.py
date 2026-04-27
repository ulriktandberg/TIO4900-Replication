import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm.auto import tqdm

def _fit_window_imputer(X_train, strategy, all_missing_fill_value):
    if strategy not in {"median"}:
        raise ValueError(f"Unsupported impute_strategy: {strategy}")

    if isinstance(X_train, pd.DataFrame):
        if strategy == "median":
            fill_values = X_train.median(axis=0, skipna=True)
        fill_values = fill_values.fillna(all_missing_fill_value)
        return fill_values

    X_arr = np.asarray(X_train, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    fill_values = np.nanmedian(X_arr, axis=0)
    fill_values = np.where(np.isnan(fill_values), all_missing_fill_value, fill_values)
    return fill_values


def _availability_timing_mask(X_values):
    if isinstance(X_values, pd.DataFrame):
        mask = pd.DataFrame(False, index=X_values.index, columns=X_values.columns)
        row_index = X_values.index
        for col in X_values.columns:
            series = X_values[col]
            na = series.isna()
            if not na.any():
                continue
            valid_idx = row_index[series.notna()]
            if len(valid_idx) == 0:
                mask[col] = na
                continue
            first_valid = valid_idx[0]
            last_valid = valid_idx[-1]
            mask[col] = na & ((row_index < first_valid) | (row_index > last_valid))
        return mask

    X_arr = np.asarray(X_values, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    mask = np.zeros_like(X_arr, dtype=bool)
    for j in range(X_arr.shape[1]):
        col = X_arr[:, j]
        valid_idx = np.flatnonzero(~np.isnan(col))
        if len(valid_idx) == 0:
            mask[:, j] = np.isnan(col)
            continue
        first_valid = valid_idx[0]
        last_valid = valid_idx[-1]
        idx = np.arange(len(col))
        mask[:, j] = np.isnan(col) & ((idx < first_valid) | (idx > last_valid))
    return mask


def _apply_window_imputer(X_values, fill_values, preserve_mask=None):
    if isinstance(X_values, pd.DataFrame):
        filled = X_values.fillna(fill_values)
        if preserve_mask is None:
            return filled
        return filled.mask(preserve_mask)

    X_arr = np.asarray(X_values, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    out = X_arr.copy()
    nan_rows, nan_cols = np.where(np.isnan(out))
    out[nan_rows, nan_cols] = fill_values[nan_cols]
    if preserve_mask is not None:
        out[np.asarray(preserve_mask, dtype=bool)] = np.nan
    return out


def _select_train_available_columns(X_train):
    if isinstance(X_train, pd.DataFrame):
        keep = X_train.notna().all(axis=0)
        return X_train.columns[keep]

    X_arr = np.asarray(X_train, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    return np.flatnonzero(~np.isnan(X_arr).any(axis=0))


def _apply_column_selection(X_values, selected_columns):
    if isinstance(X_values, pd.DataFrame):
        return X_values.loc[:, selected_columns]

    X_arr = np.asarray(X_values, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    return X_arr[:, selected_columns]


def _carry_forward_latest(X_values):
    if isinstance(X_values, pd.DataFrame):
        return X_values.ffill()

    X_arr = np.asarray(X_values, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    out = X_arr.copy()
    for j in range(out.shape[1]):
        last = np.nan
        for i in range(out.shape[0]):
            if np.isnan(out[i, j]):
                out[i, j] = last
            else:
                last = out[i, j]
    return out


def _realtime_feature_frame(X, panel_transformed, t):
    X_current = X.iloc[: t + 1].copy()
    macro = panel_transformed.reindex(X_current.index)

    if isinstance(X_current.columns, pd.MultiIndex):
        top_level = X_current.columns.get_level_values(0)
        if "fred" not in top_level:
            raise ValueError(
                "realtime=True requires a 'fred' feature block in the first column level."
            )
        fred_cols = X_current.loc[:, top_level == "fred"].columns
        fred_series = list(fred_cols.get_level_values(-1))
        missing = [name for name in dict.fromkeys(fred_series) if name not in macro.columns]
        if missing:
            raise ValueError(
                "Forecast-vintage panel is missing required macro columns: "
                f"{missing}"
            )
        X_current.loc[:, fred_cols] = macro.reindex(columns=fred_series).to_numpy()
        return X_current

    missing = [name for name in X_current.columns if name not in macro.columns]
    if missing:
        raise ValueError(
            "realtime=True with a flat-column design matrix expects macro-only columns. "
            f"Missing forecast-vintage columns: {missing}"
        )
    X_current.loc[:, X_current.columns] = macro.reindex(columns=X_current.columns).to_numpy()
    return X_current


def expanding_window(model_class, X, y, dates, oos_start,
                     gap=0, refit_freq=1, coef_callback=None,
                     save_callback=None,
                     progress=True,
                     tqdm_position=None,
                     tqdm_desc="expanding window",
                     tqdm_leave=False,
                     realtime=False,
                     realtime_store=None,
                     impute_strategy=None,
                     all_missing_fill_value=0.0,
                     preserve_availability_timing=True,
                     drop_unavailable_columns=True,
                     carry_forward_latest=True):
    """
    Unified Forecasting Engine.
    """
    if y.ndim == 1:
        y_forecast = np.full(len(y), np.nan)
    else:
        y_forecast = np.full(y.shape, np.nan)

    oos_indices = np.where(dates >= oos_start)[0]
    model = None
    realtime_panel_cache = {}
    selected_columns = None

    if realtime:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("realtime=True requires X to be a pandas DataFrame.")
        if realtime_store is None:
            from utils.forecast_vintages import ForecastVintageMacroStore

            realtime_store = ForecastVintageMacroStore()

    iterator = oos_indices
    if progress:
        iterator = tqdm(
            oos_indices,
            desc=tqdm_desc,
            position=tqdm_position,
            leave=tqdm_leave,
            dynamic_ncols=True
        )

    for i, t in enumerate(iterator):
        X_current = X
        realtime_panel = None
        if realtime:
            forecast_date = pd.Timestamp(dates[t])
            realtime_panel = realtime_panel_cache.get(forecast_date)
            if realtime_panel is None:
                realtime_panel = realtime_store.panel_for_forecast_date(
                    forecast_date,
                    start=dates[0],
                    end=forecast_date,
                )
                realtime_panel_cache[forecast_date] = realtime_panel
            X_current = _realtime_feature_frame(X, realtime_panel.transformed, t)

        if i % refit_freq == 0:
            if model is None:
                current_model = deepcopy(model_class)
            else:
                current_model = deepcopy(model)

            train_end = t - gap
            X_model = X_current
            if impute_strategy is not None:
                fill_values = _fit_window_imputer(
                    X_current.iloc[:train_end],
                    strategy=impute_strategy,
                    all_missing_fill_value=all_missing_fill_value,
                )
                preserve_mask = (
                    _availability_timing_mask(X_current)
                    if preserve_availability_timing else None
                )
                X_model = _apply_window_imputer(X_current, fill_values, preserve_mask=preserve_mask)

            if drop_unavailable_columns:
                selected_columns = _select_train_available_columns(X_model.iloc[:train_end])
                X_model = _apply_column_selection(X_model, selected_columns)
            else:
                selected_columns = None

            if carry_forward_latest:
                X_model = _carry_forward_latest(X_model)

            X_train, y_train = X_model.iloc[:train_end], y[:train_end]
            current_model.fit(X_train, y_train)
            model = current_model

            if save_callback is not None:
                save_callback(
                    model=current_model,
                    refit_i=i,
                    t_index=t,
                    date_value=dates[t],
                    realtime_panel=realtime_panel,
                    impute_strategy=impute_strategy,
                    selected_columns=selected_columns,
                )

            if coef_callback is not None and hasattr(model, "model"):
                try:
                    coef_callback(current_model.model.coef_)
                except AttributeError:
                    print("Warning: Model does not have .model.coef_ attribute for callback.")
        else:
            X_model = X_current
            if impute_strategy is not None:
                fill_values = _fit_window_imputer(
                    X_current.iloc[: t - gap],
                    strategy=impute_strategy,
                    all_missing_fill_value=all_missing_fill_value,
                )
                preserve_mask = (
                    _availability_timing_mask(X_current)
                    if preserve_availability_timing else None
                )
                X_model = _apply_window_imputer(X_current, fill_values, preserve_mask=preserve_mask)
            if drop_unavailable_columns and selected_columns is not None:
                X_model = _apply_column_selection(X_model, selected_columns)
            if carry_forward_latest:
                X_model = _carry_forward_latest(X_model)

        # Use sequence architectures' own prediction method if available
        if hasattr(model, "predict_at"):
            pred = model.predict_at(X_model, t)
        else:
            pred = model.predict(X_model.iloc[[t]])

        if y.ndim == 1:
            y_forecast[t] = pred[-1] if isinstance(pred, np.ndarray) else pred
        else:
            y_forecast[t, :] = pred.flatten()

    return y_forecast

def oos_r2(y_true, y_forecast, benchmark='hist_mean', gap=0, **kwargs):
    """
    Campbell-Thompson OOS R^2 with selectable benchmark.
    Supports single-output (T,) or multi-output (T, n_outputs).
    For multi-output, returns an array of R^2 values, one per output.
    
    Parameters
    ----------
    y_true : np.array
    y_forecast : np.array
    benchmark : str, one of:
        'hist_mean'   - expanding-window historical mean (default, Campbell-Thompson)
        'ewma'        - exponentially weighted moving average (specify `halflife` in kwargs)
        'rolling'     - rolling window mean (specify `window` in kwargs)
        'ar1'         - expanding-window AR(1)
        'zero'        - constant zero forecast (pure EH null: no excess return, GKX benchmark)
    **kwargs : additional parameters for the benchmark
        halflife : int, EWMA half-life in periods (default 60)
        window   : int, rolling mean window in periods (default 60)
    
    Returns
    -------
    float : OOS R^2
    """
    if y_true.ndim == 2:
        n_outputs = y_true.shape[1]
        return np.array([
            oos_r2(y_true[:, i], y_forecast[:, i], benchmark=benchmark, **kwargs)
            for i in range(n_outputs)
        ])

    valid = ~np.isnan(y_forecast)
    
    if benchmark == 'hist_mean':
        # At time t, with gap=g, only y[0],...,y[t-g-1] are realized
        y_bench = np.full(len(y_true), np.nan)
        for t in range(1, len(y_true)):
            end = t - gap if gap > 0 else t
            if end < 1:
                continue
            y_bench[t] = np.mean(y_true[:end])
    
    elif benchmark == 'ewma':
        halflife = kwargs.get('halflife', 60)
        alpha = 1 - np.exp(-np.log(2) / halflife)
        y_bench = np.full(len(y_true), np.nan)
        ewma = y_true[0]
        for t in range(1, len(y_true)):
            y_bench[t] = ewma  # forecast made before observing y[t]
            ewma = alpha * y_true[t] + (1 - alpha) * ewma
    
    elif benchmark == 'rolling':
        window = kwargs.get('window', 60)
        y_bench = np.full(len(y_true), np.nan)
        for t in range(1, len(y_true)):
            start = max(0, t - window)
            y_bench[t] = np.mean(y_true[start:t])
    
    elif benchmark == 'ar1':
        y_bench = np.full(len(y_true), np.nan)
        for t in range(2, len(y_true)):
            y_train = y_true[:t]
            # OLS: y_t = a + b * y_{t-1}
            x = np.column_stack([np.ones(t - 1), y_train[:-1]])
            coeffs = np.linalg.lstsq(x, y_train[1:], rcond=None)[0]
            y_bench[t] = coeffs[0] + coeffs[1] * y_true[t - 1]
    
    elif benchmark == 'zero':
        y_bench = np.zeros(len(y_true))
    
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    
    # Mask benchmark NaNs as well
    valid = valid & ~np.isnan(y_bench)
    
    ss_res = np.nansum((y_true[valid] - y_forecast[valid]) ** 2)
    ss_tot = np.nansum((y_true[valid] - y_bench[valid]) ** 2)
    
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot
