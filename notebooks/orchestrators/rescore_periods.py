"""Re-score saved per-date forecasts into per-period OOS R^2 / RSZ significance.

Reads results/linear_forecasts.csv (the append-only per-date forecast log) and
recomputes the eval-period metrics WITHOUT re-running the expanding window.

Window fix: a standalone END_DATE=cut load drops its final `holding` months
(their forward return isn't observable yet), so the last realizable OOS month for
a period ending at `cut` is `cut - holding`. We subtract it here so that a period
derived from a longer run reproduces the standalone run (e.g. 2018-from-2025 no
longer appends Dec-2018, whose lone L&N outlier distorted the non-winsorized R^2).
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as tstat

HERE = os.path.dirname(os.path.abspath(__file__))
FORECASTS_CSV = os.path.join(HERE, "results", "linear_forecasts.csv")
OUT_CSV = os.path.join(HERE, "results", "linear_runs_rescored.csv")

EVAL_PERIODS = {
    "2018": pd.Timestamp("2018-12-31"),
    "2025": pd.Timestamp("2025-06-30"),
}


def oos_r2(y_true, y_forecast, gap=0):
    y_true = np.asarray(y_true, dtype=float)
    y_forecast = np.asarray(y_forecast, dtype=float)
    valid = ~np.isnan(y_forecast)
    y_bench = np.full(len(y_true), np.nan)
    for t in range(1, len(y_true)):
        end = t - gap if gap > 0 else t
        if end < 1:
            continue
        y_bench[t] = np.mean(y_true[:end])
    valid = valid & ~np.isnan(y_bench)
    ss_res = np.nansum((y_true[valid] - y_forecast[valid]) ** 2)
    ss_tot = np.nansum((y_true[valid] - y_bench[valid]) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def rsz_signif(y_true, y_forecast, gap=0):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_forecast = np.asarray(y_forecast, dtype=float).ravel()
    y_condmean = np.full_like(y_true, np.nan, dtype=float)
    for t in range(1, len(y_true)):
        end = t - gap if gap > 0 else t
        if end < 1:
            continue
        y_condmean[t] = np.mean(y_true[:end])
    y_condmean[np.isnan(y_forecast)] = np.nan
    f = (
        np.square(y_true - y_condmean)
        - np.square(y_true - y_forecast)
        + np.square(y_condmean - y_forecast)
    )
    x = np.ones(np.shape(f))
    model = sm.OLS(f, x, missing="drop", hasconst=True)
    results = model.fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    return 1 - tstat.cdf(results.tvalues[0], results.nobs - 1)


def main():
    fc = pd.read_csv(FORECASTS_CSV, parse_dates=["date"])

    group_keys = [
        "run_id", "model", "maturity", "winsorized", "frequency", "gap",
        "dataset", "revised_macro_static_X", "expanding_window_realtime",
    ]
    rows = []
    for keys, g in fc.groupby(group_keys, dropna=False):
        rec = dict(zip(group_keys, keys))
        g = g.sort_values("date")
        dates = pd.to_datetime(pd.Index(g["date"].values))
        y = g["y_true"].to_numpy(dtype=float)
        yhat = g["y_hat"].to_numpy(dtype=float)
        gap = int(rec["gap"]) if pd.notna(rec["gap"]) else 0
        holding = 12 if str(rec["frequency"]) == "annual" else 1
        run_last = dates.max()

        for label, cut in EVAL_PERIODS.items():
            eff = pd.Timestamp(cut) - pd.offsets.MonthEnd(holding)
            # Only score a period the run actually reaches.
            if run_last < eff:
                continue
            idx = int((dates <= eff).sum())
            if idx < 2:
                continue
            ys, yhs = y[:idx], yhat[:idx]
            if np.isfinite(yhs).sum() < 2:
                continue
            r2 = float(oos_r2(ys, yhs, gap=gap))
            try:
                pval = float(rsz_signif(ys, yhs, gap=gap))
            except Exception:
                pval = np.nan
            rows.append({
                **rec,
                "eval_period": label,
                "eval_end_nominal": pd.Timestamp(cut).strftime("%Y-%m-%d"),
                "eval_end_effective": eff.strftime("%Y-%m-%d"),
                "n_oos": int(np.isfinite(yhs).sum()),
                "r2_oos": r2,
                "rsz_pvalue": pval,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} rescored rows to {OUT_CSV}\n")

    # Focused view: monthly L&N, both periods, winsor vs non-winsor.
    view = out[(out["frequency"] == "monthly") & (out["model"] == "L&N (2009)")]
    cols = ["run_id", "maturity", "winsorized", "eval_period",
            "eval_end_effective", "n_oos", "r2_oos", "rsz_pvalue"]
    if not view.empty:
        view = view.sort_values(["run_id", "maturity", "winsorized", "eval_period"])
        print(view[cols].to_string(index=False))


if __name__ == "__main__":
    main()
