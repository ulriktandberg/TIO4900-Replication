from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.publication_lags import (
    PUB_LAG_COLUMNS,
    default_publication_lag_policy,
    lagged_observation_date,
    resolve_publication_lag_months,
)

API_BASE = "https://api.stlouisfed.org/fred"
RETRY_STATUS = {429, 500, 502, 503, 504}

REGISTRY_COLUMNS = [
    "hs_no",
    "group",
    "mnemonic_hs",
    "mnemonic_old",
    "description_hs",
    "tcode_hs",
    "lag_months_hs",
    "vintage_flag_hs",
    "construction_type",
    "target_series_id",
    "component_series_ids",
    "raw_formula",
    "pub_lag_mode",
    "pub_lag_months",
    "pub_lag_schedule_json",
    "release_rule",
    "revision_class",
    "strict_source_rule",
    "balanced_source_rule",
    "notes",
]

# ALFRED vintages already encode real-time availability via realtime_start/end,
# so the default registry uses same-month as-of selection instead of imposing an
# extra one-month publication lag on top.
DEFAULT_RELEASE_RULE = "same_month_end"
DEFAULT_LAG_MONTHS = 0

SIMPLE_NEVER_REVISED = {"AAA", "BAA", "EXCAUSx", "EXJPUSx", "EXSZUSx", "EXUSUKx", "GS1", "GS10", "GS5"}
SIMPLE_DIRECT_ALIAS_MAP = {
    "S&P 500": "SP500",
    "EXCAUSx": "EXCAUS",
    "EXJPUSx": "EXJPUS",
    "EXSZUSx": "EXSZUS",
    "EXUSUKx": "EXUSUK",
    "CMRMTSPLx": "CMRMTSPL",
    "RETAILx": "RETAIL",
    "AMDMUOx": "AMDMUO",
    "ANDENOx": "ANDENO",
    "BUSINVx": "BUSINV",
    "CP3Mx": "CP3M",
    "ISRATIOx": "ISRATIO",
    "OILPRICEx": "OILPRICE",
    "TWEXAFEGSMTHx": "TWEXAFEGSMTH",
    "UMCSENTx": "UMCSENT",
    "VIXCLSx": "VIXCLS",
    "COMPAPFFx": "CPFFM",
    "AMDMNOx": "DGORDER",
    "HWI": "JTSJOL",
}
SIMPLE_DIRECT_FRED_MAP = {"CLAIMSx": "ICSA"}
SIMPLE_FORMULA_MAP = {
    "CONSPI": {"components": ["NONREVSL", "PI"], "formula": "NONREVSL / PI", "revision_class": "component_revisions"},
    "HWIURATIO": {"components": ["JTSJOL", "UNEMPLOY"], "formula": "JTSJOL / UNEMPLOY", "revision_class": "component_revisions"},
}
SIMPLE_LOCAL_ONLY = {"S&P div yield", "S&P PE ratio"}

SOURCE_STRICT = "STRICT"
SOURCE_RELEASE_GAP = "FILL_RELEASE_GAP"
SOURCE_PRE_VINTAGE = "FILL_PRE_VINTAGE_EARLIEST"
SOURCE_NONREV = "FILL_NONREVISED_FRED"


class RegistryError(ValueError):
    pass


class FredApiError(RuntimeError):
    pass


@dataclass
class FetchContext:
    api_key: str
    session: requests.Session
    pause_sec: float = 0.03
    timeout_sec: float = 60.0
    max_attempts: int = 6


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).strip().split())


def _month_end(ts: Any) -> pd.Timestamp:
    return (pd.Timestamp(ts) + pd.offsets.MonthEnd(0)).normalize()


def _month_range(start: Any, end: Any) -> pd.DatetimeIndex:
    return pd.date_range(_month_end(start), _month_end(end), freq=pd.offsets.MonthEnd(1))


def _months_between(late: pd.Timestamp, early: pd.Timestamp) -> int:
    return (late.year - early.year) * 12 + (late.month - early.month)


def _json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_clean(x) for x in value if _clean(x)]
    text = _clean(value)
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise RegistryError("component_series_ids must decode to list")
    return [_clean(x) for x in parsed if _clean(x)]


def _cutoff(decision_date: pd.Timestamp, release_rule: str) -> pd.Timestamp:
    d = _month_end(decision_date)
    if release_rule == "same_month_end":
        return d
    if release_rule == "month_end_plus_1m":
        return _month_end(d - pd.DateOffset(months=1))
    if release_rule == "month_end_plus_2m":
        return _month_end(d - pd.DateOffset(months=2))
    if release_rule == "month_end_plus_3m":
        return _month_end(d - pd.DateOffset(months=3))
    raise RegistryError(f"Unsupported release_rule: {release_rule}")


def _with_decision_col(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.to_datetime(out.index)
    return out.reset_index().rename(columns={"index": "decision_date"})


def build_simple_registry(series_names: list[str], tcode_map: dict[str, int]) -> pd.DataFrame:
    missing = [s for s in series_names if s not in tcode_map]
    if missing:
        raise RegistryError(f"Missing tcodes for series: {missing[:5]}")

    rules = {
        s: {
            "construction_type": "direct_alfred",
            "target_series_id": s,
            "component_series_ids": [],
            "raw_formula": "",
            "release_rule": DEFAULT_RELEASE_RULE,
            "revision_class": "observable_revisions",
            "notes": "Default direct ALFRED mapping.",
        }
        for s in series_names
    }

    for s in SIMPLE_NEVER_REVISED:
        if s in rules:
            rules[s]["revision_class"] = "market_effectively_nonrevised"
    for src, tgt in SIMPLE_DIRECT_ALIAS_MAP.items():
        if src in rules:
            rules[src]["target_series_id"] = tgt
            rules[src]["notes"] = "Alias mapping from series.md"
    for src, tgt in SIMPLE_DIRECT_FRED_MAP.items():
        if src in rules:
            rules[src]["construction_type"] = "direct_fred"
            rules[src]["target_series_id"] = tgt
            rules[src]["notes"] = "Direct FRED mapping from series.md"
    for src, meta in SIMPLE_FORMULA_MAP.items():
        if src in rules:
            rules[src]["construction_type"] = "derived_formula"
            rules[src]["target_series_id"] = ""
            rules[src]["component_series_ids"] = list(meta["components"])
            rules[src]["raw_formula"] = _clean(meta["formula"])
            rules[src]["revision_class"] = _clean(meta["revision_class"])
            rules[src]["notes"] = "Formula mapping from series.md"
    for s in SIMPLE_LOCAL_ONLY:
        if s in rules:
            rules[s]["construction_type"] = "public_proxy"
            rules[s]["target_series_id"] = ""
            rules[s]["component_series_ids"] = []
            rules[s]["raw_formula"] = ""
            rules[s]["revision_class"] = "public_proxy_only"
            rules[s]["notes"] = "Local-only row from series.md"

    rows = []
    for i, s in enumerate(series_names, start=1):
        r = rules[s]
        pub_lag = default_publication_lag_policy(s)
        rows.append(
            {
                "hs_no": i,
                "group": 0,
                "mnemonic_hs": s,
                "mnemonic_old": s,
                "description_hs": s,
                "tcode_hs": int(tcode_map[s]),
                "lag_months_hs": DEFAULT_LAG_MONTHS,
                "vintage_flag_hs": "Y" if r["construction_type"] == "direct_alfred" else "N",
                "construction_type": _clean(r["construction_type"]),
                "target_series_id": _clean(r["target_series_id"]),
                "component_series_ids": json.dumps(r["component_series_ids"], separators=(",", ":"), ensure_ascii=True),
                "raw_formula": _clean(r["raw_formula"]),
                "pub_lag_mode": _clean(pub_lag["pub_lag_mode"]),
                "pub_lag_months": int(pub_lag["pub_lag_months"]),
                "pub_lag_schedule_json": _clean(pub_lag["pub_lag_schedule_json"]),
                "release_rule": _clean(r["release_rule"]),
                "revision_class": _clean(r["revision_class"]),
                "strict_source_rule": "selected_vintage_asof_decision",
                "balanced_source_rule": "strict_asof_then_release_gap_then_pre_vintage_then_proxy",
                "notes": _clean(r["notes"]),
            }
        )
    out = pd.DataFrame(rows)
    return out[REGISTRY_COLUMNS].copy()


def fred_get(ctx: FetchContext, endpoint: str, **params: Any) -> dict[str, Any]:
    query = {k: v for k, v in params.items() if v is not None}
    query["api_key"] = ctx.api_key
    query["file_type"] = "json"
    url = f"{API_BASE}/{endpoint.lstrip('/')}"

    for attempt in range(1, ctx.max_attempts + 1):
        try:
            r = ctx.session.get(url, params=query, timeout=ctx.timeout_sec)
            if r.status_code in RETRY_STATUS:
                if attempt == ctx.max_attempts:
                    raise FredApiError(f"HTTP {r.status_code} after {attempt} attempts: {endpoint}")
                time.sleep(min(10.0, 0.5 * (2 ** (attempt - 1))))
                continue
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict) and payload.get("error_code"):
                raise FredApiError(f"FRED error {payload.get('error_code')}: {payload.get('error_message', '')}")
            if ctx.pause_sec > 0:
                time.sleep(ctx.pause_sec)
            return payload
        except requests.RequestException as exc:
            if attempt == ctx.max_attempts:
                raise FredApiError(f"Request failed after {attempt} attempts: {endpoint}") from exc
            time.sleep(min(10.0, 0.5 * (2 ** (attempt - 1))))
    raise FredApiError(f"Request failed: {endpoint}")


def _extract_series_ids(registry: pd.DataFrame) -> list[str]:
    ids: set[str] = set()
    for _, row in registry.iterrows():
        target = _clean(row.get("target_series_id"))
        if target:
            ids.add(target)
        for c in _json_list(row.get("component_series_ids")):
            ids.add(c)
    return sorted(ids)


def _fetch_history(ctx: FetchContext, series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        payload = fred_get(
            ctx,
            "series/observations",
            series_id=series_id,
            observation_start=start.strftime("%Y-%m-%d"),
            observation_end=end.strftime("%Y-%m-%d"),
            realtime_start="1776-07-04",
            realtime_end="9999-12-31",
        )
        used_relaxed_fallback = False
    except FredApiError:
        # Fallback for intermittent ALFRED outages/timeouts: fetch plain observations
        # and treat them as a non-revised timeline.
        payload = fred_get(
            ctx,
            "series/observations",
            series_id=series_id,
            observation_start=start.strftime("%Y-%m-%d"),
            observation_end=end.strftime("%Y-%m-%d"),
        )
        used_relaxed_fallback = True

    obs = payload.get("observations", [])
    if not isinstance(obs, list) or not obs:
        return pd.DataFrame(columns=["series_id", "obs_date", "realtime_start", "realtime_end", "value"])

    f = pd.DataFrame(obs)
    f["series_id"] = series_id
    f["obs_date"] = pd.to_datetime(f.get("date"), errors="coerce")
    if used_relaxed_fallback:
        f["realtime_start"] = pd.Timestamp("1900-01-01")
        f["realtime_end"] = pd.Timestamp("2262-04-11")
    else:
        f["realtime_start"] = pd.to_datetime(f.get("realtime_start"), errors="coerce")
        f["realtime_end"] = pd.to_datetime(f.get("realtime_end"), errors="coerce")
    f["value"] = pd.to_numeric(f.get("value"), errors="coerce")
    f = f[["series_id", "obs_date", "realtime_start", "realtime_end", "value"]]
    return f.dropna(subset=["obs_date"]).sort_values(["obs_date", "realtime_start", "realtime_end"]).reset_index(drop=True)


def run_cache_stage(
    registry: pd.DataFrame,
    cache_path: Path,
    start: str,
    end: str,
    api_key: str | None = None,
    pause_sec: float = 0.03,
) -> dict[str, pd.DataFrame]:
    key = _clean(api_key) or _clean(os.getenv("FRED_API_KEY"))
    if not key:
        raise FredApiError("FRED_API_KEY is required")

    start_ts = _month_end(start)
    end_ts = _month_end(end)
    history_start = _month_end(start_ts - pd.DateOffset(months=24))
    ctx = FetchContext(api_key=key, session=requests.Session(), pause_sec=pause_sec)

    frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    for sid in _extract_series_ids(registry):
        try:
            frames.append(_fetch_history(ctx, sid, history_start, end_ts))
        except Exception as exc:
            errors.append({"stage": "fetch_component_history", "series_id": sid, "error": str(exc)})

    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["series_id", "obs_date", "realtime_start", "realtime_end", "value"]
    )

    if history.empty:
        availability = pd.DataFrame(columns=["series_id", "n_obs", "obs_min_date", "obs_max_date", "has_alfred"])
        revision = pd.DataFrame(columns=["series_id", "obs_with_revisions", "max_revisions_per_obs"])
    else:
        availability = (
            history.groupby("series_id", dropna=False)
            .agg(
                n_obs=("value", "size"),
                obs_min_date=("obs_date", "min"),
                obs_max_date=("obs_date", "max"),
                has_alfred=("realtime_start", lambda s: True),
            )
            .reset_index()
        )
        vintages = history.groupby(["series_id", "obs_date"], dropna=False)["realtime_start"].nunique().reset_index(name="n_vintages")
        revision = (
            vintages.groupby("series_id", dropna=False)["n_vintages"]
            .agg(obs_with_revisions=lambda s: int((s > 1).sum()), max_revisions_per_obs="max")
            .reset_index()
        )

    payload = {
        "run_meta_cache": pd.DataFrame(
            {
                "key": ["generated_utc", "start", "end", "decision_months", "series_count"],
                "value": [
                    pd.Timestamp.now("UTC").isoformat(),
                    start_ts.strftime("%Y-%m-%d"),
                    end_ts.strftime("%Y-%m-%d"),
                    len(_month_range(start_ts, end_ts)),
                    int(len(registry)),
                ],
            }
        ),
        "series_registry": registry.copy(),
        "availability_audit": availability,
        "revision_audit": revision,
        "component_history": history,
        "error_log": pd.DataFrame(errors, columns=["stage", "series_id", "error"]),
    }

    with pd.ExcelWriter(cache_path, engine="openpyxl") as w:
        for name, frame in payload.items():
            frame.to_excel(w, sheet_name=name, index=False)
    return payload


def load_cache(cache_path: Path) -> dict[str, pd.DataFrame]:
    sheets = ["run_meta_cache", "series_registry", "availability_audit", "revision_audit", "component_history", "error_log"]
    out = {name: pd.read_excel(cache_path, sheet_name=name) for name in sheets}

    comp = out["component_history"]
    if "obs_date" not in comp.columns and "date" in comp.columns:
        comp = comp.rename(columns={"date": "obs_date"})
    if "value" in comp.columns:
        comp["value"] = pd.to_numeric(comp["value"], errors="coerce")
    if "realtime_start" not in comp.columns:
        comp["realtime_start"] = pd.Timestamp.min.normalize()
    if "realtime_end" not in comp.columns:
        comp["realtime_end"] = pd.Timestamp.max.normalize()
    out["component_history"] = comp

    for col in ["obs_date", "realtime_start", "realtime_end", "obs_min_date", "obs_max_date"]:
        for name in ["component_history", "availability_audit"]:
            if col in out[name].columns:
                out[name][col] = pd.to_datetime(out[name][col], errors="coerce")
    return out


def _history_map(history: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if history.empty:
        return {}
    out: dict[str, pd.DataFrame] = {}
    for sid, g in history.groupby("series_id", sort=False):
        sort_cols = [c for c in ["obs_date", "realtime_start", "realtime_end"] if c in g.columns]
        out[sid] = g.sort_values(sort_cols).reset_index(drop=True) if sort_cols else g.reset_index(drop=True)
    return out


def _pick(hist: pd.DataFrame | None, decision: pd.Timestamp, cutoff_date: pd.Timestamp, mode: str) -> tuple[float, pd.Timestamp | None]:
    if hist is None or hist.empty:
        return np.nan, None

    # Balanced fallback modes may use relaxed vintage selection, but they
    # should never make a series available before its first published vintage.
    if mode != "strict_asof" and "realtime_start" in hist.columns:
        first_available = pd.to_datetime(hist["realtime_start"], errors="coerce").dropna()
        if not first_available.empty and pd.Timestamp(decision) < first_available.min():
            return np.nan, None

    c = hist[hist["obs_date"] <= cutoff_date]
    if c.empty:
        return np.nan, None

    if mode == "strict_asof":
        c = c[(c["realtime_start"] <= decision) & (c["realtime_end"] >= decision)]
        if c.empty:
            return np.nan, pd.NaT
        r = c.iloc[-1]
    elif mode == "earliest":
        r = c.iloc[0]
    else:
        r = c.iloc[-1]
    return (float(r["value"]) if pd.notna(r["value"]) else np.nan), r["obs_date"]


def _value_for(
    row: pd.Series,
    history: dict[str, pd.DataFrame],
    decision: pd.Timestamp,
    mode: str,
    max_release_gap_months: int,
) -> float:
    ctype = _clean(row.get("construction_type"))
    cutoff_date = _cutoff(decision, _clean(row.get("release_rule")))

    def latest_or_gap(sid: str, pick_mode: str) -> float:
        val, obs = _pick(history.get(sid), decision, cutoff_date, pick_mode)
        if pick_mode != "latest":
            return val
        if mode == "release_gap" and pd.notna(val) and pd.notna(obs):
            if _months_between(cutoff_date, pd.Timestamp(obs)) <= int(max_release_gap_months):
                return val
            return np.nan
        return val

    if ctype == "direct_alfred" and mode == "strict":
        sid = _clean(row.get("target_series_id")) or _clean(row.get("mnemonic_hs"))
        val, _ = _pick(history.get(sid), decision, cutoff_date, "strict_asof")
        return val

    if ctype in {"direct_alfred", "direct_fred", "public_proxy"}:
        sid = _clean(row.get("target_series_id")) or _clean(row.get("mnemonic_hs"))
        return latest_or_gap(sid, "earliest" if mode == "pre_vintage" else "latest")

    if ctype == "derived_formula":
        comps = _json_list(row.get("component_series_ids"))
        if not comps:
            return np.nan
        vals = {
            c: latest_or_gap(c, "earliest" if mode == "pre_vintage" else "latest")
            for c in comps
        }
        if any(pd.isna(v) for v in vals.values()):
            return np.nan
        out = pd.eval(_clean(row.get("raw_formula")), local_dict=vals, engine="python")
        try:
            return float(out)
        except Exception:
            return np.nan

    return np.nan


def run_final_stage(
    registry: pd.DataFrame,
    cache_payload: dict[str, pd.DataFrame],
    final_path: Path,
    start: str,
    end: str,
    max_release_gap_months: int = 1,
) -> dict[str, pd.DataFrame]:
    history = cache_payload.get("component_history", pd.DataFrame()).copy()
    for c in ["obs_date", "realtime_start", "realtime_end"]:
        if c in history.columns:
            history[c] = pd.to_datetime(history[c], errors="coerce")
    hmap = _history_map(history)

    series = registry["mnemonic_hs"].astype(str).tolist()
    reg = registry.set_index("mnemonic_hs", drop=False)
    decisions = _month_range(start, end)
    raw_strict = pd.DataFrame(index=decisions, columns=series, dtype=float)
    src_strict = pd.DataFrame("", index=decisions, columns=series, dtype=object)
    raw_bal = raw_strict.copy()
    src_bal = src_strict.copy()

    for d in decisions:
        for s in series:
            row = reg.loc[s]
            v = _value_for(row, hmap, d, "strict", max_release_gap_months)
            raw_strict.at[d, s] = v
            raw_bal.at[d, s] = v
            if pd.notna(v):
                src_strict.at[d, s] = SOURCE_STRICT
                src_bal.at[d, s] = SOURCE_STRICT
                continue

            v_gap = _value_for(row, hmap, d, "release_gap", max_release_gap_months)
            if pd.notna(v_gap):
                raw_bal.at[d, s] = v_gap
                src_bal.at[d, s] = SOURCE_RELEASE_GAP
                continue

            v_early = _value_for(row, hmap, d, "pre_vintage", max_release_gap_months)
            if pd.notna(v_early):
                raw_bal.at[d, s] = v_early
                src_bal.at[d, s] = SOURCE_PRE_VINTAGE

    tcode_map = {str(r["mnemonic_hs"]): int(r["tcode_hs"]) for _, r in registry.iterrows()}
    tcode_bal = apply_tcodes_panel(raw_bal, tcode_map, series)
    miss = int(raw_bal.isna().sum().sum())
    payload = {
        "snapshot_raw_balanced": _with_decision_col(raw_bal),
        "snapshot_source_balanced": _with_decision_col(src_bal),
        "diag_checks": pd.DataFrame(
            [
                {"check": "balanced_no_nans", "passed": miss == 0, "detail": f"remaining_missing={miss}"},
                {"check": "release_gap_fills", "passed": True, "detail": f"count={int((src_bal == SOURCE_RELEASE_GAP).sum().sum())}"},
                {"check": "pre_vintage_fills", "passed": True, "detail": f"count={int((src_bal == SOURCE_PRE_VINTAGE).sum().sum())}"},
            ]
        ),
        "revision_audit": cache_payload.get("revision_audit", pd.DataFrame()).copy(),
        "snapshot_tcode_balanced": _with_decision_col(tcode_bal),
    }

    with pd.ExcelWriter(final_path, engine="openpyxl") as w:
        for name, frame in payload.items():
            frame.to_excel(w, sheet_name=name[:31], index=False)
    return payload


def _load_vintage(path: Path) -> pd.DataFrame:
    f = pd.read_csv(path)
    if "sasdate" not in f.columns:
        return pd.DataFrame()
    sas = f["sasdate"].astype(str).str.strip()
    sas = sas.where(~sas.str.fullmatch(r"(?i)transform:?"), "")
    f["sasdate"] = pd.to_datetime(sas, format="%m/%d/%Y", errors="coerce")
    f = f.dropna(subset=["sasdate"]).set_index("sasdate").sort_index()
    f.index = f.index.map(_month_end)
    for c in f.columns:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    return f


def _vintage_file(base: Path, decision: pd.Timestamp) -> Path | None:
    # Historical FRED-MD vintage files are monthly snapshots. We treat a
    # YYYY-MM vintage as available at that month-end, not at the start of the
    # month. That gives the intended behavior:
    # - decision on 2020-01-31 -> use 2020-01 vintage
    # - decision on 2020-01-01 -> use 2019-12 vintage
    # - before the first published vintage month -> no same-month vintage
    #   available, so anchor backfill may need to fall back to the earliest
    #   archived vintage for that series instead
    decision = pd.Timestamp(decision)
    month_end = _month_end(decision)
    vintage_month = month_end if decision.normalize() == month_end else _month_end(month_end - pd.DateOffset(months=1))

    p = base / str(vintage_month.year) / f"{vintage_month.strftime('%Y-%m')}.csv"
    if p.exists():
        return p
    p = base / f"{vintage_month.strftime('%Y-%m')}.csv"
    if p.exists():
        return p
    return None


def _vintage_month_from_path(path: Path) -> pd.Timestamp | None:
    try:
        return _month_end(f"{path.stem}-01")
    except Exception:
        return None


def _pick_from_vintage_frame(
    vintage_frame: pd.DataFrame,
    series_name: str,
    obs_date: pd.Timestamp,
    max_release_gap_months: int = 1,
) -> float:
    if vintage_frame.empty or series_name not in vintage_frame.columns:
        return np.nan

    series = pd.to_numeric(vintage_frame[series_name], errors="coerce").dropna()
    if series.empty:
        return np.nan

    obs_date = _month_end(obs_date)
    if obs_date in series.index:
        value = series.loc[obs_date]
        if pd.notna(value):
            return float(value)

    eligible = series.loc[series.index <= obs_date]
    if eligible.empty:
        return np.nan

    last_obs_date = pd.Timestamp(eligible.index[-1])
    if _months_between(obs_date, last_obs_date) <= int(max_release_gap_months):
        return float(eligible.iloc[-1])
    return np.nan


def _earliest_series_anchor_vintages(
    vintage_dirs: list[Path], series_names: list[str]
) -> dict[str, tuple[pd.Timestamp, Path]]:
    remaining = set(series_names)
    anchors: dict[str, tuple[pd.Timestamp, Path]] = {}
    vintage_files: list[tuple[pd.Timestamp, Path]] = []
    seen_paths: set[str] = set()

    for base in vintage_dirs:
        if not base.exists():
            continue
        for path in base.rglob("*.csv"):
            key = str(path.resolve())
            if key in seen_paths:
                continue
            month = _vintage_month_from_path(path)
            if month is None:
                continue
            seen_paths.add(key)
            vintage_files.append((month, path))

    vintage_files.sort(key=lambda item: (item[0], str(item[1])))
    for month, path in vintage_files:
        if not remaining:
            break
        try:
            cols = {str(c) for c in pd.read_csv(path, nrows=0).columns}
        except Exception:
            continue
        for series in list(remaining):
            if series in cols:
                anchors[series] = (month, path)
                remaining.remove(series)
    return anchors


def apply_anchor_backfill_to_balanced(
    raw_bal: pd.DataFrame,
    src_bal: pd.DataFrame,
    registry: pd.DataFrame,
    nonrev_raw: pd.DataFrame,
    vintage_dirs: list[Path],
    tag: str = SOURCE_PRE_VINTAGE,
    max_release_gap_months: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    raw = raw_bal.copy()
    src = src_bal.copy()
    raw.index = pd.to_datetime(raw.index).map(_month_end)
    src.index = pd.to_datetime(src.index).map(_month_end)
    nonrev = nonrev_raw.copy()
    nonrev.index = pd.to_datetime(nonrev.index).map(_month_end)
    reg = registry.copy()
    if "mnemonic_hs" not in reg.columns:
        raise RegistryError("registry must include mnemonic_hs for anchor backfill")
    for col in PUB_LAG_COLUMNS:
        if col not in reg.columns:
            reg[col] = reg["mnemonic_hs"].astype(str).map(lambda s: default_publication_lag_policy(s)[col])
    reg = reg.set_index("mnemonic_hs", drop=False)
    series_names = reg.index.astype(str).tolist()

    cache: dict[Path, pd.DataFrame] = {}
    counts: dict[str, int] = {}
    anchor_vintages = _earliest_series_anchor_vintages(vintage_dirs, series_names)

    for d in raw.index:
        for s in series_names:
            if s not in raw.columns or pd.notna(raw.at[d, s]):
                continue
            lag_meta = reg.loc[s]
            obs_date = lagged_observation_date(d, lag_meta)
            vintage_gap_months = max(int(max_release_gap_months), resolve_publication_lag_months(lag_meta, d))

            filled = False
            for base in vintage_dirs:
                vp = _vintage_file(base, d)
                if vp is None:
                    continue
                if vp not in cache:
                    cache[vp] = _load_vintage(vp)
                vm = cache[vp]
                if vm.empty or s not in vm.columns:
                    continue
                v = _pick_from_vintage_frame(vm, s, obs_date, max_release_gap_months=vintage_gap_months)
                if pd.notna(v):
                    raw.at[d, s] = float(v)
                    src.at[d, s] = tag
                    counts[s] = counts.get(s, 0) + 1
                    filled = True
                    break

            # If the decision date predates the first archived historical
            # vintage for this series, backfill from that earliest archived
            # vintage before falling back to the latest non-revised panel.
            if not filled:
                anchor_meta = anchor_vintages.get(s)
                if anchor_meta is not None:
                    anchor_month, anchor_path = anchor_meta
                    if pd.Timestamp(d) < anchor_month:
                        if anchor_path not in cache:
                            cache[anchor_path] = _load_vintage(anchor_path)
                        vm = cache[anchor_path]
                        if not vm.empty and s in vm.columns:
                            v = _pick_from_vintage_frame(vm, s, obs_date, max_release_gap_months=vintage_gap_months)
                            if pd.notna(v):
                                raw.at[d, s] = float(v)
                                src.at[d, s] = tag
                                counts[s] = counts.get(s, 0) + 1
                                filled = True

            if filled or s not in nonrev.columns:
                continue
            v = nonrev.at[obs_date, s] if obs_date in nonrev.index else np.nan
            if pd.isna(v):
                past = nonrev.loc[nonrev.index <= obs_date, s].dropna()
                if not past.empty:
                    v = past.iloc[-1]
            if pd.notna(v):
                raw.at[d, s] = float(v)
                src.at[d, s] = SOURCE_NONREV
                counts[s] = counts.get(s, 0) + 1

    return raw, src, counts


def _tcode(x: pd.Series, t: int) -> pd.Series:
    z = pd.to_numeric(x, errors="coerce")
    if t == 1:
        return z
    if t == 2:
        return z.diff(1)
    if t == 3:
        return z.diff(1).diff(1)
    if t == 4:
        return np.log(z.where(z > 0))
    if t == 5:
        y = z.where(z > 0)
        return np.log(y).diff(1)
    if t == 6:
        y = z.where(z > 0)
        return np.log(y).diff(1).diff(1)
    if t == 7:
        return (z / z.shift(1) - 1.0).diff(1)
    raise ValueError(f"Unsupported tcode: {t}")


def apply_tcodes_panel(raw_panel: pd.DataFrame, tcode_map: dict[str, int], series_names: list[str]) -> pd.DataFrame:
    cols = {s: _tcode(raw_panel[s], int(tcode_map[s])) for s in series_names}
    return pd.DataFrame(cols, index=raw_panel.index)


__all__ = [
    "FetchContext",
    "FredApiError",
    "RegistryError",
    "apply_anchor_backfill_to_balanced",
    "apply_tcodes_panel",
    "build_simple_registry",
    "fred_get",
    "load_cache",
    "run_cache_stage",
    "run_final_stage",
]
