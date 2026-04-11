import json
from pathlib import Path
from typing import Any
import os

import pandas as pd

PUB_LAG_MODE_FIXED = "fixed"
PUB_LAG_MODE_SCHEDULE = "decision_month_schedule"
PUB_LAG_DEFAULT_MONTHS = 1
PUB_LAG_COLUMNS = ["pub_lag_mode", "pub_lag_months", "pub_lag_schedule_json"]

PUB_LAG_OVERRIDES = {
    "ACOGNO": {"pub_lag_mode": PUB_LAG_MODE_FIXED, "pub_lag_months": 2},
    "BUSINVx": {"pub_lag_mode": PUB_LAG_MODE_FIXED, "pub_lag_months": 2},
    "ISRATIOx": {"pub_lag_mode": PUB_LAG_MODE_FIXED, "pub_lag_months": 2},
    "CONSPI": {"pub_lag_mode": PUB_LAG_MODE_FIXED, "pub_lag_months": 2},
    "NONREVSL": {"pub_lag_mode": PUB_LAG_MODE_FIXED, "pub_lag_months": 2},
    "DTCOLNVHFNM": {"pub_lag_mode": PUB_LAG_MODE_FIXED, "pub_lag_months": 2},
    "DTCTHFNM": {"pub_lag_mode": PUB_LAG_MODE_FIXED, "pub_lag_months": 2},
    "HWI": {"pub_lag_mode": PUB_LAG_MODE_FIXED, "pub_lag_months": 2},
    "HWIURATIO": {"pub_lag_mode": PUB_LAG_MODE_FIXED, "pub_lag_months": 2},
    "CMRMTSPLx": {"pub_lag_mode": PUB_LAG_MODE_FIXED, "pub_lag_months": 3},
    "S&P div yield": {
        "pub_lag_mode": PUB_LAG_MODE_SCHEDULE,
        "pub_lag_months": PUB_LAG_DEFAULT_MONTHS,
        "pub_lag_schedule_json": json.dumps(
            {1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3, 7: 1, 8: 2, 9: 3, 10: 1, 11: 2, 12: 3},
            separators=(",", ":"),
            ensure_ascii=True,
        ),
    },
    "S&P PE ratio": {
        "pub_lag_mode": PUB_LAG_MODE_SCHEDULE,
        "pub_lag_months": PUB_LAG_DEFAULT_MONTHS,
        "pub_lag_schedule_json": json.dumps(
            {1: 6, 2: 4, 3: 5, 4: 6, 5: 4, 6: 5, 7: 6, 8: 4, 9: 5, 10: 6, 11: 4, 12: 5},
            separators=(",", ":"),
            ensure_ascii=True,
        ),
    },
}


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).strip().split())


def month_end(ts: Any) -> pd.Timestamp:
    return (pd.Timestamp(ts) + pd.offsets.MonthEnd(0)).normalize()


def default_publication_lag_policy(series_name: str = "") -> dict[str, Any]:
    policy = {
        "pub_lag_mode": PUB_LAG_MODE_FIXED,
        "pub_lag_months": PUB_LAG_DEFAULT_MONTHS,
        "pub_lag_schedule_json": "",
    }
    override = PUB_LAG_OVERRIDES.get(series_name, {})
    policy.update(override)
    return policy


def publication_lag_policy_for_series(series_name: str) -> dict[str, Any]:
    return default_publication_lag_policy(series_name)


def parse_publication_lag_schedule(value: Any) -> dict[int, int]:
    text = _clean(value)
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("pub_lag_schedule_json must decode to an object")
    out: dict[int, int] = {}
    for month, lag in parsed.items():
        out[int(month)] = int(lag)
    return out


def build_publication_lag_lookup(registry: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if registry is None or registry.empty or "mnemonic_hs" not in registry.columns:
        return {}

    lookup: dict[str, dict[str, Any]] = {}
    for _, row in registry.iterrows():
        series_name = _clean(row.get("mnemonic_hs"))
        policy = default_publication_lag_policy(series_name)
        if "pub_lag_mode" in row:
            mode = _clean(row.get("pub_lag_mode"))
            if mode:
                policy["pub_lag_mode"] = mode
        if "pub_lag_months" in row and pd.notna(row.get("pub_lag_months")):
            policy["pub_lag_months"] = int(row.get("pub_lag_months"))
        if "pub_lag_schedule_json" in row:
            schedule_json = _clean(row.get("pub_lag_schedule_json"))
            if schedule_json:
                policy["pub_lag_schedule_json"] = schedule_json
        lookup[series_name] = policy
    return lookup


def resolve_publication_lag_months(metadata: pd.Series | dict[str, Any] | None, decision_date: Any) -> int:
    series_name = ""
    if metadata is None:
        policy = default_publication_lag_policy()
    else:
        if isinstance(metadata, pd.Series):
            series_name = _clean(metadata.get("mnemonic_hs"))
            payload = metadata.to_dict()
        else:
            series_name = _clean(metadata.get("mnemonic_hs"))
            payload = dict(metadata)
        policy = default_publication_lag_policy(series_name)
        mode = _clean(payload.get("pub_lag_mode"))
        if mode:
            policy["pub_lag_mode"] = mode
        lag_months = payload.get("pub_lag_months")
        if lag_months is not None and not pd.isna(lag_months):
            policy["pub_lag_months"] = int(lag_months)
        schedule_json = _clean(payload.get("pub_lag_schedule_json"))
        if schedule_json:
            policy["pub_lag_schedule_json"] = schedule_json

    if policy["pub_lag_mode"] == PUB_LAG_MODE_SCHEDULE:
        schedule = parse_publication_lag_schedule(policy.get("pub_lag_schedule_json"))
        lag = schedule.get(month_end(decision_date).month)
        if lag is not None:
            return int(lag)
    return int(policy.get("pub_lag_months", PUB_LAG_DEFAULT_MONTHS))


def lagged_observation_date(decision_date: Any, metadata: pd.Series | dict[str, Any] | None) -> pd.Timestamp:
    decision = month_end(decision_date)
    lag_months = resolve_publication_lag_months(metadata, decision)
    return month_end(decision - pd.DateOffset(months=lag_months))


def apply_publication_lag_to_panel(
    data: pd.DataFrame,
    registry: pd.DataFrame | None = None,
    series_names: list[str] | None = None,
) -> pd.DataFrame:
    if data.empty:
        return data.copy()

    source = data.copy()
    source.index = pd.DatetimeIndex(pd.to_datetime(source.index)).map(month_end)
    source = source.sort_index()

    out = source.copy()
    lookup = build_publication_lag_lookup(registry)
    columns = series_names or list(out.columns)
    index = pd.DatetimeIndex(out.index)

    for series_name in columns:
        if series_name not in out.columns:
            continue
        metadata = lookup.get(series_name, default_publication_lag_policy(series_name))
        obs_dates = pd.DatetimeIndex([lagged_observation_date(d, metadata) for d in index])
        out[series_name] = source[series_name].reindex(obs_dates).to_numpy()
    return out


def load_publication_lag_registry(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_PUB_LAG_REGISTRY = os.path.join(_REPO_ROOT, 'data', 'ALFRED', 'simple_outputs', 'mapping_registry.csv')

def apply_fred_md_publication_lag(fred_md: pd.DataFrame, registry_path: str | None = None) -> pd.DataFrame:
    """
    Apply the shared per-series publication lag policy to a transformed
    latest-snapshot FRED-MD panel.
    """
    registry_path = registry_path or _DEFAULT_PUB_LAG_REGISTRY
    if not os.path.isabs(registry_path):
        registry_path = os.path.join(_REPO_ROOT, registry_path)
    registry = load_publication_lag_registry(registry_path)
    return apply_publication_lag_to_panel(fred_md, registry=registry)
