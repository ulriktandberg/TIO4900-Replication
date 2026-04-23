"""Forecast-vintage macro data loader.

This module builds macro panels for inspection under an explicit no-lookahead
policy:

* A forecast dated month-end t uses the next-month vintage/information set.
* ALFRED component history is the primary source where available.
* Historical FRED-MD vintage files are archive fallback only.
* Latest revised FRED-MD files are never used as fallback.
* Missing values remain missing and are tagged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from utils.base_utils import prepare_macro_panel_for_project


SOURCE_ALFRED_ASOF = "ALFRED_ASOF"
SOURCE_ARCHIVE_FALLBACK = "ARCHIVE_FALLBACK"
SOURCE_EARLIEST_ARCHIVE_ANCHOR = "EARLIEST_ARCHIVE_ANCHOR"
SOURCE_MISSING = "MISSING"

OPEN_END_SENTINEL = pd.Timestamp.max.normalize()

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CACHE_PATH = _REPO_ROOT / "data" / "ALFRED" / "fred_md_simple_rt_cache.xlsx"
_DEFAULT_REGISTRY_PATH = _REPO_ROOT / "data" / "ALFRED" / "simple_outputs" / "mapping_registry.csv"
_DEFAULT_ARCHIVE_DIR = (
    _REPO_ROOT
    / "data"
    / "ALFRED"
    / "Historical-vintages-of-FRED-MD-1999-08-to-2024-12"
)


ARCHIVE_COLUMN_ALIASES = {
    "AMBSL": "BOGMBASE",
    "TWEXMMTH": "TWEXAFEGSMTHx",
    "VXOCLSx": "VIXCLSx",
    "PPIFGS": "WPSFD49207",
    "PPIFCG": "WPSFD49502",
    "PPIITM": "WPSID61",
    "PPICRM": "WPSID62",
}


@dataclass(frozen=True)
class VintageSelection:
    forecast_date: pd.Timestamp
    requested_vintage_month: pd.Timestamp
    selected_archive_month: pd.Timestamp
    archive_path: Path
    archive_source_tag: str
    asof_date: pd.Timestamp


@dataclass
class ForecastVintagePanel:
    raw: pd.DataFrame
    transformed: pd.DataFrame
    source_tags: pd.DataFrame
    metadata: dict[str, Any]


def month_end(value: Any) -> pd.Timestamp:
    return (pd.Timestamp(value) + pd.offsets.MonthEnd(0)).normalize()


def month_range(start: Any, end: Any) -> pd.DatetimeIndex:
    return pd.date_range(month_end(start), month_end(end), freq=pd.offsets.MonthEnd(1))


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).strip().split())


def _json_list(value: Any) -> list[str]:
    text = _clean(value)
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list")
    return [_clean(x) for x in parsed if _clean(x)]


def _parse_realtime_end(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    parsed = pd.to_datetime(values, errors="coerce")
    parsed = parsed.mask(text.eq("9999-12-31"), OPEN_END_SENTINEL)
    return parsed.fillna(OPEN_END_SENTINEL)


def _tcode(series: pd.Series, code: int) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if code == 1:
        return x
    if code == 2:
        return x.diff(1)
    if code == 3:
        return x.diff(1).diff(1)
    if code == 4:
        return np.log(x.where(x > 0))
    if code == 5:
        return np.log(x.where(x > 0)).diff(1)
    if code == 6:
        return np.log(x.where(x > 0)).diff(1).diff(1)
    if code == 7:
        return (x / x.shift(1) - 1.0).diff(1)
    raise ValueError(f"Unsupported FRED-MD transformation code: {code}")


def _combine_duplicate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.is_unique:
        return frame

    out = pd.DataFrame(index=frame.index)
    for col in dict.fromkeys(frame.columns):
        same_name = frame.loc[:, frame.columns == col]
        out[col] = same_name.bfill(axis=1).iloc[:, 0]
    return out


class ForecastVintageMacroStore:
    """Cache-backed forecast-vintage macro data store."""

    def __init__(
        self,
        *,
        cache_path: str | Path = _DEFAULT_CACHE_PATH,
        registry_path: str | Path = _DEFAULT_REGISTRY_PATH,
        archive_dir: str | Path = _DEFAULT_ARCHIVE_DIR,
        forecast_vintage_offset_months: int = 1,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.registry_path = Path(registry_path)
        self.archive_dir = Path(archive_dir)
        self.forecast_vintage_offset_months = int(forecast_vintage_offset_months)

        if not self.cache_path.exists():
            raise FileNotFoundError(f"ALFRED cache workbook not found: {self.cache_path}")
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Mapping registry not found: {self.registry_path}")
        if not self.archive_dir.exists():
            raise FileNotFoundError(f"Historical FRED-MD archive not found: {self.archive_dir}")

        self.registry = pd.read_csv(self.registry_path)
        self.series_names = self.registry["mnemonic_hs"].astype(str).tolist()
        self.tcode_map = {
            str(row["mnemonic_hs"]): int(row["tcode_hs"])
            for _, row in self.registry.iterrows()
        }
        self._archive_index = self._build_archive_index()
        self._component_history: pd.DataFrame | None = None
        self._history_by_series: dict[str, pd.DataFrame] | None = None
        self._archive_raw_cache: dict[pd.Timestamp, pd.DataFrame] = {}

    @property
    def first_archive_month(self) -> pd.Timestamp:
        return min(self._archive_index)

    @property
    def last_archive_month(self) -> pd.Timestamp:
        return max(self._archive_index)

    def _build_archive_index(self) -> dict[pd.Timestamp, Path]:
        out: dict[pd.Timestamp, Path] = {}
        for path in self.archive_dir.rglob("*.csv"):
            try:
                vintage_month = month_end(f"{path.stem}-01")
            except Exception:
                continue
            out[vintage_month] = path
        if not out:
            raise FileNotFoundError(f"No archive CSV files found under {self.archive_dir}")
        return dict(sorted(out.items(), key=lambda item: item[0]))

    def _load_component_history(self) -> pd.DataFrame:
        if self._component_history is not None:
            return self._component_history

        hist = pd.read_excel(
            self.cache_path,
            sheet_name="component_history",
            usecols=["series_id", "obs_date", "realtime_start", "realtime_end", "value"],
        )
        hist["series_id"] = hist["series_id"].astype(str)
        hist["obs_date"] = pd.to_datetime(hist["obs_date"], errors="coerce").map(month_end)
        hist["realtime_start"] = pd.to_datetime(hist["realtime_start"], errors="coerce")
        hist["realtime_end"] = _parse_realtime_end(hist["realtime_end"])
        hist["value"] = pd.to_numeric(hist["value"], errors="coerce")
        hist = hist.dropna(subset=["series_id", "obs_date", "realtime_start"])
        hist = hist.sort_values(["series_id", "obs_date", "realtime_start", "realtime_end"])
        self._component_history = hist.reset_index(drop=True)
        return self._component_history

    def _history_map(self) -> dict[str, pd.DataFrame]:
        if self._history_by_series is not None:
            return self._history_by_series

        hist = self._load_component_history()
        self._history_by_series = {
            sid: group.reset_index(drop=True)
            for sid, group in hist.groupby("series_id", sort=False)
        }
        return self._history_by_series

    def cache_coverage_summary(self) -> pd.DataFrame:
        hist = self._load_component_history()
        return (
            hist.groupby("series_id", dropna=False)
            .agg(
                n_rows=("value", "size"),
                obs_min=("obs_date", "min"),
                obs_max=("obs_date", "max"),
                realtime_min=("realtime_start", "min"),
                realtime_max=("realtime_start", "max"),
            )
            .reset_index()
        )

    def archive_coverage_summary(self) -> pd.DataFrame:
        rows = []
        for month, path in self._archive_index.items():
            try:
                cols = pd.read_csv(path, nrows=0).columns.tolist()
            except Exception:
                cols = []
            rows.append(
                {
                    "vintage_month": month,
                    "path": str(path),
                    "series_columns": max(len(cols) - 1, 0),
                }
            )
        return pd.DataFrame(rows)

    def error_log(self) -> pd.DataFrame:
        return pd.read_excel(self.cache_path, sheet_name="error_log")

    def select_vintage_for_forecast_date(self, forecast_date: Any) -> VintageSelection:
        forecast = month_end(forecast_date)
        requested = month_end(forecast + pd.DateOffset(months=self.forecast_vintage_offset_months))
        asof = requested

        if requested in self._archive_index:
            selected = requested
            tag = SOURCE_ARCHIVE_FALLBACK
        elif requested < self.first_archive_month:
            selected = self.first_archive_month
            tag = SOURCE_EARLIEST_ARCHIVE_ANCHOR
        else:
            available = [m for m in self._archive_index if m <= requested]
            if not available:
                selected = self.first_archive_month
                tag = SOURCE_EARLIEST_ARCHIVE_ANCHOR
            else:
                selected = max(available)
                tag = SOURCE_ARCHIVE_FALLBACK

        return VintageSelection(
            forecast_date=forecast,
            requested_vintage_month=requested,
            selected_archive_month=selected,
            archive_path=self._archive_index[selected],
            archive_source_tag=tag,
            asof_date=asof,
        )

    def date_mapping_examples(self, dates: Iterable[Any]) -> pd.DataFrame:
        rows = []
        for date in dates:
            sel = self.select_vintage_for_forecast_date(date)
            rows.append(
                {
                    "forecast_date": sel.forecast_date,
                    "requested_vintage_month": sel.requested_vintage_month,
                    "selected_archive_month": sel.selected_archive_month,
                    "archive_source_tag": sel.archive_source_tag,
                    "archive_path": str(sel.archive_path),
                    "asof_date": sel.asof_date,
                }
            )
        return pd.DataFrame(rows)

    def _alfred_series_asof(
        self,
        series_id: str,
        *,
        asof_date: pd.Timestamp,
        index: pd.DatetimeIndex,
    ) -> tuple[pd.Series, pd.Series]:
        values = pd.Series(np.nan, index=index, dtype=float)
        tags = pd.Series(SOURCE_MISSING, index=index, dtype=object)
        hist = self._history_map().get(series_id)
        if hist is None or hist.empty:
            return values, tags

        active = hist[
            (hist["obs_date"] >= index[0])
            & (hist["obs_date"] <= index[-1])
            & (hist["realtime_start"] <= asof_date)
            & (hist["realtime_end"] >= asof_date)
        ]
        if active.empty:
            return values, tags

        latest = active.sort_values(["obs_date", "realtime_start", "realtime_end"]).groupby("obs_date").tail(1)
        picked = pd.Series(latest["value"].to_numpy(), index=pd.DatetimeIndex(latest["obs_date"]))
        values = picked.reindex(index).astype(float)
        tags.loc[values.notna()] = SOURCE_ALFRED_ASOF
        return values, tags

    def _archive_raw(self, selected_archive_month: pd.Timestamp) -> pd.DataFrame:
        if selected_archive_month in self._archive_raw_cache:
            return self._archive_raw_cache[selected_archive_month]

        path = self._archive_index[selected_archive_month]
        raw = pd.read_csv(path, skiprows=[1])
        date_col = "sasdate" if "sasdate" in raw.columns else raw.columns[0]
        raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
        raw = raw.dropna(subset=[date_col]).set_index(date_col).sort_index()
        raw.index = pd.DatetimeIndex(raw.index).map(month_end)
        raw.index.name = "date"
        raw = raw.rename(columns=ARCHIVE_COLUMN_ALIASES)
        raw = _combine_duplicate_columns(raw)
        for col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        self._archive_raw_cache[selected_archive_month] = raw
        return raw

    def _apply_archive_fallback(
        self,
        raw: pd.DataFrame,
        tags: pd.DataFrame,
        selection: VintageSelection,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        archive = self._archive_raw(selection.selected_archive_month)
        archive = archive.reindex(raw.index)
        fallback_cols = [c for c in self.series_names if c in archive.columns and c in raw.columns]

        for col in fallback_cols:
            missing = raw[col].isna()
            fill_values = archive[col].where(missing)
            can_fill = fill_values.notna()
            if can_fill.any():
                raw.loc[can_fill, col] = fill_values.loc[can_fill]
                tags.loc[can_fill, col] = selection.archive_source_tag

        tags = tags.where(raw.notna(), SOURCE_MISSING)
        return raw, tags

    def _formula_source_tags(self, source_frames: list[pd.Series], index: pd.DatetimeIndex) -> pd.Series:
        if not source_frames:
            return pd.Series(SOURCE_MISSING, index=index, dtype=object)
        frame = pd.concat(source_frames, axis=1)
        out = pd.Series(SOURCE_ALFRED_ASOF, index=index, dtype=object)
        out[frame.eq(SOURCE_MISSING).any(axis=1)] = SOURCE_MISSING
        out[frame.eq(SOURCE_ARCHIVE_FALLBACK).any(axis=1)] = SOURCE_ARCHIVE_FALLBACK
        out[frame.eq(SOURCE_EARLIEST_ARCHIVE_ANCHOR).any(axis=1)] = SOURCE_EARLIEST_ARCHIVE_ANCHOR
        return out

    def _raw_panel_from_alfred(
        self,
        *,
        selection: VintageSelection,
        index: pd.DatetimeIndex,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        raw = pd.DataFrame(np.nan, index=index, columns=self.series_names, dtype=float)
        tags = pd.DataFrame(SOURCE_MISSING, index=index, columns=self.series_names, dtype=object)
        component_cache: dict[str, tuple[pd.Series, pd.Series]] = {}

        def component(series_id: str) -> tuple[pd.Series, pd.Series]:
            if series_id not in component_cache:
                component_cache[series_id] = self._alfred_series_asof(
                    series_id,
                    asof_date=selection.asof_date,
                    index=index,
                )
            return component_cache[series_id]

        for _, row in self.registry.iterrows():
            name = str(row["mnemonic_hs"])
            ctype = _clean(row.get("construction_type"))
            if ctype == "derived_formula":
                component_ids = _json_list(row.get("component_series_ids"))
                values_by_component: dict[str, pd.Series] = {}
                source_parts: list[pd.Series] = []
                for sid in component_ids:
                    values, source = component(sid)
                    values_by_component[sid] = values
                    source_parts.append(source)
                if values_by_component:
                    try:
                        formula_values = pd.eval(
                            _clean(row.get("raw_formula")),
                            local_dict=values_by_component,
                            engine="python",
                        )
                        raw[name] = pd.to_numeric(formula_values, errors="coerce")
                        tags[name] = self._formula_source_tags(source_parts, index)
                        tags.loc[raw[name].isna(), name] = SOURCE_MISSING
                    except Exception:
                        raw[name] = np.nan
                        tags[name] = SOURCE_MISSING
                continue

            target = _clean(row.get("target_series_id")) or name
            if not target:
                continue
            values, source = component(target)
            raw[name] = values
            tags[name] = source

        return raw, tags

    def _transform_panel(self, raw: pd.DataFrame) -> pd.DataFrame:
        columns = {
            name: _tcode(raw[name], self.tcode_map[name])
            for name in raw.columns
        }
        return pd.DataFrame(columns, index=raw.index)

    def panel_for_forecast_date(
        self,
        forecast_date: Any,
        *,
        start: Any = "1959-01-31",
        end: Any | None = None,
        extra_drop_series: Iterable[str] = (),
        include_raw_warmup: bool = True,
    ) -> ForecastVintagePanel:
        """Build the as-of macro panel for one forecast date.

        Parameters
        ----------
        forecast_date:
            Month-end forecast date t.
        start, end:
            Output sample range. If end is omitted, it defaults to forecast_date.
        extra_drop_series:
            Additional macro series to drop after construction.
        include_raw_warmup:
            If True, fetch two months before start to compute differenced
            transformations, then slice output back to start:end.
        """
        selection = self.select_vintage_for_forecast_date(forecast_date)
        start_ts = month_end(start)
        end_ts = month_end(end or forecast_date)
        raw_start = month_end(start_ts - pd.DateOffset(months=2)) if include_raw_warmup else start_ts
        raw_index = month_range(raw_start, end_ts)

        raw, tags = self._raw_panel_from_alfred(selection=selection, index=raw_index)
        raw, tags = self._apply_archive_fallback(raw, tags, selection)
        transformed = self._transform_panel(raw)

        raw = raw.loc[start_ts:end_ts]
        transformed = transformed.loc[start_ts:end_ts]
        tags = tags.loc[start_ts:end_ts]

        raw = prepare_macro_panel_for_project(raw, extra_drop_series=extra_drop_series)
        transformed = prepare_macro_panel_for_project(transformed, extra_drop_series=extra_drop_series)
        tags = tags.loc[:, transformed.columns]

        metadata = {
            "forecast_date": selection.forecast_date,
            "requested_vintage_month": selection.requested_vintage_month,
            "selected_archive_month": selection.selected_archive_month,
            "archive_path": str(selection.archive_path),
            "archive_source_tag": selection.archive_source_tag,
            "asof_date": selection.asof_date,
            "source_policy": "ALFRED cache first, historical FRED-MD archive fallback only",
            "uses_latest_revised_fred_md": False,
        }
        return ForecastVintagePanel(raw=raw, transformed=transformed, source_tags=tags, metadata=metadata)

    def row_panel_for_dates(
        self,
        dates: Iterable[Any],
        *,
        start: Any = "1959-01-31",
        extra_drop_series: Iterable[str] = (),
    ) -> ForecastVintagePanel:
        """Return one transformed row per forecast date for inspection."""
        rows = []
        source_rows = []
        raw_rows = []
        metadata_rows = []
        for date in dates:
            panel = self.panel_for_forecast_date(
                date,
                start=start,
                end=date,
                extra_drop_series=extra_drop_series,
            )
            forecast_date = month_end(date)
            rows.append(panel.transformed.loc[[forecast_date]])
            raw_rows.append(panel.raw.loc[[forecast_date]])
            source_rows.append(panel.source_tags.loc[[forecast_date]])
            metadata_rows.append(panel.metadata)

        transformed = pd.concat(rows).sort_index()
        raw = pd.concat(raw_rows).sort_index()
        tags = pd.concat(source_rows).sort_index()
        metadata = {
            "panels": metadata_rows,
            "uses_latest_revised_fred_md": False,
        }
        return ForecastVintagePanel(raw=raw, transformed=transformed, source_tags=tags, metadata=metadata)


__all__ = [
    "ForecastVintageMacroStore",
    "ForecastVintagePanel",
    "VintageSelection",
    "SOURCE_ALFRED_ASOF",
    "SOURCE_ARCHIVE_FALLBACK",
    "SOURCE_EARLIEST_ARCHIVE_ANCHOR",
    "SOURCE_MISSING",
    "month_end",
]
