"""
Pipeline (monthly + yearly):
1) Compute avg visits PER MONTH and PER YEAR from 2 parquet datasets (CAT, RS).
2) Aggregate WAPE per target_code and forecast_horizon from runs.csv.
3) Join WAPE with avg visits WITHOUT row duplication:
   - exact match first
   - optional normalized fallback for unmatched targets (pick ONE best candidate)
4) Plots can use visits/month OR visits/year (selectable), with optional log transforms.

Fixes:
- Robust build_avg_visits_long(): NO MultiIndex melt (avoids melt var_name=['dataset','measure'] errors).
- Optional log transforms controlled by booleans.
- Interpretable bins via pd.cut (separate bins for month vs year).
- Includes hexbin + binned mean plots for each metric and for month/year.

Author: Guillem Hernández
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Sequence, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================

@dataclass(frozen=True)
class Paths:
    cat_parquet: Path
    rs_parquet: Path
    runs_csv: Path
    out_avg_visits_long: Path
    out_wape_agg: Path
    out_wape_with_visits_month: Path
    out_wape_with_visits_year: Path
    out_join_report_month: Path
    out_join_report_year: Path


DEFAULT_PATHS = Paths(
    cat_parquet=Path(r"C:\Users\Guillem\Desktop\TRANSFORMERS\data\FINAL_DB\full_CAT1.parquet"),
    rs_parquet=Path(r"C:\Users\Guillem\Desktop\TRANSFORMERS\data\FINAL_DB\full_RS1_clean_apostrofs.parquet"),
    runs_csv=Path(r"C:\Users\Guillem\Desktop\TRANSFORMERS\data\results\runs.csv"),
    out_avg_visits_long=Path("avg_visits_long_month_year.csv"),
    out_wape_agg=Path("wape_aggregated.csv"),
    out_wape_with_visits_month=Path("wape_with_avg_visits_month.csv"),
    out_wape_with_visits_year=Path("wape_with_avg_visits_year.csv"),
    out_join_report_month=Path("join_report_month.csv"),
    out_join_report_year=Path("join_report_year.csv"),
)

JOIN_DATASET_DEFAULT: Literal["CAT", "RS"] = "CAT"
ENABLE_NORMALIZED_FALLBACK = True

# Plot transforms
USE_LOG_X = False          # if True -> x = log10(visits)
USE_LOG_Y = False          # if True -> y = log10(WAPE + eps)
WAPE_EPS = 1e-6

# Which horizons to display in hexbin plots (edit freely)
HEXBIN_HORIZONS = (7, 30, 182, 365)

# Metrics to plot/analyze
METRIC_COLS = ["univ_mean_wape", "seasonal_mean_wape", "diagnostics_mean_wape"]

# Bins for month/year (different scale)
VISITS_MONTH_BINS = [0, 50, 200, 500, 2000, np.inf]
VISITS_MONTH_LABELS = ["<50", "50–200", "200–500", "500–2000", ">2000"]

VISITS_YEAR_BINS = [0, 500, 2000, 6000, 24000, np.inf]
VISITS_YEAR_LABELS = ["<500", "500–2k", "2k–6k", "6k–24k", ">24k"]


# =============================================================================
# HELPERS
# =============================================================================

def assert_columns_exist(df: pd.DataFrame, required: Iterable[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def read_parquet_checked(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} parquet not found: {path}")
    return pd.read_parquet(path)


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"runs.csv not found: {path}")
    return pd.read_csv(path)


# =============================================================================
# NORMALIZATION (fallback only)
# =============================================================================

_REGION_SUFFIX_RE = re.compile(r"_[A-ZÀ-Ÿ0-9]+_\d+$", flags=re.UNICODE)

def normalize_code(code: str) -> str:
    """Strip trailing region suffix like '_GIRONA_64'."""
    s = str(code).strip()
    return _REGION_SUFFIX_RE.sub("", s)


# =============================================================================
# AGGREGATION: VISITS PER MONTH / YEAR
# =============================================================================

def _coerce_targets_numeric(df: pd.DataFrame, timestamp_col: str) -> Tuple[pd.Series, pd.DataFrame]:
    """Return (timestamp_series, numeric_targets_df)."""
    assert_columns_exist(df, [timestamp_col], "parquet dataframe")
    X = df.copy()
    ts = pd.to_datetime(X[timestamp_col], errors="coerce")
    targets = X.drop(columns=[timestamp_col], errors="ignore").apply(pd.to_numeric, errors="coerce")
    return ts, targets


def compute_avg_visits_per_month(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.Series:
    """
    Avg visits per MONTH:
      - group by calendar month
      - sum within month
      - mean across months
    """
    ts, targets = _coerce_targets_numeric(df, timestamp_col)
    mask = ts.notna()
    ts = ts[mask]
    targets = targets.loc[mask]

    month_index = ts.dt.to_period("M")
    monthly_sum = targets.groupby(month_index).sum(min_count=1)
    return monthly_sum.mean(axis=0, skipna=True)


def compute_avg_visits_per_year(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.Series:
    """
    Avg visits per YEAR:
      - group by calendar year
      - sum within year
      - mean across years
    """
    ts, targets = _coerce_targets_numeric(df, timestamp_col)
    mask = ts.notna()
    ts = ts[mask]
    targets = targets.loc[mask]

    year_index = ts.dt.to_period("Y")
    yearly_sum = targets.groupby(year_index).sum(min_count=1)
    return yearly_sum.mean(axis=0, skipna=True)


def build_avg_visits_long(df_cat: pd.DataFrame, df_rs: pd.DataFrame) -> pd.DataFrame:
    """
    Robust (no MultiIndex melt):
    Columns:
      variable, dataset, avg_visits_per_month, avg_visits_per_year, variable_norm
    """
    # CAT series
    cat_m = compute_avg_visits_per_month(df_cat)
    cat_y = compute_avg_visits_per_year(df_cat)

    # RS series
    rs_m = compute_avg_visits_per_month(df_rs)
    rs_y = compute_avg_visits_per_year(df_rs)

    cat_df = pd.DataFrame(
        {
            "variable": cat_m.index.astype(str),
            "dataset": "CAT",
            "avg_visits_per_month": cat_m.to_numpy(),
            "avg_visits_per_year": cat_y.reindex(cat_m.index).to_numpy(),
        }
    )

    rs_df = pd.DataFrame(
        {
            "variable": rs_m.index.astype(str),
            "dataset": "RS",
            "avg_visits_per_month": rs_m.to_numpy(),
            "avg_visits_per_year": rs_y.reindex(rs_m.index).to_numpy(),
        }
    )

    out = (
        pd.concat([cat_df, rs_df], ignore_index=True)
        .assign(variable_norm=lambda d: d["variable"].map(normalize_code))
        .sort_values(["dataset", "variable"])
        .reset_index(drop=True)
    )
    return out


# =============================================================================
# WAPE AGGREGATION
# =============================================================================

def aggregate_wape(runs: pd.DataFrame) -> pd.DataFrame:
    required = [
        "target_code",
        "forecast_horizon",
        "eval/residual_seasonal_model_wape",
        "eval/residual_diagnostics_model_wape",
        "eval/univ_transformer_wape",
    ]
    assert_columns_exist(runs, required, "runs.csv")

    df = runs[required].copy()

    df["forecast_horizon"] = pd.to_numeric(df["forecast_horizon"], errors="coerce")

    for col in [
        "eval/residual_seasonal_model_wape",
        "eval/residual_diagnostics_model_wape",
        "eval/univ_transformer_wape",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["target_code", "forecast_horizon"], how="any")
    df = df.dropna(
        subset=[
            "eval/residual_seasonal_model_wape",
            "eval/residual_diagnostics_model_wape",
            "eval/univ_transformer_wape",
        ],
        how="all",
    )

    agg = (
        df.groupby(["target_code", "forecast_horizon"])
        .agg(
            seasonal_mean_wape=("eval/residual_seasonal_model_wape", "mean"),
            seasonal_std_wape=("eval/residual_seasonal_model_wape", "std"),
            seasonal_n=("eval/residual_seasonal_model_wape", "count"),

            diagnostics_mean_wape=("eval/residual_diagnostics_model_wape", "mean"),
            diagnostics_std_wape=("eval/residual_diagnostics_model_wape", "std"),
            diagnostics_n=("eval/residual_diagnostics_model_wape", "count"),

            univ_mean_wape=("eval/univ_transformer_wape", "mean"),
            univ_std_wape=("eval/univ_transformer_wape", "std"),
            univ_n=("eval/univ_transformer_wape", "count"),
        )
        .reset_index()
        .assign(target_code_norm=lambda d: d["target_code"].map(normalize_code))
        .sort_values(["target_code", "forecast_horizon"])
        .reset_index(drop=True)
    )

    # std undefined for n<=1 -> set 0.0 for clean tables
    agg.loc[agg["seasonal_n"] <= 1, "seasonal_std_wape"] = 0.0
    agg.loc[agg["diagnostics_n"] <= 1, "diagnostics_std_wape"] = 0.0
    agg.loc[agg["univ_n"] <= 1, "univ_std_wape"] = 0.0

    return agg


# =============================================================================
# JOIN (month/year selectable)
# =============================================================================

def _pick_best_fallback_candidate(candidates: pd.DataFrame, vol_col: str) -> Optional[pd.Series]:
    """
    Pick a single candidate to avoid row explosion.
    Strategy:
      1) require non-NaN volume
      2) take the max volume (usually the 'main' aggregate series)
    """
    if candidates.empty:
        return None
    c = candidates.dropna(subset=[vol_col])
    if c.empty:
        return None
    return c.sort_values(vol_col, ascending=False).iloc[0]


def join_wape_with_avg_visits(
    agg_wape: pd.DataFrame,
    avg_visits_long: pd.DataFrame,
    dataset: Literal["CAT", "RS"],
    volume_mode: Literal["month", "year"],
    enable_fallback: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Join WAPE with avg visits (month or year) without duplicating rows.
    """
    if volume_mode not in {"month", "year"}:
        raise ValueError("volume_mode must be 'month' or 'year'")

    vol_col = "avg_visits_per_month" if volume_mode == "month" else "avg_visits_per_year"

    assert_columns_exist(agg_wape, ["target_code", "target_code_norm"], "agg_wape")
    assert_columns_exist(avg_visits_long, ["dataset", "variable", "variable_norm", vol_col], "avg_visits_long")

    avg_ds = avg_visits_long.query("dataset == @dataset").copy()

    # A) Exact join
    joined = agg_wape.merge(
        avg_ds[["variable", vol_col]].rename(columns={"variable": "target_code"}),
        on="target_code",
        how="left",
    )

    report_rows = []
    missing_mask = joined[vol_col].isna()

    for code, miss in zip(joined["target_code"], missing_mask):
        report_rows.append(
            {
                "target_code": code,
                "match_type": "exact" if not miss else "missing_after_exact",
                "matched_variable": code if not miss else None,
            }
        )

    # B) Fallback only for missing
    if enable_fallback and missing_mask.any():
        vars_by_norm = avg_ds[["variable", "variable_norm", vol_col]]

        fixes = []
        for idx in joined.index[missing_mask]:
            code = joined.at[idx, "target_code"]
            norm = joined.at[idx, "target_code_norm"]

            candidates = vars_by_norm[vars_by_norm["variable_norm"] == norm]
            best = _pick_best_fallback_candidate(candidates, vol_col)

            if best is not None:
                joined.at[idx, vol_col] = best[vol_col]
                fixes.append((code, best["variable"], "fallback_norm"))
            else:
                fixes.append((code, None, "no_match"))

        fix_map = {c: (mv, mt) for c, mv, mt in fixes}
        for r in report_rows:
            if r["match_type"] == "missing_after_exact":
                mv, mt = fix_map.get(r["target_code"], (None, "no_match"))
                r["match_type"] = mt
                r["matched_variable"] = mv

    return joined, pd.DataFrame(report_rows)


# =============================================================================
# PLOTTING
# =============================================================================

def _transform_x(x: np.ndarray, label: str) -> Tuple[np.ndarray, str]:
    if USE_LOG_X:
        return np.log10(x), f"log10({label})"
    return x, label


def _transform_y(y: np.ndarray, metric_col: str) -> Tuple[np.ndarray, str]:
    if USE_LOG_Y:
        return np.log10(np.maximum(y, 0.0) + WAPE_EPS), f"log10({metric_col} + {WAPE_EPS:g})"
    return y, metric_col


def prep_plot_df(df: pd.DataFrame, metric_col: str, volume_mode: Literal["month", "year"]) -> pd.DataFrame:
    vol_col = "avg_visits_per_month" if volume_mode == "month" else "avg_visits_per_year"
    vol_label = "avg visits/month" if volume_mode == "month" else "avg visits/year"

    assert_columns_exist(df, [vol_col, "forecast_horizon", metric_col], "joined dataframe")

    d = df.dropna(subset=[vol_col, "forecast_horizon", metric_col]).copy()
    d = d[d[vol_col] > 0].copy()

    x_raw = d[vol_col].to_numpy(dtype=float)
    y_raw = pd.to_numeric(d[metric_col], errors="coerce").to_numpy(dtype=float)

    x, x_label = _transform_x(x_raw, vol_label)
    y, y_label = _transform_y(y_raw, metric_col)

    d["x_visits"] = x
    d["y_err"] = y
    d["x_label"] = x_label
    d["y_label"] = y_label
    d["vol_col"] = vol_col
    return d


def plot_hexbin_by_horizon(
    df: pd.DataFrame,
    metric_col: str,
    volume_mode: Literal["month", "year"],
    horizons: Sequence[int] = HEXBIN_HORIZONS,
    gridsize: int = 35,
) -> None:
    d = prep_plot_df(df, metric_col, volume_mode)
    d["forecast_horizon"] = d["forecast_horizon"].astype(int)

    horizons = [h for h in horizons if h in set(d["forecast_horizon"])]
    n = len(horizons)

    fig = plt.figure(figsize=(12, 3.5 * max(1, n)))
    for i, h in enumerate(horizons, 1):
        ax = fig.add_subplot(n, 1, i)
        sub = d[d["forecast_horizon"] == int(h)]
        hb = ax.hexbin(sub["x_visits"], sub["y_err"], gridsize=gridsize, mincnt=1)
        ax.set_title(f"{metric_col} | horizon={h} | volume={volume_mode}")
        ax.set_xlabel(d["x_label"].iloc[0])
        ax.set_ylabel(d["y_label"].iloc[0])
        fig.colorbar(hb, ax=ax, label="count")
    plt.tight_layout()
    plt.show()


def plot_binned_means_cut(
    df: pd.DataFrame,
    metric_col: str,
    volume_mode: Literal["month", "year"],
) -> None:
    d = prep_plot_df(df, metric_col, volume_mode)

    if volume_mode == "month":
        bins, labels = VISITS_MONTH_BINS, VISITS_MONTH_LABELS
        raw_label = "avg visits/month"
    else:
        bins, labels = VISITS_YEAR_BINS, VISITS_YEAR_LABELS
        raw_label = "avg visits/year"

    if len(labels) != (len(bins) - 1):
        raise ValueError("labels must have len(bins) - 1 elements")

    vol_col = d["vol_col"].iloc[0]
    d["visits_bin"] = pd.cut(d[vol_col], bins=bins, labels=labels, right=False, include_lowest=True)

    summary = (
        d.groupby(["forecast_horizon", "visits_bin"], observed=True)
         .agg(
             mean_y=("y_err", "mean"),
             mean_visits=(vol_col, "mean"),
             n=(vol_col, "count"),
         )
         .reset_index()
    )

    horizons = sorted(summary["forecast_horizon"].unique())

    plt.figure()
    for h in horizons:
        sub = summary[summary["forecast_horizon"] == h].copy()
        sub["visits_bin"] = pd.Categorical(sub["visits_bin"], categories=labels, ordered=True)
        sub = sub.sort_values("visits_bin")

        x_raw = sub["mean_visits"].to_numpy(dtype=float)
        x, x_label = _transform_x(x_raw, raw_label)
        plt.plot(x, sub["mean_y"], marker="o", label=f"fh={int(h)}")

    plt.xlabel(x_label + " (bin means)")
    plt.ylabel(d["y_label"].iloc[0] + " (mean)")
    plt.title(f"Binned (pd.cut): {metric_col} vs {raw_label} | volume={volume_mode}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\n[{metric_col} | {volume_mode}] counts per visits_bin (pooled):")
    print(d["visits_bin"].value_counts(dropna=False))


# =============================================================================
# MAIN
# =============================================================================

def main(paths: Paths = DEFAULT_PATHS, join_dataset: Literal["CAT", "RS"] = JOIN_DATASET_DEFAULT) -> int:
    try:
        df_cat = read_parquet_checked(paths.cat_parquet, "CAT")
        df_rs = read_parquet_checked(paths.rs_parquet, "RS")
        runs = read_csv_checked(paths.runs_csv)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Optional de-dup if present
    if "Run ID" in runs.columns:
        runs = runs.drop_duplicates(subset=["Run ID"])

    # 1) Avg visits (month + year)
    avg_visits_long = build_avg_visits_long(df_cat, df_rs)
    avg_visits_long.to_csv(paths.out_avg_visits_long, index=False)
    print(f"Saved: {paths.out_avg_visits_long.resolve()}  (rows={len(avg_visits_long):,})")

    # 2) WAPE aggregation
    agg_wape = aggregate_wape(runs)
    agg_wape.to_csv(paths.out_wape_agg, index=False)
    print(f"Saved: {paths.out_wape_agg.resolve()}  (rows={len(agg_wape):,})")

    # 3) Join twice (month + year)
    joined_month, report_month = join_wape_with_avg_visits(
        agg_wape=agg_wape,
        avg_visits_long=avg_visits_long,
        dataset=join_dataset,
        volume_mode="month",
        enable_fallback=ENABLE_NORMALIZED_FALLBACK,
    )
    joined_year, report_year = join_wape_with_avg_visits(
        agg_wape=agg_wape,
        avg_visits_long=avg_visits_long,
        dataset=join_dataset,
        volume_mode="year",
        enable_fallback=ENABLE_NORMALIZED_FALLBACK,
    )

    joined_month.to_csv(paths.out_wape_with_visits_month, index=False)
    report_month.to_csv(paths.out_join_report_month, index=False)
    print(f"Saved: {paths.out_wape_with_visits_month.resolve()}  (rows={len(joined_month):,})")
    print(f"Saved: {paths.out_join_report_month.resolve()}  (rows={len(report_month):,})")

    joined_year.to_csv(paths.out_wape_with_visits_year, index=False)
    report_year.to_csv(paths.out_join_report_year, index=False)
    print(f"Saved: {paths.out_wape_with_visits_year.resolve()}  (rows={len(joined_year):,})")
    print(f"Saved: {paths.out_join_report_year.resolve()}  (rows={len(report_year):,})")

    print("\nJoin report summary (month):")
    print(report_month["match_type"].value_counts(dropna=False))
    print("\nJoin report summary (year):")
    print(report_year["match_type"].value_counts(dropna=False))

    # 4) Plots for BOTH month and year
    for mode, joined in [("month", joined_month), ("year", joined_year)]:
        for m in METRIC_COLS:
            plot_hexbin_by_horizon(joined, m, volume_mode=mode, horizons=HEXBIN_HORIZONS)
            plot_binned_means_cut(joined, m, volume_mode=mode)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
