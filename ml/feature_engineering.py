"""
Feature Engineering Pipeline
==============================
Transforms raw shift-level steel plant data into model-ready features.

Features added:
    - Cyclical encoding of shift (A/B/C → sin/cos)
    - Calendar features from date (day_of_week, month, week_of_year)
    - Rolling averages (window=3) for temperature, yield, energy cost
    - Lag features: prev-shift yield_pct, scrap_rate_pct, unplanned_downtime_minutes
    - Grade-change interaction: grade_change_flag × scrap_ratio_pct
    - Time-since-last-maintenance proxy (cumulative shifts since last maintenance)

Usage:
    from ml.feature_engineering import build_features

    df_fe = build_features(df_raw)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Mapping from shift label to angle in radians (evenly spaced on unit circle)
SHIFT_ANGLE: dict[str, float] = {
    "A": 0.0,
    "B": 2 * np.pi / 3,
    "C": 4 * np.pi / 3,
}

#: Columns for which rolling mean (window=3) is computed
ROLLING_COLS: list[str] = [
    "avg_furnace_temperature_c",
    "yield_pct",
    "energy_cost_usd",
]

#: Columns for which lag-1 features are created
LAG_COLS: list[str] = [
    "yield_pct",
    "scrap_rate_pct",
    "unplanned_downtime_minutes",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_features(df: pd.DataFrame, sort: bool = True) -> pd.DataFrame:
    """Build the full feature-engineered DataFrame from raw shift data.

    The function is non-destructive — the original ``df`` is not modified.
    Rows are sorted chronologically (by ``date`` then ``shift``) before any
    lag/rolling computations so that temporal order is preserved.  The sort
    order is restored *only* if ``sort=False``.

    Parameters
    ----------
    df:
        Raw DataFrame loaded from the CSV dataset.  Expected columns include
        ``date``, ``shift``, ``avg_furnace_temperature_c``, ``yield_pct``,
        ``energy_cost_usd``, ``scrap_rate_pct``,
        ``unplanned_downtime_minutes``, ``grade_change_flag``,
        ``scrap_ratio_pct``, and ``maintenance_status``.
    sort:
        If ``True`` (default), return rows in chronological order.
        Set to ``False`` to preserve the original row order after features
        are computed.

    Returns
    -------
    pd.DataFrame
        Feature-engineered DataFrame with NaN rows from lag/rolling ops
        forward-filled where sensible, then dropped for remaining NaNs in
        lag columns.
    """
    df = df.copy()

    # ------------------------------------------------------------------ #
    # 1. Sort chronologically for temporal features                        #
    # ------------------------------------------------------------------ #
    shift_order = {"A": 0, "B": 1, "C": 2}
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["_shift_order"] = df["shift"].map(shift_order).fillna(0).astype(int)
        df = df.sort_values(["date", "_shift_order"]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 2. Calendar features from date                                       #
    # ------------------------------------------------------------------ #
    if "date" in df.columns:
        df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon … 6=Sun
        df["month"] = df["date"].dt.month                     # 1–12
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # ------------------------------------------------------------------ #
    # 3. Cyclical encoding of shift (A/B/C)                               #
    # ------------------------------------------------------------------ #
    if "shift" in df.columns:
        angles = df["shift"].map(SHIFT_ANGLE).fillna(0.0)
        df["shift_sin"] = np.sin(angles)
        df["shift_cos"] = np.cos(angles)

    # ------------------------------------------------------------------ #
    # 4. Rolling averages (window=3, min_periods=1)                        #
    # ------------------------------------------------------------------ #
    for col in ROLLING_COLS:
        if col in df.columns:
            df[f"{col}_roll3"] = (
                df[col]
                .rolling(window=3, min_periods=1)
                .mean()
            )

    # ------------------------------------------------------------------ #
    # 5. Lag features (previous shift's values)                            #
    # ------------------------------------------------------------------ #
    for col in LAG_COLS:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)

    # ------------------------------------------------------------------ #
    # 6. Grade-change interaction term                                     #
    # ------------------------------------------------------------------ #
    if "grade_change_flag" in df.columns and "scrap_ratio_pct" in df.columns:
        df["grade_change_x_scrap"] = (
            df["grade_change_flag"].astype(float) * df["scrap_ratio_pct"]
        )

    # ------------------------------------------------------------------ #
    # 7. Time-since-last-maintenance proxy                                 #
    #    Counts cumulative shifts since the last shift where               #
    #    maintenance_status == 1 (or "maintenance").                       #
    # ------------------------------------------------------------------ #
    if "maintenance_status" in df.columns:
        df["_is_maintenance"] = _parse_maintenance(df["maintenance_status"])
        df["shifts_since_maintenance"] = _time_since_event(df["_is_maintenance"])
        df.drop(columns=["_is_maintenance"], inplace=True)

    # ------------------------------------------------------------------ #
    # 8. Drop helper columns and rows with NaN in critical lag columns     #
    # ------------------------------------------------------------------ #
    if "_shift_order" in df.columns:
        df.drop(columns=["_shift_order"], inplace=True)

    # Drop the very first row(s) that have NaN lag values (no prior shift)
    lag_feature_cols = [f"{c}_lag1" for c in LAG_COLS if c in df.columns]
    existing_lag_cols = [c for c in lag_feature_cols if c in df.columns]
    if existing_lag_cols:
        df.dropna(subset=existing_lag_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return the canonical split of feature columns into controllable vs context.

    Parameters
    ----------
    df:
        Feature-engineered DataFrame (output of :func:`build_features`).

    Returns
    -------
    dict with keys ``"controllable"``, ``"context"``, ``"engineered"``,
    and ``"all_features"``.
    """
    controllable = [
        c for c in CONTROLLABLE_VARS if c in df.columns
    ]
    context = [
        c for c in CONTEXT_VARS if c in df.columns
    ]
    engineered = [
        c for c in df.columns
        if c.endswith("_roll3")
        or c.endswith("_lag1")
        or c in ("shift_sin", "shift_cos",
                 "day_of_week", "month", "week_of_year",
                 "grade_change_x_scrap", "shifts_since_maintenance")
    ]
    all_features = controllable + context + engineered
    # Deduplicate while preserving order
    seen: set[str] = set()
    all_features_dedup: list[str] = []
    for col in all_features:
        if col not in seen:
            seen.add(col)
            all_features_dedup.append(col)

    return {
        "controllable": controllable,
        "context": context,
        "engineered": engineered,
        "all_features": all_features_dedup,
    }


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

#: Variables the optimizer can control (setpoint levers)
CONTROLLABLE_VARS: list[str] = [
    "avg_furnace_temperature_c",
    "oxygen_flow_rate",
    "charge_weight_tons",
    "scrap_ratio_pct",
    "iron_ore_ratio_pct",
    "alloy_addition_kg",
    "flux_addition_kg",
    "num_furnaces_running",
    "labor_count",
    "planned_runtime_hours",
]

#: Variables that are observed but cannot be optimized
CONTEXT_VARS: list[str] = [
    "ambient_temperature_c",
    "humidity_pct",
    "raw_material_quality_index",
    "moisture_content_pct",
    "power_supply_stability_index",
    "product_grade",
    "operator_experience_level",
    "maintenance_status",
    "grade_change_flag",
]

#: Continuous regression targets
REGRESSION_TARGETS: list[str] = [
    "yield_pct",
    "steel_output_tons",
    "energy_cost_usd",
    "production_cost_usd",
    "scrap_rate_pct",
]

#: Binary classification targets
CLASSIFICATION_TARGETS: list[str] = [
    "quality_grade_pass",
    "rework_required",
]

ALL_TARGETS: list[str] = REGRESSION_TARGETS + CLASSIFICATION_TARGETS


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_maintenance(series: pd.Series) -> pd.Series:
    """Return a boolean Series: True where a maintenance event occurred."""
    # Handle numeric (1/0) or string ("maintenance"/"normal"/etc.)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float) == 1.0
    # String: treat "maintenance" or "yes" or "1" as True
    return series.astype(str).str.lower().isin({"1", "maintenance", "yes", "true"})


def _time_since_event(flag: pd.Series) -> pd.Series:
    """Count cumulative steps since last True in ``flag``.

    For each row *i*, returns the number of rows elapsed since the most
    recent row where ``flag`` was True.  If no prior event occurred, the
    value increases from 0 at the start of the series.
    """
    counter = np.zeros(len(flag), dtype=int)
    since = 0
    for i, is_event in enumerate(flag):
        if is_event:
            since = 0
        else:
            since += 1
        counter[i] = since
    return pd.Series(counter, index=flag.index)
