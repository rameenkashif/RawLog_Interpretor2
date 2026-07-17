"""
petrophysics.py
================
Petrophysical interpretation engine.

Each function below implements exactly ONE calculation, documented with its
formula, so it can be unit-tested and swapped out independently. Every
tunable constant (percentiles, matrix density, Rw, Archie exponents,
Swirr, zone cutoffs, ...) is supplied via a `config` dict produced by
`app.config_loader.get_well_config()` -- nothing is hardcoded inline.

Input: a `pandas.DataFrame` with (at minimum) the raw curves
    DEPT, GR, RESISTIVITY, RHOB, NPHI, DT
Output: the same DataFrame with additional computed curve columns appended.

NOTE ON ASSUMPTIONS
--------------------
Several calculations below are proxies/heuristics used because only five
raw curves are available (no PEF, no core plugs, no checkshot/VSP, no
deviation survey). Each function's docstring calls out the assumption
explicitly. These should be reviewed/calibrated by a petrophysicist (SME)
before being used for reserves or completion decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NULL_VALUE = -9999.25  # Standard LAS null value sentinel


# -----------------------------------------------------------------------------
# 3.1 VSH -- Volume of Shale
# -----------------------------------------------------------------------------
def compute_vsh(df: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    """Volume of shale from the Gamma Ray Index with a Larionov correction.

    Gamma Ray Index:
        IGR = (GR - GR_clean) / (GR_shale - GR_clean)

    GR_clean / GR_shale default to the 5th / 95th percentile of the well's
    own GR curve (a common, log-agnostic way to bound the "clean sand" and
    "shale baseline" GR response without hardcoding a field-wide value).
    An explicit override can be supplied via config instead.

    Larionov correction (config["vsh"]["method"]):
        "older"    : VSH = 0.33 * (2^(2*IGR)   - 1)   (consolidated / older rocks)
        "tertiary" : VSH = 0.083 * (2^(3.7*IGR) - 1)  (Tertiary / unconsolidated)

    Result is clipped to [0, 1].
    """
    vsh_cfg = config["vsh"]
    gr = df["GR"].to_numpy(dtype=float)

    if vsh_cfg.get("use_percentiles", True):
        gr_clean = np.nanpercentile(gr, vsh_cfg.get("gr_clean_percentile", 5))
        gr_shale = np.nanpercentile(gr, vsh_cfg.get("gr_shale_percentile", 95))
    else:
        gr_clean = vsh_cfg["gr_clean_override"]
        gr_shale = vsh_cfg["gr_shale_override"]

    # Guard against a degenerate (near-constant) GR curve.
    denom = gr_shale - gr_clean
    if denom == 0 or np.isnan(denom):
        igr = np.zeros_like(gr)
    else:
        igr = (gr - gr_clean) / denom
    igr = np.clip(igr, 0.0, 1.0)

    method = vsh_cfg.get("method", "older")
    if method == "tertiary":
        vsh = 0.083 * (2 ** (3.7 * igr) - 1)
    else:
        vsh = 0.33 * (2 ** (2 * igr) - 1)

    vsh = np.clip(vsh, 0.0, 1.0)
    return pd.Series(vsh, index=df.index, name="VSH")


# -----------------------------------------------------------------------------
# 3.2 PHIT -- Total Porosity (density-derived)
# -----------------------------------------------------------------------------
def compute_phit(df: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    """Total porosity from the density log.

        PHIT = (RHOB_matrix - RHOB) / (RHOB_matrix - RHOB_fluid)

    RHOB_matrix defaults to 2.65 g/cc (sandstone). Common presets:
        sandstone = 2.65, limestone = 2.71, dolomite = 2.87
    RHOB_fluid defaults to 1.0 g/cc (fresh mud filtrate).

    Not clipped here -- PHIT can legitimately be used unclipped as an
    intermediate value for PHIE, but we clip to a physically sane [0, 0.5]
    band to guard against washouts / bad RHOB readings blowing up downstream
    calculations.
    """
    phit_cfg = config["phit"]
    rhob_matrix = phit_cfg.get("rhob_matrix", 2.65)
    rhob_fluid = phit_cfg.get("rhob_fluid", 1.0)

    rhob = df["RHOB"].to_numpy(dtype=float)
    phit = (rhob_matrix - rhob) / (rhob_matrix - rhob_fluid)
    phit = np.clip(phit, 0.0, 0.5)
    return pd.Series(phit, index=df.index, name="PHIT")


# -----------------------------------------------------------------------------
# 3.3 PHIE -- Effective Porosity (shale-corrected)
# -----------------------------------------------------------------------------
def compute_phie(
    df: pd.DataFrame,
    vsh: pd.Series,
    phit: pd.Series,
    config: dict[str, Any],
) -> pd.Series:
    """Effective (shale-corrected) porosity.

        PHIE = PHIT - (VSH * PHIT_shale)

    PHIT_shale is PHIT evaluated at the shale baseline point, approximated
    here as the PHIT value at the depth where VSH is at its maximum (i.e.
    the "purest shale" interval logged in this well). This avoids needing a
    separate hardcoded shale-point density.

    Result is clipped to [0, PHIT] (PHIE can never physically exceed PHIT).
    """
    phit_arr = phit.to_numpy(dtype=float)
    vsh_arr = vsh.to_numpy(dtype=float)

    if len(phit_arr) == 0 or np.all(np.isnan(vsh_arr)):
        phit_shale = 0.0
    else:
        shale_idx = np.nanargmax(vsh_arr)
        phit_shale = phit_arr[shale_idx]

    phie = phit_arr - (vsh_arr * phit_shale)
    phie = np.clip(phie, 0.0, phit_arr)
    return pd.Series(phie, index=df.index, name="PHIE")


def compute_phie_density_neutron(
    df: pd.DataFrame, phit: pd.Series, config: dict[str, Any]
) -> pd.Series:
    """Cross-check PHIE using a density-neutron combination.

        PHID = PHIT (density porosity, computed already)
        PHIN = NPHI (neutron porosity, read directly off the log)

        PHIE_DN = (PHID + PHIN) / 2                              if PHIN >= PHID
        PHIE_DN = sqrt((PHID^2 + PHIN^2) / 2)  (gas-corrected)    if PHIN <  PHID

    The gas-corrected (root-mean-square) form suppresses the porosity
    over-estimation that occurs in gas-bearing zones, where neutron
    porosity reads lower than density porosity due to the neutron
    (hydrogen-index) gas effect.
    """
    phid = phit.to_numpy(dtype=float)
    phin = df["NPHI"].to_numpy(dtype=float)

    avg = (phid + phin) / 2.0
    gas_corrected = np.sqrt((phid**2 + phin**2) / 2.0)
    phie_dn = np.where(phin < phid, gas_corrected, avg)
    phie_dn = np.clip(phie_dn, 0.0, 0.5)
    return pd.Series(phie_dn, index=df.index, name="PHIE_DN")


# -----------------------------------------------------------------------------
# 3.4 SWE -- Water Saturation (Archie's Equation)
# -----------------------------------------------------------------------------
def compute_swe(df: pd.DataFrame, phie: pd.Series, config: dict[str, Any]) -> pd.Series:
    """Water saturation via Archie's equation.

        Sw = ((a * Rw) / (PHIE^m * Rt)) ^ (1/n)

    Defaults: a = 1, m = 2, n = 2 (clean-sand Archie parameters).
    Rt is the deep RESISTIVITY curve. Rw (formation water resistivity at
    formation temperature) MUST be supplied per well/field via config --
    it is never hardcoded here, since it varies with formation water
    salinity and temperature gradient.

    Result is clipped to [0, 1]. Zero/near-zero PHIE intervals (e.g. tight
    shale) are guarded against division blow-up by treating them as Sw = 1
    (fully water-saturated / non-reservoir), which is the physically
    sensible answer for a rock with no effective porosity.
    """
    swe_cfg = config["swe"]
    a = swe_cfg.get("a", 1.0)
    m = swe_cfg.get("m", 2.0)
    n = swe_cfg.get("n", 2.0)
    rw = swe_cfg["rw"]

    rt = df["RESISTIVITY"].to_numpy(dtype=float)
    phie_arr = phie.to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        sw = ((a * rw) / (np.power(phie_arr, m) * rt)) ** (1.0 / n)

    # Non-physical results (division by ~0 porosity, negative resistivity, etc.)
    sw = np.where(phie_arr <= 1e-4, 1.0, sw)
    sw = np.nan_to_num(sw, nan=1.0, posinf=1.0, neginf=1.0)
    sw = np.clip(sw, 0.0, 1.0)
    return pd.Series(sw, index=df.index, name="SWE")


# -----------------------------------------------------------------------------
# 3.5 DPTM -- Depth/Time Track (sonic-integration approximation)
# -----------------------------------------------------------------------------
DPTM_VENDOR_MIN_VALID_FRACTION = 0.8


def compute_dptm(
    df: pd.DataFrame, config: dict[str, Any], step_depth: float
) -> pd.Series:
    """Two-way-time depth/time track for this well.

    Prefers a vendor-precomputed DPTM curve straight from the LAS file (see
    las_loader.py's OPTIONAL_CURVES) when one is present and mostly valid:
    a real, already-calibrated-to-the-seismic-datum time-depth curve beats
    any approximation we could derive here, and confirming it independently
    (e.g. checking it falls inside the seismic's recorded TWT window) is
    exactly what well_seismic_tie's tie search does downstream by actually
    correlating against the seismic. Falls back to sonic integration below
    for wells whose LAS export doesn't carry one.

    *** SONIC-INTEGRATION FALLBACK IS AN APPROXIMATION ONLY -- pending real
    checkshot/VSP data. *** If a real checkshot/VSP time-depth table is
    available, it should be used instead of either path here; this
    integration accumulates sonic logging error and does not account for
    velocity anisotropy.

        TWT_increment = DT * step_depth / (2 * 3.28084 * 1e6)   [seconds]
        DPTM = cumulative_sum(TWT_increment)

    DT is in us/ft; step_depth (m) is converted to ft via the 3.28084
    factor. The factor of 2 converts one-way time to two-way time, and the
    1e6 converts microseconds to seconds.
    """
    if not config.get("dptm", {}).get("enabled", True):
        return pd.Series(np.full(len(df), np.nan), index=df.index, name="DPTM")

    if "DPTM" in df.columns:
        vendor = df["DPTM"].to_numpy(dtype=float)
        valid = np.isfinite(vendor) & (vendor > 0)
        min_valid_fraction = config.get("dptm", {}).get(
            "vendor_min_valid_fraction", DPTM_VENDOR_MIN_VALID_FRACTION
        )
        if len(vendor) > 0 and valid.sum() >= min_valid_fraction * len(vendor):
            return pd.Series(vendor, index=df.index, name="DPTM")

    dt = df["DT"].to_numpy(dtype=float)
    dt_filled = np.nan_to_num(dt, nan=0.0)
    twt_increment = dt_filled * step_depth / (2 * 3.28084 * 1e6)
    dptm = np.cumsum(twt_increment)
    return pd.Series(dptm, index=df.index, name="DPTM")


# -----------------------------------------------------------------------------
# 3.6 MD / TVD -- placeholder pending deviation survey
# -----------------------------------------------------------------------------
def compute_md_tvd(
    df: pd.DataFrame, deviation_survey: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Measured Depth / True Vertical Depth.

    *** PLACEHOLDER ASSUMPTION ***
    Without a deviation survey (inclination/azimuth vs depth), this module
    assumes the well is vertical/near-vertical, so:
        TVD = MD = DEPT

    If `deviation_survey` (columns: MD, INCL, AZIM) is supplied, TVD should
    be computed via the minimum curvature method:
        beta   = arccos(cos(I1-I2) - sin(I1)*sin(I2)*(1-cos(A2-A1)))
        RF     = (2/beta) * tan(beta/2)               (ratio factor)
        dTVD   = (dMD/2) * (cos(I1) + cos(I2)) * RF

    This is left as a documented extension point -- minimum curvature is
    not implemented yet because no deviation survey is available for
    Z-02..Z-08.
    """
    result = df.copy()
    result["MD"] = df["DEPT"]

    if deviation_survey is None or deviation_survey.empty:
        result["TVD"] = df["DEPT"]
        return result

    raise NotImplementedError(
        "Minimum curvature TVD calculation is not yet implemented. "
        "Supply deviation_survey=None until this extension point is built."
    )


# -----------------------------------------------------------------------------
# 3.7 PERM_TIXIER -- Tixier Permeability
# -----------------------------------------------------------------------------
def compute_perm_tixier(
    df: pd.DataFrame, phie: pd.Series, swe: pd.Series, config: dict[str, Any]
) -> pd.Series:
    """Tixier permeability estimate (medium-gravity oil).

        K = 250 * (PHIE^3 / Swirr)^2      [mD]

    Swirr (irreducible water saturation) is either:
      - supplied directly via config["perm_tixier"]["swirr"], or
      - estimated automatically as the SWE value in the cleanest interval
        (i.e. the depth with the highest PHIE in this well), falling back
        to config["perm_tixier"]["swirr_default"] if that can't be
        determined (e.g. an all-null well).
    """
    perm_cfg = config["perm_tixier"]
    swirr = perm_cfg.get("swirr")

    phie_arr = phie.to_numpy(dtype=float)
    swe_arr = swe.to_numpy(dtype=float)

    if swirr is None:
        if len(phie_arr) == 0 or np.all(np.isnan(phie_arr)):
            swirr = perm_cfg.get("swirr_default", 0.25)
        else:
            best_idx = np.nanargmax(phie_arr)
            swirr = swe_arr[best_idx]
            if np.isnan(swirr) or swirr <= 0:
                swirr = perm_cfg.get("swirr_default", 0.25)

    swirr = max(swirr, 1e-3)  # guard against divide-by-zero
    k = 250.0 * (np.power(phie_arr, 3) / swirr) ** 2
    k = np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(k, index=df.index, name="PERM_TIXIER")


# -----------------------------------------------------------------------------
# 3.8 CORE_PERM_PRED -- Predicted Core Permeability (proxy regression model)
# -----------------------------------------------------------------------------
def train_core_perm_model(
    training_df: pd.DataFrame,
    config: dict[str, Any],
    target_col: str = "PERM_TIXIER",
):
    """Train a RandomForestRegressor to predict core permeability.

    *** PROXY MODEL ***
    No real core plug measurements are available yet, so `target_col`
    defaults to PERM_TIXIER itself -- i.e. this model currently learns to
    reproduce the Tixier estimate from PHIE/VSH/PERM_TIXIER. This is a
    scaffold only: as soon as real core permeability measurements are
    available, retrain with `target_col="CORE_PERM_MEASURED"` (or
    equivalent) to get a genuine core-calibrated permeability predictor.

    Features: PHIE, VSH, PERM_TIXIER (log10-transformed to linearize the
    permeability range before training, since permeability is
    log-normally distributed).
    """
    from sklearn.ensemble import RandomForestRegressor

    perm_cfg = config.get("core_perm_pred", {})
    features = training_df[["PHIE", "VSH", "PERM_TIXIER"]].copy()
    features["PERM_TIXIER"] = np.log10(features["PERM_TIXIER"].clip(lower=1e-3))

    target = np.log10(training_df[target_col].clip(lower=1e-3))

    valid = features.notna().all(axis=1) & target.notna()

    model = RandomForestRegressor(
        n_estimators=perm_cfg.get("n_estimators", 200),
        max_depth=perm_cfg.get("max_depth", 8),
        random_state=perm_cfg.get("random_state", 42),
    )
    model.fit(features[valid], target[valid])
    return model


def predict_core_perm(
    df: pd.DataFrame, model, phie_col: str = "PHIE", vsh_col: str = "VSH"
) -> pd.Series:
    """Predict CORE_PERM_PRED (mD) from a trained model (see train_core_perm_model).

    Inverse-transforms the log10 target back to linear mD before returning.
    """
    features = pd.DataFrame(
        {
            "PHIE": df[phie_col],
            "VSH": df[vsh_col],
            "PERM_TIXIER": np.log10(df["PERM_TIXIER"].clip(lower=1e-3)),
        }
    )
    valid = features.notna().all(axis=1)

    pred = np.full(len(df), np.nan)
    if valid.any():
        pred[valid.to_numpy()] = model.predict(features[valid])

    pred_mD = np.where(np.isnan(pred), np.nan, 10**pred)
    return pd.Series(pred_mD, index=df.index, name="CORE_PERM_PRED")


def save_model(model, model_path: str | Path) -> None:
    """Persist a trained sklearn model to disk (joblib)."""
    import joblib

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: str | Path):
    """Load a previously trained sklearn model from disk (joblib)."""
    import joblib

    return joblib.load(model_path)


# -----------------------------------------------------------------------------
# 3.9 VVOLC -- Volume of Volcanics (density-neutron crossplot heuristic)
# -----------------------------------------------------------------------------
def compute_vvolc(
    df: pd.DataFrame, vsh: pd.Series, config: dict[str, Any]
) -> pd.Series:
    """Heuristic volume-of-volcanics proxy from a density-neutron crossplot rule.

    *** HEURISTIC LITHOLOGY PROXY -- NOT true mineralogical decomposition. ***
    No PEF curve is available to properly separate volcanics from other
    high-density, low-porosity lithologies, so this is a rule-of-thumb
    flag only. It should be calibrated against cuttings/core descriptions
    before being relied on for anything beyond a first-pass lithology
    screen.

    Flag as a volcanic candidate where:
        RHOB > rhob_threshold (2.7)  AND
        NPHI < nphi_threshold (0.15) AND
        GR   < gr_shale_fraction (0.6) * GR_shale

    Where flagged, VVOLC is scaled linearly within the RHOB range
    [rhob_scale_min, rhob_scale_max] -> [0, 1]. Elsewhere VVOLC = 0.
    """
    vvolc_cfg = config["vvolc"]
    vsh_cfg = config["vsh"]

    rhob = df["RHOB"].to_numpy(dtype=float)
    nphi = df["NPHI"].to_numpy(dtype=float)
    gr = df["GR"].to_numpy(dtype=float)

    if vsh_cfg.get("use_percentiles", True):
        gr_shale = np.nanpercentile(gr, vsh_cfg.get("gr_shale_percentile", 95))
    else:
        gr_shale = vsh_cfg["gr_shale_override"]

    is_candidate = (
        (rhob > vvolc_cfg.get("rhob_threshold", 2.7))
        & (nphi < vvolc_cfg.get("nphi_threshold", 0.15))
        & (gr < vvolc_cfg.get("gr_shale_fraction", 0.6) * gr_shale)
    )

    scale_min = vvolc_cfg.get("rhob_scale_min", 2.7)
    scale_max = vvolc_cfg.get("rhob_scale_max", 2.9)
    scaled = (rhob - scale_min) / (scale_max - scale_min)
    scaled = np.clip(scaled, 0.0, 1.0)

    vvolc = np.where(is_candidate, scaled, 0.0)
    return pd.Series(vvolc, index=df.index, name="VVOLC")


# -----------------------------------------------------------------------------
# 3.10 ZONES -- Reservoir Zonation
# -----------------------------------------------------------------------------
ZONE_PAY = 1
ZONE_RESERVOIR_NON_PAY = 2
ZONE_NON_RESERVOIR = 3

ZONE_LABELS = {
    ZONE_PAY: "Pay",
    ZONE_RESERVOIR_NON_PAY: "Reservoir (non-pay)",
    ZONE_NON_RESERVOIR: "Non-reservoir",
}


def compute_zones(
    df: pd.DataFrame,
    vsh: pd.Series,
    phie: pd.Series,
    swe: pd.Series,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Categorical reservoir zonation.

        Reservoir     : VSH < vsh_max  AND PHIE > phie_min AND SWE < swe_reservoir_max
        Pay           : Reservoir AND SWE < swe_pay_max
        Non-reservoir : everything else

    Returns a DataFrame with two columns:
        ZONES       : categorical code (1 = Pay, 2 = Reservoir non-pay, 3 = Non-reservoir)
        ZONES_LABEL : human-readable label matching the code above

    All cutoffs are configurable per well/field via config["zones"].
    """
    zones_cfg = config["zones"]
    vsh_max = zones_cfg.get("vsh_max", 0.4)
    phie_min = zones_cfg.get("phie_min", 0.08)
    swe_reservoir_max = zones_cfg.get("swe_reservoir_max", 0.65)
    swe_pay_max = zones_cfg.get("swe_pay_max", 0.5)

    vsh_arr = vsh.to_numpy(dtype=float)
    phie_arr = phie.to_numpy(dtype=float)
    swe_arr = swe.to_numpy(dtype=float)

    is_reservoir = (
        (vsh_arr < vsh_max) & (phie_arr > phie_min) & (swe_arr < swe_reservoir_max)
    )
    is_pay = is_reservoir & (swe_arr < swe_pay_max)

    zones = np.full(len(df), ZONE_NON_RESERVOIR, dtype=int)
    zones[is_reservoir] = ZONE_RESERVOIR_NON_PAY
    zones[is_pay] = ZONE_PAY

    labels = [ZONE_LABELS[z] for z in zones]

    return pd.DataFrame(
        {"ZONES": zones, "ZONES_LABEL": labels},
        index=df.index,
    )


# -----------------------------------------------------------------------------
# Orchestration -- run the full interpretation pipeline for one well
# -----------------------------------------------------------------------------
def run_full_interpretation(
    df: pd.DataFrame,
    config: dict[str, Any],
    step_depth: float,
    deviation_survey: pd.DataFrame | None = None,
    core_perm_model=None,
) -> pd.DataFrame:
    """Run the complete petrophysical interpretation pipeline on one well's
    raw curve DataFrame and return a new DataFrame with all computed curves
    appended. This is the single entry point routers/services should call.
    """
    result = df.copy()

    result = compute_md_tvd(result, deviation_survey)

    vsh = compute_vsh(result, config)
    result["VSH"] = vsh

    phit = compute_phit(result, config)
    result["PHIT"] = phit

    phie = compute_phie(result, vsh, phit, config)
    result["PHIE"] = phie

    if config.get("phie", {}).get("compute_density_neutron_crosscheck", True):
        result["PHIE_DN"] = compute_phie_density_neutron(result, phit, config)

    swe = compute_swe(result, phie, config)
    result["SWE"] = swe

    result["DPTM"] = compute_dptm(result, config, step_depth)

    perm_tixier = compute_perm_tixier(result, phie, swe, config)
    result["PERM_TIXIER"] = perm_tixier

    if core_perm_model is not None:
        result["CORE_PERM_PRED"] = predict_core_perm(result, core_perm_model)

    result["VVOLC"] = compute_vvolc(result, vsh, config)

    zones_df = compute_zones(result, vsh, phie, swe, config)
    result["ZONES"] = zones_df["ZONES"]
    result["ZONES_LABEL"] = zones_df["ZONES_LABEL"]

    return result
