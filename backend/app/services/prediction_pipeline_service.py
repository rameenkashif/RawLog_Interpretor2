"""
services/prediction_pipeline_service.py
------------------------------------------
Blind-well VSH/PHIE/SWE prediction from CWT/SSWT amplitude via Ridge
regression, plus a true-vs-predicted inline section image -- ported from
a user-supplied reference pipeline (full_pipeline.py), with ONLY the
well->trace tie/coordinate-calibration step (that script's steps 3-4)
replaced: this module resolves each well via direct_tie_service.
resolve_direct_tie (the proven direct nearest-trace + DPTM full-window-
search tie already used by spectral_property_prediction_service.py, and
matching what tie_service.get_well_seismic_tie validates on the
Well-to-Seismic Tie page) instead of the reference script's own
coordinate transform, which used hand-fit, survey-specific magic-number
anchors (X0=363124, Y0=2949830, XSTEP=YSTEP=30) with no independent
validation.

Deliberately a SEPARATE model/page from spectral_property_prediction_
service.py, not a replacement -- everything else here intentionally
mirrors the reference script's own choices rather than this app's
existing spectral-prediction module:
- Ridge regression (StandardScaler + Ridge(alpha=5.0), predictions
  clipped to [0,1]), not RandomForestRegressor.
- ONE fixed blind well held out at a time (as requested/named by the
  caller), not a full leave-one-well-out sweep across every well.
- Feature/target alignment is by NEAREST seismic time-sample index with
  duplicate-depth averaging (matching the reference script exactly), not
  the continuous depth<->time interpolation
  spectral_property_prediction_service.py uses.

CWT/SSWT amplitude itself is still computed via seismic_processor.
get_spectral_decomposition_trace (this app's existing, tested
implementation, built on the same PyWavelets/ssqueezepy stack the
reference script called directly) rather than re-implementing it, so
this module's numbers stay consistent with every other CWT/SSWT view in
the app.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app import well_seismic_tie as wst
from app.services import direct_tie_service as dts
from app.services import well_service

# Same regularization the reference script used -- not independently
# tuned here.
RIDGE_ALPHA = 5.0

TARGET_LAS_NAMES = {"vsh": "VSH", "phie": "PHIE", "swe": "SWE"}
METHOD_ENERGY_KEYS = {"cwt": "energy", "sswt": "sswt_amplitude"}
METHOD_FREQ_KEYS = {"cwt": "freq_hz", "sswt": "sswt_freq_hz"}

MIN_VALID_SAMPLES = 5


@dataclass
class WellPredictionFeatures:
    well_id: str
    inline_number: int
    crossline_number: int
    correlation: float
    best_freq_hz: float
    polarity: int
    bulk_shift_ms: float
    depths_m: np.ndarray
    twt_ms: np.ndarray
    targets: dict[str, np.ndarray]  # 'vsh'/'phie'/'swe' -> (n_samples,), aligned to depths_m/twt_ms
    features: dict[str, np.ndarray]  # 'cwt'/'sswt' -> (n_samples, n_freq)
    freq_hz: dict[str, np.ndarray]  # 'cwt'/'sswt' -> (n_freq,)


def _extract_curve(rows: list[dict], name: str) -> np.ndarray:
    arr = np.array(
        [row.get(name) if row.get(name) is not None else np.nan for row in rows], dtype=float
    )
    arr[arr <= -9999.0] = np.nan  # guard against LAS null sentinel leaking through
    return arr


def _build_well_features(volume, well_id: str) -> WellPredictionFeatures:
    """Raises TieError/WellNotFoundError/SegyVolumeError -- caller treats
    that as "excluded", never silently proceeding on a well that
    couldn't actually be tied or aligned."""
    tie = dts.resolve_direct_tie(volume, well_id)

    result = volume.get_spectral_decomposition_trace(
        tie.inline_number, tie.crossline_number, method="cwt", include_sswt=True
    )
    time_ms = np.array(result["time_ms"], dtype=float)

    curves_response = well_service.get_well_curves(well_id)
    rows = curves_response["data"]
    depth_all = _extract_curve(rows, "DEPT")
    dptm_all = _extract_curve(rows, "DPTM")
    targets_all = {k: _extract_curve(rows, v) for k, v in TARGET_LAS_NAMES.items()}

    valid = np.isfinite(depth_all) & np.isfinite(dptm_all)
    for arr in targets_all.values():
        valid &= np.isfinite(arr)
    if valid.sum() < MIN_VALID_SAMPLES:
        raise wst.TieError(
            f"Well '{well_id}' has too few samples with DEPT, DPTM, and all of VSH/PHIE/SWE "
            f"valid together ({int(valid.sum())}, need >= {MIN_VALID_SAMPLES})."
        )

    depth_v = depth_all[valid]
    t_shifted = dptm_all[valid] + tie.bulk_shift_ms
    targets_v = {k: v[valid] for k, v in targets_all.items()}

    in_range = (t_shifted >= time_ms[0]) & (t_shifted <= time_ms[-1])
    if in_range.sum() < MIN_VALID_SAMPLES:
        raise wst.TieError(
            f"Well '{well_id}''s logged interval doesn't overlap the seismic recording window "
            "after the direct tie's bulk shift."
        )
    depth_v = depth_v[in_range]
    t_shifted = t_shifted[in_range]
    targets_v = {k: v[in_range] for k, v in targets_v.items()}

    # Nearest seismic-time-sample index per depth (matching the reference
    # pipeline exactly, not this app's usual continuous interpolation),
    # then average away duplicate depths landing on the same sample --
    # routine given log sampling is far finer than the seismic's ~2-4ms.
    idxs = np.array([int(np.argmin(np.abs(time_ms - t))) for t in t_shifted])
    uniq_idx, inverse = np.unique(idxs, return_inverse=True)

    def _agg(arr: np.ndarray) -> np.ndarray:
        return np.array([arr[inverse == k].mean() for k in range(len(uniq_idx))])

    agg_depth = _agg(depth_v)
    agg_targets = {k: _agg(v) for k, v in targets_v.items()}

    features: dict[str, np.ndarray] = {}
    freq_hz: dict[str, np.ndarray] = {}
    for method, energy_key in METHOD_ENERGY_KEYS.items():
        arr = np.array(result[energy_key], dtype=float)  # (n_time, n_freq)
        features[method] = arr[uniq_idx, :]
        freq_hz[method] = np.array(result[METHOD_FREQ_KEYS[method]], dtype=float)

    return WellPredictionFeatures(
        well_id=well_id,
        inline_number=tie.inline_number,
        crossline_number=tie.crossline_number,
        correlation=tie.correlation,
        best_freq_hz=tie.best_freq_hz,
        polarity=tie.polarity,
        bulk_shift_ms=tie.bulk_shift_ms,
        depths_m=agg_depth,
        twt_ms=time_ms[uniq_idx],
        targets=agg_targets,
        features=features,
        freq_hz=freq_hz,
    )


def _eligible_well_features(volume) -> tuple[dict[str, WellPredictionFeatures], list[dict]]:
    features: dict[str, WellPredictionFeatures] = {}
    excluded: list[dict] = []
    for summary in well_service.list_well_summaries():
        try:
            features[summary.well_id] = _build_well_features(volume, summary.well_id)
        except (wst.TieError, well_service.WellNotFoundError) as exc:
            excluded.append({"well_id": summary.well_id, "reason": str(exc)})
    return features, excluded


def _fit_model(
    well_features: dict[str, WellPredictionFeatures], train_ids: list[str], method: str, target: str
):
    """Fit StandardScaler+Ridge on the given training wells' features --
    factored out of _predict_blind so the SAME fitted model can also be
    applied to a full inline's worth of traces (render_property_inline_
    maps_image) without duplicating the fit logic."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    X_train = np.concatenate([well_features[w].features[method] for w in train_ids], axis=0)
    y_train = np.concatenate([well_features[w].targets[target] for w in train_ids], axis=0)
    scaler = StandardScaler().fit(X_train)
    model = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(X_train), y_train)
    return scaler, model, int(len(y_train))


def _predict_blind(
    well_features: dict[str, WellPredictionFeatures], blind_well_id: str, method: str, target: str
) -> dict:
    from sklearn.metrics import r2_score

    train_ids = [w for w in well_features if w != blind_well_id]
    scaler, model, n_train_samples = _fit_model(well_features, train_ids, method, target)
    blind = well_features[blind_well_id]
    X_test, y_test = blind.features[method], blind.targets[target]
    y_pred = np.clip(model.predict(scaler.transform(X_test)), 0.0, 1.0)

    r2 = float(r2_score(y_test, y_pred)) if len(y_test) >= 2 else None
    return {
        "r2": r2,
        "n_train_samples": n_train_samples,
        "n_train_wells": len(train_ids),
        "n_test_samples": int(len(y_test)),
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "depths_m": blind.depths_m.tolist(),
        "twt_ms": blind.twt_ms.tolist(),
    }


def get_full_loocv_r2(target: str, method: str) -> dict:
    """TRUE leave-one-well-out: every well takes a turn as the held-out
    well (not just one fixed blind well), pooled into a single R^2 --
    matches the reference "improved_pipeline"'s pooled-LOOCV selection
    metric. Returns {'r2': float|None, 'n_wells': int, 'per_well':
    {well_id: r2_or_None}}."""
    from sklearn.metrics import r2_score

    from app.services import seismic_processor as sp

    volume = sp.get_segy_volume()
    well_features, _excluded = _eligible_well_features(volume)
    well_ids = list(well_features.keys())
    if len(well_ids) < 2:
        return {"r2": None, "n_wells": len(well_ids), "per_well": {}}

    all_true: list[float] = []
    all_pred: list[float] = []
    per_well: dict[str, float | None] = {}
    for held_out in well_ids:
        pred = _predict_blind(well_features, held_out, method, target)
        per_well[held_out] = pred["r2"]
        all_true.extend(pred["y_true"])
        all_pred.extend(pred["y_pred"])

    pooled_r2 = float(r2_score(all_true, all_pred)) if len(all_true) >= 2 else None
    return {"r2": pooled_r2, "n_wells": len(well_ids), "per_well": per_well}


def get_prediction_result(blind_well_id: str, target: str, method: str) -> dict:
    if target not in TARGET_LAS_NAMES:
        raise ValueError(f"target must be one of {list(TARGET_LAS_NAMES)}, got {target!r}.")
    if method not in METHOD_ENERGY_KEYS:
        raise ValueError(f"method must be one of {list(METHOD_ENERGY_KEYS)}, got {method!r}.")

    from app.services import seismic_processor as sp

    volume = sp.get_segy_volume()
    well_features, excluded = _eligible_well_features(volume)

    base = {
        "blind_well_id": blind_well_id,
        "target": target,
        "method": method,
        "excluded_wells": excluded,
    }

    empty_tie_fields = {
        "inline_number": None,
        "crossline_number": None,
        "tie_correlation": None,
        "tie_best_freq_hz": None,
        "tie_polarity": None,
        "tie_bulk_shift_ms": None,
    }
    if blind_well_id not in well_features:
        return {
            **base,
            **empty_tie_fields,
            "status": "insufficient_data",
            "message": f"'{blind_well_id}' does not currently have a usable direct tie -- see excluded_wells for why.",
            "n_train_wells": 0,
            "result": None,
        }
    if len(well_features) < 2:
        return {
            **base,
            **empty_tie_fields,
            "status": "insufficient_data",
            "message": (
                f"Only {len(well_features)} well(s) have a usable tie -- need at least 2 (the "
                "blind well plus at least one other to train on)."
            ),
            "n_train_wells": 0,
            "result": None,
        }

    pred = _predict_blind(well_features, blind_well_id, method, target)
    blind = well_features[blind_well_id]
    return {
        **base,
        "status": "validated",
        "message": None,
        "inline_number": blind.inline_number,
        "crossline_number": blind.crossline_number,
        "tie_correlation": blind.correlation,
        "tie_best_freq_hz": blind.best_freq_hz,
        "tie_polarity": blind.polarity,
        "tie_bulk_shift_ms": blind.bulk_shift_ms,
        "n_train_wells": pred["n_train_wells"],
        "result": pred,
    }


def render_full_loocv_heatmap_image() -> bytes:
    """PNG heatmap: TRUE pooled leave-one-well-out R^2 (every well takes a
    turn as the held-out well, see get_full_loocv_r2) for all 3 targets x
    2 methods -- the reference "improved_pipeline"'s own honest selection
    metric, not a single fixed blind well's score. Independent of which
    well is currently selected as "blind" elsewhere on the page."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from app.services import seismic_processor as sp

    volume = sp.get_segy_volume()
    well_features, _excluded = _eligible_well_features(volume)

    targets = list(TARGET_LAS_NAMES.keys())
    methods = list(METHOD_ENERGY_KEYS.keys())
    grid = np.full((len(targets), len(methods)), np.nan)
    n_wells = len(well_features)

    if n_wells >= 2:
        for i, target in enumerate(targets):
            for j, method in enumerate(methods):
                loocv = get_full_loocv_r2(target, method)
                if loocv["r2"] is not None:
                    grid[i, j] = loocv["r2"]

    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=140)
    finite = grid[np.isfinite(grid)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    im = ax.imshow(grid, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(methods)), [m.upper() for m in methods])
    ax.set_yticks(range(len(targets)), [t.upper() for t in targets])
    for i in range(len(targets)):
        for j in range(len(methods)):
            val = grid[i, j]
            text = f"{val:.3f}" if np.isfinite(val) else "n/a"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=10)
    ax.set_title(f"Full LOOCV pooled R² (n={n_wells} wells)")
    fig.colorbar(im, ax=ax, label="R² (can be negative -- worse than the mean)", fraction=0.046)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_frequency_map_image(blind_well_id: str) -> bytes:
    """PNG: amplitude-spectrum frequency map (frequency x crossline,
    averaged over time) for the inline through the given well -- a plain
    FFT (much cheaper than CWT/SSWT), giving a spatial view of which
    frequencies carry energy along that inline, e.g. for spotting a
    low-frequency shadow. Reference pipeline step 2 (amplitude_spectrum_
    cube) computed this for the whole volume but never actually plotted
    it -- this renders it for one inline, which is what's actually useful
    to look at."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from app.services import seismic_processor as sp

    volume = sp.get_segy_volume()
    tie = dts.resolve_direct_tie(volume, blind_well_id)
    section = volume.get_inline_section(tie.inline_number)
    amplitude = np.array(section["amplitude"], dtype=float)  # (n_samples, n_traces)
    position_axis = np.array(section["crossline_axis"], dtype=int)

    dt_s = volume.sample_interval_ms / 1000.0
    spectrum = np.abs(np.fft.rfft(amplitude, axis=0))  # (n_freq, n_traces)
    freqs = np.fft.rfftfreq(amplitude.shape[0], d=dt_s)
    nyquist = freqs[-1]
    band = freqs <= min(nyquist, 100.0)  # seismic energy is rarely useful above ~100 Hz

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(freqs[band], spectrum[band].mean(axis=1), color="#2563EB")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Mean amplitude")
    axes[0].set_title(f"Average spectrum -- Inline {tie.inline_number}")
    axes[0].grid(alpha=0.3)

    im = axes[1].imshow(
        spectrum[band], aspect="auto", cmap="viridis",
        extent=[position_axis[0], position_axis[-1], freqs[band][-1], freqs[band][0]],
    )
    axes[1].axvline(tie.crossline_number, color="white", lw=1, ls="--", alpha=0.8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Crossline")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title(f"Frequency map -- through {blind_well_id}")
    fig.colorbar(im, ax=axes[1], label="Amplitude", fraction=0.046)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_property_inline_maps_image(blind_well_id: str, method: str) -> bytes:
    """PNG: for VSH, PHIE, and SWE, a real-vs-predicted pair along the
    blind well's own inline -- LEFT is the actual (unmodified) seismic
    amplitude section, RIGHT is the SAME inline with the blind-well-
    trained model's predicted property value painted across EVERY trace
    along it (not just the well's own location), each property in its
    own colormap. A real property value doesn't exist away from a well,
    so "real" here means the actual seismic data, not a true property
    map -- there is nothing to compare the right panel against except
    the well's own logged value at its one location.

    Computes each trace's CWT/SSWT ONCE and reuses it for all 3 targets'
    predictions (the per-trace spectral decomposition, not the
    prediction itself, is the expensive part -- one call per crossline
    along the inline).

    EXPLORATORY, not a validated spatial map: this pipeline's blind R^2
    has been at or below zero for most (target, method) combinations on
    this app's real wells (see the LOOCV heatmap) -- a confidently
    colored map does not mean the colors are trustworthy. That caveat is
    rendered directly into the image, not just documented here.
    """
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from app.services import seismic_processor as sp

    if method not in METHOD_ENERGY_KEYS:
        raise ValueError(f"method must be one of {list(METHOD_ENERGY_KEYS)}, got {method!r}.")

    volume = sp.get_segy_volume()
    well_features, _excluded = _eligible_well_features(volume)
    if blind_well_id not in well_features:
        raise ValueError(f"'{blind_well_id}' does not currently have a usable direct tie.")
    if len(well_features) < 2:
        raise ValueError(f"Only {len(well_features)} well(s) have a usable tie -- need at least 2.")

    targets = list(TARGET_LAS_NAMES.keys())
    train_ids = [w for w in well_features if w != blind_well_id]
    fitted = {t: _fit_model(well_features, train_ids, method, t) for t in targets}
    blind = well_features[blind_well_id]

    section = volume.get_inline_section(blind.inline_number)
    amplitude = np.array(section["amplitude"], dtype=float)  # (n_samples, n_traces)
    position_axis = np.array(section["crossline_axis"], dtype=int)
    twt_axis_ms = np.array(section["twt_axis_ms"], dtype=float)
    vmax = float(np.percentile(np.abs(amplitude), 98)) or 1e-6
    energy_key = METHOD_ENERGY_KEYS[method]

    n_time = len(twt_axis_ms)
    pred_grids = {t: np.full((n_time, len(position_axis)), np.nan) for t in targets}
    for j, xl in enumerate(position_axis):
        try:
            spec = volume.get_spectral_decomposition_trace(
                blind.inline_number, int(xl), method="cwt", include_sswt=(method == "sswt")
            )
        except Exception:  # noqa: BLE001 -- a gap trace shouldn't abort the whole map
            continue
        X_line = np.array(spec[energy_key], dtype=float)  # (n_time, n_freq)
        for t in targets:
            scaler, model, _n = fitted[t]
            pred_grids[t][:, j] = np.clip(model.predict(scaler.transform(X_line)), 0.0, 1.0)

    cmaps = {"vsh": "YlOrBr", "phie": "Greens", "swe": "Blues"}
    extent = [position_axis[0], position_axis[-1], twt_axis_ms[-1], twt_axis_ms[0]]

    fig, axes = plt.subplots(3, 2, figsize=(12, 14), sharex=True, sharey=True)
    for row, t in enumerate(targets):
        ax_real, ax_pred = axes[row]
        ax_real.imshow(amplitude, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax, extent=extent)
        ax_real.set_title(f"Real seismic -- Inline {blind.inline_number}" if row == 0 else "")
        ax_real.set_ylabel("Two-Way Time (ms)")

        im = ax_pred.imshow(pred_grids[t], aspect="auto", cmap=cmaps[t], vmin=0.0, vmax=1.0, extent=extent)
        ax_pred.set_title(f"Predicted {t.upper()} ({method.upper()}, blind: {blind_well_id})")
        fig.colorbar(im, ax=ax_pred, label=t.upper(), fraction=0.04)

        for ax in (ax_real, ax_pred):
            ax.axvline(blind.crossline_number, color="black", lw=1, ls="--", alpha=0.7)
        ax_real.text(
            blind.crossline_number, twt_axis_ms[0], f" {blind_well_id}",
            fontsize=8, va="bottom", color="black",
        )

    for ax in axes[-1]:
        ax.set_xlabel("Crossline")

    fig.suptitle(
        f"EXPLORATORY -- blind well: {blind_well_id}. This pipeline's blind R² has been near "
        "zero or negative for most target/method combinations on real wells (see LOOCV heatmap): "
        "colors show the model's estimate, not a validated spatial result.",
        fontsize=10, color="firebrick", y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_prediction_image(blind_well_id: str, target: str, method: str) -> bytes:
    """PNG bytes: side-by-side inline-section images (TRUE vs. PREDICTED
    target, painted as a colored strip at the blind well's crossline
    position) -- direct port of the reference pipeline's
    plot_inline_true_vs_pred, using this module's tie/features instead."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from app.services import seismic_processor as sp

    data = get_prediction_result(blind_well_id, target, method)
    if data["status"] != "validated":
        raise ValueError(data["message"])

    volume = sp.get_segy_volume()
    section = volume.get_inline_section(data["inline_number"])
    amplitude = np.array(section["amplitude"], dtype=float)  # (n_samples, n_traces)
    position_axis = np.array(section["crossline_axis"], dtype=float)
    twt_axis_ms = np.array(section["twt_axis_ms"], dtype=float)
    vmax = float(np.percentile(np.abs(amplitude), 98)) or 1e-6

    result = data["result"]
    true_vals = np.array(result["y_true"], dtype=float)
    pred_vals = np.array(result["y_pred"], dtype=float)
    twt_vals = np.array(result["twt_ms"], dtype=float)
    xl0 = float(data["crossline_number"])

    norm = Normalize(vmin=0.0, vmax=max(float(true_vals.max()), float(pred_vals.max()), 1e-6))
    cmap = plt.cm.YlOrBr
    xl_step = float(position_axis[1] - position_axis[0]) if len(position_axis) > 1 else 1.0
    strip_half_traces = 3

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    titles = [f"TRUE {target.upper()}", f"PREDICTED {target.upper()} (blind: {blind_well_id})"]
    for ax, title, vals in zip(axes, titles, [true_vals, pred_vals]):
        ax.imshow(
            amplitude, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax,
            extent=[position_axis[0], position_axis[-1], twt_axis_ms[-1], twt_axis_ms[0]],
        )
        for t, v in zip(twt_vals, vals):
            ax.add_patch(plt.Rectangle(
                (xl0 - strip_half_traces * xl_step, t - 4),
                2 * strip_half_traces * xl_step, 8,
                color=cmap(norm(v)),
            ))
        ax.axvline(xl0, color="black", lw=1, ls="--", alpha=0.6)
        ax.set_xlim(xl0 - 40 * xl_step, xl0 + 40 * xl_step)
        ax.set_ylim(float(twt_vals.max()) + 20, float(twt_vals.min()) - 20)
        ax.set_xlabel("Crossline")
        ax.set_title(title)
    axes[0].set_ylabel("Two-Way Time (ms)")
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes, fraction=0.03, label=target.upper())

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
