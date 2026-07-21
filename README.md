# Multi-Well Petrophysical Interpretation Platform (RawReservoirClassifier)

A full-stack application that reads raw LAS well logs (DEPT, GR, RESISTIVITY, RHOB, NPHI, DT),
computes standard petrophysical interpretation curves (VSH, PHIT, PHIE, SWE, PERM_TIXIER,
CORE_PERM_PRED, VVOLC, ZONES, ...), parses raw SEG-Y seismic data and derives seismic
attributes (including heuristic VSH/PHIE/SWE seismic proxies), exposes an Anthropic
Claude-powered petrophysics assistant, and presents everything through a light-themed
dashboard with cross-well views, single-well log tracks/crossplots, and a dedicated seismic
module.

> **Status:** Code-complete per `AGENT_BRIEF.md`. Dependencies have **not** been installed or
> run locally in this environment (no network access for `pip`/`npm`). Install and validate
> on a machine/CI with network access using the steps below before relying on this in
> production.

---

## 1. Project layout

```
backend/
  app/
    config/petrophysics_config.yaml   <- every tunable petrophysics constant lives here
    config/seismic_config.yaml        <- every tunable seismic-attribute constant lives here
    config_loader.py                  <- merges field/dataset defaults + per-well/per-dataset overrides
    las_loader.py                     <- raw LAS -> validated DataFrame + metadata
    segy_loader.py                    <- raw SEG-Y -> validated amplitude matrix + metadata
    petrophysics.py                   <- every log-derived calculation in section 3, one function each
    seismic_attributes.py             <- seismic attribute + heuristic VSH/PHIE/SWE proxy calculations
    repository.py                     <- well storage layer (Parquet/JSON today, swappable for Postgres)
    seismic_repository.py             <- seismic storage layer (.npz + Parquet/JSON, same pattern)
    models/schemas.py                 <- Pydantic request/response models
    routers/
      wells.py                        <- upload/list/curves/zones/crossplot/export
      seismic.py                      <- SEG-Y upload/list/section/attributes/export
      dashboard.py                    <- field-wide summary (wells + seismic)
      chat.py                         <- SSE-streaming Anthropic agent endpoint
    services/
      well_service.py                 <- application service layer for wells (routers call this)
      seismic_service.py              <- application service layer for seismic (routers call this)
      anthropic_agent.py              <- Claude Messages API + tool-calling agent loop
  scripts/
    bulk_load_wells.py                <- CLI to process backend/data/raw/*.las
    bulk_load_seismic.py              <- CLI to process backend/data/seismic_raw/*.sgy
    train_core_perm_model.py          <- trains the CORE_PERM_PRED proxy model
  data/
    raw/                              <- put Z-02.las ... Z-08.las here
    processed/                        <- Parquet + JSON cache (gitignored, regenerable)
    seismic_raw/                      <- put raw .sgy/.segy files here
    seismic_processed/                <- .npz + Parquet + JSON cache (gitignored, regenerable)
    models/                           <- trained sklearn models (gitignored, regenerable)
  tests/
    conftest.py                       <- synthetic 3-zone well fixture
    test_petrophysics.py              <- unit tests for every well formula in section 3
    test_seismic_attributes.py        <- unit tests for every seismic attribute/proxy calculation
  main.py
  requirements.txt
  .env.example
frontend/
  src/
    api/                              <- client.ts (REST + SSE), types.ts
    components/                       <- LogTrackViewer, CrossplotBuilder, ChatPanel, ...
    pages/                            <- DashboardPage, WellDetailPage
    store/                            <- Zustand UI state (chat open/closed, conversations)
    styles/                           <- light-theme tokens (tokens.ts mirrors tailwind.config.js)
  package.json
  .env.example
AGENT_BRIEF.md                        <- original spec this project was built from
```

---

## 2. Getting started

### Backend

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY

uvicorn main:app --reload --port 8000
```

Run the unit test suite (this exercises every formula in `petrophysics.py` against a
synthetic 3-zone well -- no real LAS files required):

```bash
pytest tests/ -v
```

Load real well data (place `Z-02.las` ... `Z-08.las` in `backend/data/raw/` first):

```bash
python scripts/bulk_load_wells.py
```

Load real seismic data (place `.sgy`/`.segy` files in `backend/data/seismic_raw/` first):

```bash
python scripts/bulk_load_seismic.py
```

Train the `CORE_PERM_PRED` proxy model (optional -- only needed once wells are loaded; see
section 5 below for why this is a proxy, not a real core-calibrated model):

```bash
python scripts/train_core_perm_model.py
python scripts/bulk_load_wells.py   # reprocess so CORE_PERM_PRED gets included
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env   # optional, only needed if not using the Vite dev proxy
npm run dev
```

The Vite dev server proxies `/wells`, `/dashboard`, and `/chat` to
`http://127.0.0.1:8000` (see `frontend/vite.config.ts`), so no `.env` is required for local
development as long as the backend is running on port 8000.

Open http://localhost:5173.

---

## 3. Notes on package versions

`requirements.txt` uses `>=` version floors rather than exact pins, targeting Python 3.11+
(tested version resolution assumed Python 3.13 compatibility, since `numpy==1.26.4` has no
prebuilt wheel for 3.13 and requires a C compiler toolchain to build from source). If you
need fully reproducible installs, pin exact versions with `pip freeze > requirements.lock.txt`
after your first successful install.

---

## 4. Configuring petrophysical assumptions (for the SME / petrophysicist)

**Every constant used by the calculation engine lives in one file:**
`backend/app/config/petrophysics_config.yaml`. Nothing is hardcoded in `petrophysics.py`.
The file has a `defaults` block (field-wide) and an optional `wells.<WELL_ID>` block per
well that overrides any default. You do not need to touch Python code to retune the
interpretation -- edit the YAML and restart the backend (or call the upload/reprocess
endpoints again).

| Constant | Default | Where | Why it matters |
|---|---|---|---|
| `vsh.method` | `older` (Larionov, consolidated rocks) | `defaults.vsh.method` | Use `tertiary` for young/unconsolidated Tertiary sands -- gives a lower VSH for the same IGR. |
| `vsh.gr_clean_percentile` / `gr_shale_percentile` | 5th / 95th percentile of the well's own GR | `defaults.vsh` | Auto-derived per well by default so you don't need a field-wide GR baseline; override with `use_percentiles: false` + explicit `gr_clean_override`/`gr_shale_override` if you have known clean-sand/shale GR values. |
| `phit.rhob_matrix` | 2.65 g/cc (sandstone) | `defaults.phit` | Use 2.71 for limestone, 2.87 for dolomite, or a custom value for a mixed lithology. |
| `phit.rhob_fluid` | 1.0 g/cc (fresh mud filtrate) | `defaults.phit` | Adjust for saline/oil-based mud filtrate density if known. |
| `swe.rw` | **0.05 ohm.m -- PLACEHOLDER, must be reviewed** | `defaults.swe.rw` | Formation water resistivity at formation temperature. This is the single most field-specific, non-derivable input in the whole model -- get it from produced water analysis, SP log, or a nearby water-bearing zone at known temperature. Wrong Rw directly biases every SWE and PERM_TIXIER value. |
| `swe.a`, `swe.m`, `swe.n` | 1, 2, 2 (clean-sand Archie) | `defaults.swe` | Adjust `m` (cementation exponent) and `n` (saturation exponent) if core data or literature suggests different values for this formation's pore geometry/wettability. |
| `perm_tixier.swirr` | `null` (auto-estimated from cleanest interval) or `swirr_default: 0.25` | `defaults.perm_tixier` | Set explicitly if you have a known irreducible water saturation for this reservoir (typically 0.15-0.35 depending on rock quality). |
| `vvolc.*` thresholds | RHOB>2.7, NPHI<0.15, GR<0.6×GR_shale | `defaults.vvolc` | Purely heuristic (no PEF curve available) -- recalibrate against cuttings/core descriptions before trusting this for lithology decisions. |
| `zones.vsh_max` | 0.4 | `defaults.zones` | Reservoir cutoff. Raise for shalier reservoirs, lower for a stricter clean-sand definition. |
| `zones.phie_min` | 0.08 (8%) | `defaults.zones` | Minimum effective porosity to count as reservoir quality rock. |
| `zones.swe_reservoir_max` | 0.65 | `defaults.zones` | Water saturation ceiling for "reservoir" classification. |
| `zones.swe_pay_max` | 0.5 | `defaults.zones` | Stricter Sw ceiling that additionally qualifies a reservoir interval as "Pay". |
| `dptm.enabled` | `true` | `defaults.dptm` | Sonic-integration time-depth is an **approximation only**; disable and substitute a real checkshot/VSP time-depth table when available. |
| `core_perm_pred.*` | RandomForest, 200 trees, depth 8 | `defaults.core_perm_pred` | Hyperparameters for the CORE_PERM_PRED proxy regression (see caveat below). |

**Per-well overrides example** (e.g. Z-05 sits in a limestone section with a different Rw):

```yaml
wells:
  Z-05:
    phit:
      rhob_matrix: 2.71
    swe:
      rw: 0.08
```

---

## 5. Explicit assumptions & proxy calculations (read before trusting results)

Because only five raw curves are available (no PEF, no core plugs, no checkshot/VSP, no
deviation survey), several outputs are **documented approximations**, not direct
measurements. Each is called out in its docstring in `petrophysics.py` and flagged by the
chat assistant's system prompt whenever it's discussed:

- **MD/TVD** (`compute_md_tvd`): assumes the well is vertical (`TVD = MD = DEPT`). A real
  deviation survey (inclination/azimuth vs depth) can be passed in to compute true TVD via
  the minimum curvature method -- that code path is stubbed with `NotImplementedError` and
  documented as an extension point, since no survey was available for Z-02..Z-08.
- **DPTM** (`compute_dptm`): approximates two-way time by integrating the sonic (DT) log.
  This accumulates sonic logging error over depth and ignores velocity anisotropy. Replace
  with a real checkshot/VSP time-depth table as soon as one is available.
- **VVOLC** (`compute_vvolc`): a density-neutron crossplot *heuristic* flag, not a true
  mineralogical decomposition (would normally require a PEF curve). Needs calibration
  against cuttings/core descriptions before being used for anything beyond a first-pass
  lithology screen.
- **CORE_PERM_PRED** (`train_core_perm_model` / `predict_core_perm`): trained as a proxy
  regression against `PERM_TIXIER` itself (no real core plug measurements exist for this
  field yet). It currently just learns a smoothed generalization of the Tixier estimate.
  As soon as real core permeability measurements are available, retrain via
  `train_core_perm_model(training_df, config, target_col="CORE_PERM_MEASURED")` (or your
  actual core data column name) to get a genuine core-calibrated predictor -- the function
  signature was designed for this drop-in swap.
- **Rw and Swirr**: never hardcoded, but the shipped defaults (`Rw = 0.05 ohm.m`,
  `Swirr = 0.25`) are generic placeholders, not measured for this specific field. Set real
  values in `petrophysics_config.yaml` before using SWE/PERM_TIXIER for anything beyond
  exploratory QC.

### Seismic module caveats (read before trusting seismic proxy results)

The seismic module (`backend/app/seismic_attributes.py`, config in
`backend/app/config/seismic_config.yaml`) computes standard signal-processing attributes
(RMS amplitude, instantaneous envelope via Hilbert transform, dominant frequency) directly
from raw SEG-Y trace amplitudes -- these are well-defined and not heuristic.

However, the **VSH_SEISMIC_PROXY, PHIE_SEISMIC_PROXY, and SWE_SEISMIC_PROXY** attributes are
an entirely different category: they are simple, **uncalibrated, amplitude-based heuristics**,
not measured rock properties. Properly deriving shale volume, porosity, or water saturation
from seismic data requires inversion to acoustic/elastic impedance calibrated against a real
well tie, plus formation-specific rock-physics relationships -- none of which raw post-stack
amplitude alone can provide. Specifically:

- **VSH_SEISMIC_PROXY** = normalized average envelope amplitude. Rationale: stronger
  reflectivity often marks lithology/bedding contrasts, but this does not distinguish shale
  from any other lithology contrast.
- **PHIE_SEISMIC_PROXY** = `1 - normalized RMS amplitude`. Rationale: in clean elastic sands,
  impedance tends to anti-correlate with porosity, and RMS amplitude is used here only as a
  rough, uninverted stand-in for relative impedance trends.
- **SWE_SEISMIC_PROXY** = a "bright spot" heuristic -- traces with envelope amplitude above
  the 90th percentile (configurable) are treated as candidate hydrocarbon indicators. This
  produces many false positives (tuning effects, lithology contrasts, multiples all also
  produce bright amplitudes) and is a first-pass screening aid only.

These proxies exist so lateral trends can be eyeballed on the dashboard away from well
control. They are flagged everywhere they appear (API responses, frontend UI, and the chat
assistant's system prompt) and must be calibrated against a real seismic-to-well tie before
being used for any interpretation decision. All thresholds/percentiles live in
`backend/app/config/seismic_config.yaml`, with the same field-wide-defaults +
per-dataset-override pattern as `petrophysics_config.yaml`.

### Well-to-seismic tie

`backend/app/well_seismic_tie.py` implements a real geophysical tie (not the amplitude
heuristic above): it integrates the sonic (DT) log into a depth-time relationship, derives
acoustic impedance from DT + RHOB, builds a reflectivity series, convolves it with a Ricker
wavelet to produce a synthetic seismogram, then cross-correlates that synthetic against a real
seismic trace to find the best-fit time shift and correlation coefficient. `GET
/tie/{well_id}?seismic_dataset_id=...` runs this and the frontend plots the result (real trace
vs. shifted synthetic) in the Seismic page's "Well-to-seismic tie" panel.

**Picking which seismic trace to tie against:**

- **Coordinate-based (preferred):** each LAS file's `~Well` section can carry surface
  coordinates (`XWELL`/`YWELL`, or the aliases `XCOORD`/`YCOORD`, `SURX`/`SURY`, `X`/`Y` --
  see `las_loader.py`), and each SEG-Y file's trace headers can carry per-trace coordinates
  (`CDP_X`/`CDP_Y`, falling back to `SourceX`/`SourceY`, with the SEG-Y coordinate scalar
  applied -- see `segy_loader.py`). When both are present, `tie_service.py` calls
  `find_nearest_trace_index()` to do a real Euclidean nearest-trace search and reports the
  actual `distance_m`. The response's `tie_method` field is `"nearest_trace"` in this case.
  Well and seismic coordinates are assumed to share the same CRS/units (e.g. both UTM metres)
  -- this is not verified, so a mismatched CRS would silently return a wrong-but-plausible
  trace; sanity-check `distance_m` against the survey's known extent.
- **Manual fallback:** if either side is missing coordinates (older LAS files without a
  location header, or a SEG-Y export with blank/non-standard geometry bytes), the tie falls
  back to a manually configured `trace_index` per well in
  `backend/app/config/tie_config.yaml` (`well_coordinate_overrides`). `tie_method` is
  `"manual_override"` and the response carries a `geometry_warning` explaining that this is
  not a spatial match, with `distance_m` left `null`. `max_tie_search_radius_m` in the same
  config file caps how far the coordinate-based search is allowed to look before raising an
  error instead of silently tying to a distant trace.

Z-02 through Z-08's raw LAS files ship with `XWELL`/`YWELL` coordinates, so once a SEG-Y
dataset with real trace coordinates is uploaded, the tie automatically uses the
coordinate-based path with no config changes needed -- see the CRS caveat below for what
these coordinates currently are and aren't.

### Seismic Visualization (direct SEG-Y interpretation)

`backend/app/services/seismic_processor.py` reads a raw SEG-Y volume from
`backend/data/seismic_raw/` directly (via `segyio`, memory-mapped and fully loaded into memory
once behind a process-wide singleton -- see `get_segy_volume()`) rather than going through the
upload/attribute pipeline above, because it needs inline/crossline geometry that pipeline never
stores. It serves five interpretation displays, exposed at `/api/seismic/*` and rendered in the
frontend's **Seismic Visualization** panel (bottom of the `/seismic` page):

- **Inline / crossline sections** -- amplitude vs. position x two-way-time, either direction.
- **Time slices** -- a map-view amplitude cut across the full inline x crossline grid at a
  chosen two-way time (nearest-sample lookup; a requested time outside the survey's range
  clamps to the nearest edge sample).
- **Well tie** -- reuses `well_seismic_tie.py`'s synthetic-seismogram pipeline (impedance,
  reflectivity, Ricker wavelet convolution) and `well_service.py`'s LAS loading, but ties
  against this volume's own traces via a real nearest-trace coordinate search rather than the
  upload pipeline's dataset. The Ricker wavelet's dominant frequency is adjustable per request
  (`wavelet_freq_hz`, default 25 Hz). Returns a `note` field flagging that the depth-time
  conversion is sonic-integration-only (no checkshot/VSP), and a clear 422 naming the missing
  curve if a well has no usable DT or RHOB log.
- **Amplitude spectrum** -- average FFT magnitude spectrum (whole volume via a systematic
  trace sample, or one inline), plus dominant frequency, -3dB bandwidth, and an uncalibrated
  S/N proxy. A single flat spectrum per trace/section -- for how frequency content *varies
  along the time axis* (tuning effects, thin beds), see spectral decomposition below.
- **Spectral decomposition** -- frequency content as a function of time along each trace, not
  a single flat spectrum, so thin-bed tuning and stratigraphic features (channels, thin
  reservoir layers) that don't show up in a plain amplitude section or flat spectrum become
  visible as brightening at specific frequencies at specific times. Two methods, selectable per
  request:
  - **STFT** (`method=stft`, default) -- `scipy.signal.stft` with a short window
    (`STFT_WINDOW_SAMPLES` = 32 samples, ~64 ms at 2 ms sampling) and 75% overlap. This is a
    deliberate tradeoff for this survey's short (~624 ms) traces: a short window gives useful
    *time* resolution to localize tuning along the trace, at the cost of coarse *frequency*
    resolution (~15.6 Hz/bin at 2 ms sampling) -- a longer window would sharpen frequency
    resolution but blur exactly *when* a given frequency's energy occurs.
  - **CWT** (`method=cwt`) -- Continuous Wavelet Transform with a complex Morlet wavelet.
    `scipy.signal.cwt`/`morlet2` were removed in recent scipy releases (not available in this
    project's scipy 1.17+), so the Morlet wavelet is hand-rolled in `seismic_processor.py`
    (`_morlet_wavelet`) the same way `well_seismic_tie.ricker_wavelet()` already hand-rolls its
    wavelet for the same reason. Runs at the trace's native sample resolution (no windowing
    time-resolution loss, unlike STFT), evaluated at a fixed default frequency grid (5-100 Hz
    in 5 Hz steps, `CWT_DEFAULT_FREQS_HZ`).
  - Both methods respect the Nyquist limit (never return frequencies above `fs/2`) and flag the
    typically-useful seismic band (5-80 Hz, `TYPICAL_USEFUL_BAND_HZ`) in the response so the
    frontend can default-frame around it rather than the full 0-Nyquist range.
  - `GET /api/seismic/spectral-decomp/inline/{n}` without `frequency_hz` returns the full
    (time x freq x position) volume for an inline -- heavier, meant for an initial load or
    export. The full per-(inline, method) decomposition is cached in memory
    (`SegyVolume._spectral_cache`) the first time it's computed, so repeated single-frequency
    slice requests (`frequency_hz=...`, what the frontend's frequency slider calls on every
    drag) index into the cached array instead of recomputing the whole STFT/CWT each time.
  - `GET /api/seismic/spectral-decomp/trace` returns the same shape for a single trace
    (inline+crossline), for a trace-inspection or well-tie-adjacent view.

**Non-standard trace header layout:** this SEG-Y file stores inline number at trace header
bytes 9-12 and crossline number at bytes 13-16 -- NOT the bytes segyio's usual
`INLINE_3D`/`CROSSLINE_3D` constants point at (189/193), which would silently read the wrong
values for this file. `SegyVolume` reads the correct bytes explicitly and cross-checks them
against a raw `struct.unpack` of trace 0's header at open time, so a segyio behavior change
would fail loudly rather than silently mis-read the geometry. SourceX/SourceY and the sample
interval/delay recording time *are* the standard SEG-Y rev1 locations, so segyio's normal
parsing is used for those.

**Coordinate reference system (CRS) check on well ties:** a well tie only proceeds if the
well's LAS coordinates fall within (a generous buffer around) the seismic survey's own
coordinate extent. If they don't, `get_well_tie` raises a clear error explaining that this is
almost always a CRS mismatch (e.g. two different projections/grids, not just a units issue)
rather than silently picking a distant, meaningless "nearest" trace.

**Current state of Z-02..Z-08's coordinates (resolved):** the field's well database
(GeoGraphix export) provides real surface `X`/`Y`/`KB`/`TD` for each well, shipped directly in
`backend/data/raw/Z-0X_raw.las`'s `~Well` section. Those fields are labeled `.m` in the LAS
header but are actually in **feet** -- confirmed by inspection (see "Unit standardization"
below) and by the fact that `X`/`Y` converted via ×0.3048 land squarely inside
`origional.segy`'s real survey extent (X ~363k-371k, Y ~2.95M-2.96M) for all 7 wells. This is
not a CRS/projection mismatch; it's a unit-labeling error in the source export.
`las_loader.py`'s `_standardize_well_header()` detects and corrects this automatically per
well (see below), so the well-tie features now work against real, correctly-converted
coordinates -- no more placeholder/rescaled values.

### Synthetic Seismogram module

A third well-tie surface, at `/api/synthetic/*` and the frontend's **Synthetic Seismogram**
page (`/synthetic`). Reuses the same computational core as "Well-to-seismic tie" and "Seismic
Visualization" above (`well_seismic_tie.py`, `seismic_processor.SegyVolume`) rather than
duplicating it, and adds the pieces those two don't have: unit-standardization QC reporting,
selectable density estimation, selectable/inspectable wavelets, a washout QC proxy, and
persisted manual stretch/squeeze.

**Unit standardization (`las_loader._standardize_well_header`):** `X`, `Y`, `KB` (Kelly
Bushing elevation), and `TD` (total depth) in the raw LAS files are labeled `.m` but are
actually in **feet** (see above). This is detected per well, not hardcoded: the `TD/STOP`
ratio is computed (`STOP` -- the curve data's own stop depth -- is always genuinely in
meters), and a ratio in the range ~2.8-4.2 (bracketing the feet-to-meters factor 3.28084 with
margin for TD normally sitting a bit deeper than the logged interval's STOP) means all four
fields are feet and get converted via ×0.3048; otherwise they're left as-is. `WellSummary`
exposes `coordinate_unit_detected` ("feet"/"meters"/`null` if unvalidated),
`unit_conversion_applied`, and `td_stop_ratio` so this is visible, not silent.

**Density estimation** (`well_seismic_tie.py`), selectable per request:
- `rhob` (default) -- the well's real RHOB curve.
- `gardner` -- Gardner's equation (`rho = a * V^b`), with coefficients **locally calibrated**
  against the well's own real RHOB via `scipy.optimize.curve_fit` when at least 20 valid
  samples exist, falling back to generic textbook constants (`a=0.31, b=0.25`) otherwise --
  the response's `gardner_coefficients.calibrated` flag says which happened.
- `rock_physics` -- an alternative estimate from the well's own VSH/PHIE outputs (matrix/shale/
  fluid density mixing model), for comparison against Gardner or real RHOB.

**Wavelet**, selectable per request:
- `statistical` (default) -- extracted from the real trace nearest the well: the trace's own
  average amplitude spectrum, assumed zero-phase, inverse-transformed and windowed.
- `ricker` -- the standard zero-phase Ricker generator (adjustable dominant frequency), same as
  the other two tie features use.

Both the wavelet's time-domain amplitude and its amplitude/phase spectra are returned, so a
mis-behaved phase (e.g. from a noisy statistical extraction) is visible before trusting the
resulting synthetic.

**Washout / hole-quality QC proxy** (`well_seismic_tie.washout_qc_flag`): no CALI curve exists
for these wells, so depth samples are flagged as "possible washout / unreliable interval" if
either (a) density porosity and NPHI disagree by more than a threshold (an enlarged/washed-out
hole reads erratically on both tools), or (b) DT deviates from a local rolling median by more
than a z-score threshold (washouts often cause sonic cycle-skipping). This is a soft heuristic,
explicitly labeled as such -- not a real caliper substitute.

**Depth-time anchoring (important limitation, fixed post-launch):** sonic integration
(`depth_to_twt`) only measures travel time *within the logged interval* -- it has no way to
know the two-way time from the surface down to the top of the log (that's exactly what a
checkshot provides, and none exists here), so on its own the curve always starts at 0 ms. A
real seismic survey's recorded time axis almost never starts at 0 ms (`origional.segy` starts
at 2030 ms). Early in this module's development, the synthetic was resampled onto the real
seismic axis with **no anchoring**, so it had zero time overlap and silently came out all-zero
(`correlation: 0.0`) whenever tied against a survey with a non-zero recording delay -- exactly
the real production case, though it went unnoticed because the existing test for this only
checked `isfinite(...)`, which is trivially true for zero. `depth_to_twt`/`build_synthetic`
now accept a `t0_ms` anchor and default it to the seismic volume's own first sample time
(`seismic_twt_axis_ms[0]`) when not given -- an arbitrary but non-degenerate starting point,
refined from there via manual stretch/squeeze. This fix applies to `well_seismic_tie.
build_synthetic()` itself, so it also fixes the same latent issue in "Well-to-seismic tie" and
"Seismic Visualization"'s well-tie endpoints above, not just this module.

**Manual stretch/squeeze:** since no real checkshot exists, a user can nudge the time-depth
curve with MD → time-shift control points (piecewise-linear interpolation between points,
held constant beyond the outermost ones -- `well_seismic_tie.apply_stretch_squeeze`).
Persisted per well in `backend/data/synthetic_processed/{well_id}.tie.json`
(`synthetic_tie_repository.py`, gitignored -- user-adjustable state, not source data) via
`PUT`/`GET`/`DELETE /api/synthetic/{well_id}/tie`, so adjustments survive across sessions
instead of recomputing from scratch. The frontend exposes this as an editable control-point
table (add/edit/remove MD+shift pairs, apply & save, clear) rather than true drag-on-chart
interaction -- functionally equivalent without the custom drag-handling a chart-based picker
would need.

**Vertical assumption:** no deviation survey exists in any of Z-02..Z-08's LAS files, so
MD = TVD for all of them (same placeholder as `petrophysics.compute_md_tvd`) -- surfaced as a
static badge, not buried.

### Dashboard combined upload

`POST /dashboard/upload` (`backend/app/services/dashboard_upload_service.py`) uploads a well
and its seismic together and auto-processes both, so the Wells/Dashboard, Seismic, and
Synthetic Seismogram pages all pick up the new data without a manual per-page re-upload or
re-selection. It does not implement or alter any tie math -- it's an orchestration layer
around the existing, unmodified `tie_service.get_well_seismic_tie`,
`synthetic_seismogram_service.generate`, and `seismic_processor.SegyVolume` calls the rest of
the app already makes.

**Two SEG-Y storage systems, fed together:** the app has always had two independent SEG-Y
paths -- the upload pipeline (`seismic_service`, a named `dataset_id` per dataset, feeding
the attribute cards and `tie_service`) and the single active volume read directly from
`backend/data/seismic_raw/` (`seismic_processor.get_segy_volume()`, feeding Seismic
Visualization and the whole Synthetic Seismogram page). A dashboard upload writes the SEG-Y
into both: `seismic_service.process_and_store_segy_bytes` first (so a corrupt file is
rejected before anything else touches it), then into `seismic_raw/` (pruning any other
`.sgy`/`.segy` there first -- this app is still single-active-volume, single-tenant by
design) followed by `get_segy_volume(refresh=True)`.

**Background job, not a queue:** the well (small file, fast) is parsed synchronously so the
response can return a `well_id` immediately; the SEG-Y upload, tie, synthetic seismogram, and
a spectral summary (via `SegyVolume.get_amplitude_spectrum` at the tied inline) run as a
single `fastapi.BackgroundTasks` job. Poll `GET /dashboard/upload/{well_id}/status` for
progress. A same-well re-upload while an earlier run is still in flight is handled with a
`run_token` compare-and-swap (no queue needed at this scale) so a slow first run can never
clobber a faster second one's result.

**Status vs. confidence, kept distinct:** `status` (`"processing"`/`"ready"`/`"failed"`)
tracks whether the pipeline ran without crashing. It does NOT mean the tie was good --
`tie_low_confidence`/`synthetic_low_confidence` (correlation below 0.3, or the shift search
pinned to its boundary) are separate, explicit flags, surfaced in both the status endpoint
and the Dashboard's upload status banner, so a failed or low-confidence tie is never rendered
as if it were a normal result. A `stale` flag catches the case where a later upload for a
different well replaced the active SEG-Y volume while this well's own background run was
still in flight.

**Disk-persisted, well_id-keyed cache:** `backend/app/well_processing_cache_repository.py`
(one JSON file per well under `backend/data/well_processing_cache/`, same repository pattern
as `synthetic_tie_repository.py`) stores scalars only (correlation, shift, flags -- never the
big time-series arrays every page already gets from the existing live endpoints). This is
also what the four new agent tools below read from.

---

## 6. Backend API reference

| Method | Path | Description |
|---|---|---|
| POST | `/wells/upload` | Upload one or more raw `.las` files (multipart form, field name `files`) |
| GET | `/wells` | List all processed wells with summary stats |
| GET | `/wells/{well_id}/curves` | Full processed curve data (raw + computed), row-oriented JSON |
| GET | `/wells/{well_id}/zones` | Zonation summary (thickness + avg PHIE/SWE/VSH per zone) |
| GET | `/wells/{well_id}/crossplot?x=NPHI&y=RHOB&color=VSH` | Generic crossplot data, any curve pair + optional color-by |
| GET | `/dashboard/summary` | Field-wide aggregated stats + per-well summaries |
| POST | `/dashboard/upload` | Combined well (`.las`) + seismic (`.sgy`/`.segy`) upload (multipart form, fields `las_file`/`segy_file`). The well is processed immediately; seismic upload, well-to-seismic tie, synthetic seismogram, and a spectral summary run as a background job -- see "Dashboard combined upload" below |
| GET | `/dashboard/upload/{well_id}/status` | Poll the background pipeline's progress: `"processing"`\|`"ready"`\|`"failed"`, plus explicit `tie_low_confidence`/`synthetic_low_confidence`/`stale` flags (never silently rendered as a normal result) |
| POST | `/chat` | Anthropic agent, streams Server-Sent Events |
| GET | `/wells/{well_id}/export?format=csv\|las` | Download interpreted curves |
| POST | `/seismic/upload` | Upload one or more raw `.sgy`/`.segy` files (multipart form, field name `files`) |
| GET | `/seismic` | List all processed seismic datasets with summary stats |
| GET | `/seismic/{dataset_id}/section` | Subsampled raw amplitude section (trace x two-way-time) for display |
| GET | `/seismic/{dataset_id}/attributes` | Per-trace RMS amplitude, envelope, dominant frequency, and VSH/PHIE/SWE seismic proxies |
| GET | `/seismic/{dataset_id}/export` | Download per-trace computed seismic attributes as CSV |
| GET | `/tie/{well_id}?seismic_dataset_id=...` | Well-to-seismic tie: sonic/density synthetic seismogram cross-correlated against the nearest real seismic trace |
| GET | `/api/seismic/survey-info` | Geometry summary of the raw SEG-Y volume (inline/crossline range, sample interval, time range, trace count) |
| GET | `/api/seismic/inline/{inline_number}` | Inline section: amplitude vs. crossline x two-way-time |
| GET | `/api/seismic/crossline/{crossline_number}` | Crossline section: amplitude vs. inline x two-way-time |
| GET | `/api/seismic/timeslice?time_ms=...` | Map-view amplitude time slice (inline x crossline grid) at the nearest sample to the requested time |
| GET | `/api/seismic/well-tie/{well_id}?wavelet_freq_hz=25` | Well tie computed directly against the raw SEG-Y volume (see "Seismic Visualization" below) |
| GET | `/api/seismic/spectrum?inline_number=...` | Average amplitude spectrum (whole volume or one inline) + dominant frequency/bandwidth/S-N-proxy stats |
| GET | `/api/seismic/spectral-decomp/inline/{inline_number}?method=stft\|cwt&frequency_hz=...` | Spectral decomposition: full time x freq x position volume if `frequency_hz` omitted, or a single frequency's energy across the section if given (fast path for a slider) |
| GET | `/api/seismic/spectral-decomp/trace?inline_number=...&crossline_number=...&method=stft\|cwt` | Spectral decomposition (time x freq) for a single trace |
| GET | `/api/synthetic/{well_id}/generate?wavelet_method=&wavelet_freq_hz=&density_method=` | Full synthetic seismogram + well tie (see "Synthetic Seismogram module" below) |
| GET | `/api/synthetic/{well_id}/nearest-trace` | Lightweight nearest-trace lookup without generating the full synthetic |
| GET / PUT / DELETE | `/api/synthetic/{well_id}/tie` | Get / save / clear a well's persisted manual stretch/squeeze control points |
| GET | `/api/synthetic/{well_id}/export?...` | CSV export: synthetic vs. real trace + tie-quality/QC summary header |
| GET | `/health` | Liveness check |

Interactive OpenAPI docs are available at `http://localhost:8000/docs` once the backend is
running.

### `/chat` request/response shape

```jsonc
// POST /chat
{
  "message": "Why is the interval at 1050-1060m classified as pay in Z-04?",
  "well_id": "Z-04",              // optional; scopes the "currently viewing" context
  "conversation_history": [       // optional; prior turns for multi-turn context
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

Response is `text/event-stream`, each event a JSON line prefixed with `data: `:

```
data: {"type": "text_delta", "text": "Looking at that interval..."}
data: {"type": "tool_call", "name": "get_curve_values", "input": {...}, "output": {...}}
data: {"type": "done"}
```

---

## 7. Anthropic agent

- Model: `claude-sonnet-5` by default (current recommended general-purpose Claude model at
  time of writing). Override with the `ANTHROPIC_MODEL` env var in `backend/.env` without
  any code changes.
- Tools exposed to Claude (all backed by real computed data, never hallucinated):
  `get_well_summary(well_id)`, `get_curve_values(well_id, curve_name, depth_min?, depth_max?)`,
  `get_zone_breakdown(well_id)`, `compare_wells(well_ids, metric)`,
  `list_seismic_datasets()`, `get_seismic_summary(dataset_id)`,
  `get_well_seismic_tie(well_id)`, `get_synthetic_seismogram(well_id)`,
  `get_spectral_decomposition(well_id)`, `get_survey_info()`.
- The last four tools are a deliberately separate family from `list_seismic_datasets`/
  `get_seismic_summary` (different subsystem -- see "Dashboard combined upload" below):
  they read the well-processing cache the dashboard's combined upload populates in the
  background (cache-first, falling back to a live computation for a well not uploaded
  through that flow), and return direct computed results (a real tie cross-correlation
  search, real spectral analysis), not heuristics. Each carries an explicit
  `low_confidence` (tie/synthetic) or `available` (spectral/survey) flag -- correlation
  below 0.3, or the shift search pinned to its boundary -- so a bad result is always
  surfaced plainly rather than narrated around.
- System prompt instructs Claude to ground every numeric claim in a tool result, to flag
  when an answer depends on a tunable assumption (Rw, Swirr, matrix density, Archie
  exponents, zone cutoffs) that should be reviewed by an SME, to always caveat
  `list_seismic_datasets`/`get_seismic_summary`'s VSH/PHIE/SWE proxies as uncalibrated
  amplitude heuristics rather than measured rock properties whenever they come up, and to
  always state a tie/synthetic/spectral tool's `low_confidence`/`available` flag plainly
  when it's true.
- The frontend keeps one shared "active well/dataset" (Zustand `useAppStore`), set by the
  Dashboard's combined upload and read by the Seismic and Synthetic Seismogram pages' chat
  panels (`wellId` prop on `ChatPanel`) -- so a question like "what's the correlation for
  this tie" on any of the three pages resolves against whichever well is currently active,
  without the user needing to name it.
- Requires `ANTHROPIC_API_KEY` in `backend/.env` (see `backend/.env.example`).

---

## 8. Frontend UI

- **Light mode only**, enforced at multiple levels: `<meta name="color-scheme" content="light only">`
  in `index.html`, `color-scheme: light only` in global CSS, no Tailwind `dark:` classes
  anywhere, and every chart (Plotly + Recharts) explicitly styled with light backgrounds/
  gridlines via `frontend/src/styles/tokens.ts` rather than relying on library defaults.
- **Dashboard** (`/`): a combined well+seismic upload widget (`DashboardUpload.tsx`, auto-
  processed in the background -- see "Dashboard combined upload" above) alongside the
  original single-file-type `UploadWells`/`SeismicUpload` widgets, field-wide summary cards,
  per-well bar chart, sortable wells table, field-wide chat panel.
- **Single-well view** (`/wells/:wellId`): multi-track log display (GR/VSH, Resistivity,
  RHOB-NPHI overlay, DT, PHIE/PHIT/SWE, ZONES color column), zone summary table, crossplot
  builder (with the 5 required presets: Neutron-Density, Pickett plot, PHIE vs
  PERM_TIXIER, VSH vs Depth, PHIE vs Depth -- plus free-form curve/color pickers), CSV/LAS
  export buttons, and a well-scoped chat panel. PNG export of any chart uses Plotly's
  built-in camera icon in each chart's mode bar.
- **Seismic module** (`/seismic`, plus a compact summary card on the dashboard): SEG-Y
  upload, dataset picker, raw amplitude section (Plotly heatmap, trace index vs two-way
  time), and computed attribute trends (RMS amplitude, envelope, dominant frequency, and
  the heuristic VSH/PHIE/SWE seismic proxies -- always shown with an on-screen "uncalibrated
  heuristic" caveat), plus a CSV export of the per-trace attributes. The same page's bottom
  "Seismic Visualization" panel (tabbed) adds inline/crossline sections, time slices, a
  direct-SEG-Y well tie, amplitude spectrum, and spectral decomposition (STFT/CWT). Now also
  has a chat panel, and its dataset picker / well-tie selection seed from the dashboard's
  shared active well/dataset after a combined upload.
- **Synthetic Seismogram module** (`/synthetic`): well selector (same pattern as the other
  pages, now also seeded from the shared active well), density/wavelet method pickers,
  prominent QC badges (vertical assumption, no checkshot, coordinate unit conversion status,
  washout interval count, and a low-confidence-tie badge below the 0.3 correlation
  threshold), acoustic impedance + reflectivity depth tracks, wavelet time-domain +
  amplitude/phase spectra, synthetic-vs-real trace overlay with correlation/shift stats, a
  washout interval list, an editable manual stretch/squeeze control-point table (persisted
  per well), CSV export, and a chat panel.
- **Cross-page active well/dataset**: `frontend/src/store/useAppStore.ts`'s `activeWellId`/
  `activeDatasetId` (set by the Dashboard's combined upload) are read by the Seismic and
  Synthetic Seismogram pages' well/dataset selectors and by all three pages' chat panels, so
  a newly uploaded well/seismic appears everywhere without manual re-selection -- a manual
  pick from any page's own selector still overrides it until the active well changes again.

---

## 9. Known limitations / follow-ups

- No database yet -- `backend/app/repository.py` defines a `WellRepository` interface with
  a `FileWellRepository` (Parquet + JSON on disk) implementation. Swap in a
  `PostgresWellRepository` there when ready; no router/service code needs to change.
- Local dependency installation and test execution were **not** performed while building
  this in the current environment (no network access for `pip`/`npm install`). Run
  `pytest tests/ -v` in `backend/` and `npm run build` in `frontend/` on a machine with
  network access before deploying.
- `CORE_PERM_PRED` is a proxy model (see section 5) until real core plug data is supplied.
- TVD/deviation survey and DPTM/checkshot integration are documented placeholders pending
  real survey/VSP data (see section 5).
- The seismic VSH/PHIE/SWE proxies are uncalibrated amplitude heuristics (see "Seismic module
  caveats" in section 5) -- calibrate against a real well tie before using them for anything
  beyond lateral trend screening.
- The seismic section viewer subsamples large SEG-Y volumes to at most 400 traces x 800
  samples per response (`MAX_SECTION_TRACES` / `MAX_SECTION_SAMPLES` in
  `seismic_service.py`) to keep browser payloads reasonable -- full-resolution display of
  very large 3D surveys would need a tiled/windowed viewer instead.
- The Synthetic Seismogram module's depth-time anchoring (`t0_ms` defaulting to the seismic
  volume's first sample time -- see "Synthetic Seismogram module" in section 5) is an
  arbitrary, non-degenerate starting point, not a physically derived one -- a real tie still
  needs the manual stretch/squeeze controls (or a real checkshot) to be trustworthy.
- The Gardner and rock-physics density alternatives in the Synthetic Seismogram module are
  comparison/fallback paths for when RHOB is unavailable, not independently validated against
  core measurements -- prefer real RHOB (the default) for these wells, since all 7 have it.
- Manual stretch/squeeze is a numeric control-point table, not true drag-on-chart interaction
  -- functionally equivalent (add/edit/remove MD+shift pairs, persisted per well) but not as
  fluid as dragging points directly on the tie plot.
