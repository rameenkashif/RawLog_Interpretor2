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

**Current state of Z-02..Z-08's coordinates, and why they aren't final:** the field's well
database (GeoGraphix export) gives real surface X/Y for each well, but those values (X ~
1.20-1.21M, Y ~9.68-9.70M) are in a visibly different coordinate system than `origional.segy`'s
trace coordinates (X ~363k-371k, Y ~2.95M-2.96M) -- not just a constant offset (a real
projection/CRS difference, confirmed by the two datasets' coordinate *ranges* not scaling the
same way, which a simple false-origin shift would preserve). Converting between them correctly
needs the source and target CRS/projection definitions (e.g. via `pyproj`), which aren't
available yet. Until then, `XWELL`/`YWELL` hold **placeholder coordinates inside
`origional.segy`'s real survey footprint**, rescaled from the real well database so the wells'
*relative* positions roughly match the real field layout -- close enough for the well-tie
feature to run end-to-end and return a real nearest trace, but the absolute location and the
resulting tie should not be trusted as geophysically correct until the real well coordinates
are converted into the SEG-Y's CRS. Swap in the converted real coordinates in
`backend/data/raw/Z-0X_raw.las`'s `~Well` section (see `las_loader.py` for the accepted
mnemonics) once the CRS is known, and re-run `bulk_load_wells.py`.

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
  `list_seismic_datasets()`, `get_seismic_summary(dataset_id)`.
- System prompt instructs Claude to ground every numeric claim in a tool result, to flag
  when an answer depends on a tunable assumption (Rw, Swirr, matrix density, Archie
  exponents, zone cutoffs) that should be reviewed by an SME, and to always caveat the
  seismic VSH/PHIE/SWE proxies as uncalibrated amplitude heuristics rather than measured
  rock properties whenever they come up.
- Requires `ANTHROPIC_API_KEY` in `backend/.env` (see `backend/.env.example`).

---

## 8. Frontend UI

- **Light mode only**, enforced at multiple levels: `<meta name="color-scheme" content="light only">`
  in `index.html`, `color-scheme: light only` in global CSS, no Tailwind `dark:` classes
  anywhere, and every chart (Plotly + Recharts) explicitly styled with light backgrounds/
  gridlines via `frontend/src/styles/tokens.ts` rather than relying on library defaults.
- **Dashboard** (`/`): upload widget, field-wide summary cards, per-well bar chart, sortable
  wells table, field-wide chat panel.
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
  heuristic" caveat), plus a CSV export of the per-trace attributes.

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
