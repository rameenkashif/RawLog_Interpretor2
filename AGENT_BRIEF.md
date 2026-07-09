# Prompt for Zed Agent Panel: Multi-Well Petrophysical Interpretation Platform

> **How to use this in Zed:** Open the Agent Panel (Assistant), start a new thread, and paste
> the whole prompt below. Zed's agent works best when you let it work in stages rather than
> asking for everything at once — after it scaffolds the backend, review the diff, accept,
> then say "continue with section 4" etc. Keep this file itself in the repo root as
> `AGENT_BRIEF.md` so the agent can re-read it in later threads/sessions.

---

## ⚠️ Non-negotiable UI requirement
**The entire frontend must use a LIGHT MODE UI by default.** White/very-light-gray
background (`#FFFFFF` / `#F8F9FB`), dark text (`#1A1A1A` or similar), no dark theme,
no `prefers-color-scheme: dark` auto-switching unless there is an explicit manual toggle
that still **defaults to light**. Every component — dashboard, single-well view, chat panel,
charts/plots — must render on a light background. If a charting library defaults to a dark
theme (e.g., some Plotly templates), explicitly override the template to a light one
(`plotly_white` or a custom light theme). Do not use Tailwind's `dark:` classes as the
default state.

---

## Context
I have raw LAS well log files (Z-02 through Z-08) for a single field, each containing only
five raw curves: **DEPT, GR, RESISTIVITY, RHOB, NPHI, DT**. I need a full-stack application
that (1) reads these raw logs, (2) computes standard petrophysical interpretation curves
from them using industry-standard formulas, (3) exposes an Anthropic Claude-powered chatbot
agent that can answer questions about the data, and (4) presents everything through a
dashboard with cross-well and single-well views, including cross-plots and scatter plots
for further interpretation.

Build this as a complete, working project — not a prototype. Include error handling,
loading states, and inline comments explaining the petrophysical logic (not just the code).

---

## 1. Tech Stack

**Backend:** Python 3.11, FastAPI, `lasio` (LAS I/O), `numpy`, `pandas`, `scikit-learn`
(for CORE_PERM_PRED regression), `pydantic` for schemas.

**Frontend:** React + TypeScript (Vite), Tailwind CSS (light theme only — see UI requirement
above), Recharts or Plotly.js for charting, Zustand or React Query for state/data fetching.

**Agent layer:** Anthropic Messages API (`claude-sonnet-4-6` or later — confirm current
recommended model string via Anthropic's docs before finalizing), called from the FastAPI
backend as a dedicated `/chat` endpoint.

**Storage:** Start with in-memory / local file cache (processed Parquet or CSV per well) so
it runs without a DB; make it swappable for Postgres later (leave a clean repository layer).

---

## 2. Input Data Contract

Each well's raw LAS file contains:
| Curve | Description | Unit |
|---|---|---|
| DEPT | Depth | m |
| GR | Gamma Ray | API |
| RESISTIVITY | Deep resistivity | ohm.m |
| RHOB | Bulk density | g/cc |
| NPHI | Neutron porosity | v/v |
| DT | Sonic transit time | us/ft |

Build a LAS parser module (`las_loader.py`) that:
- Reads all wells in a folder using `lasio`
- Validates required curves are present, flags missing/null (`-9999.25`) values
- Returns a clean `pandas.DataFrame` per well plus metadata (well name, start/stop depth, step)

---

## 3. Petrophysical Calculation Engine

Create a module `petrophysics.py` implementing each of the following. Use clean, documented
functions — one calculation per function — so they can be unit tested and swapped out.
Include the formulas as docstrings.

### 3.1 VSH — Volume of Shale
Gamma Ray Index:
```
IGR = (GR - GR_clean) / (GR_shale - GR_clean)
```
`GR_clean` and `GR_shale` should be derived automatically as the 5th and 95th percentile
of the GR curve per well (with override capability via config), not hardcoded.

**Larionov correction** (use "older rocks" formula by default, expose a toggle for
"Tertiary/unconsolidated"):
```
VSH_older    = 0.33 * (2^(2*IGR) - 1)
VSH_tertiary = 0.083 * (2^(3.7*IGR) - 1)
```
Clip VSH to [0, 1].

### 3.2 PHIT — Total Porosity (density-derived)
```
PHIT = (RHOB_matrix - RHOB) / (RHOB_matrix - RHOB_fluid)
```
Default `RHOB_matrix = 2.65` (sandstone), `RHOB_fluid = 1.0` (fresh mud filtrate) — both
configurable (limestone matrix = 2.71, dolomite = 2.87).

### 3.3 PHIE — Effective Porosity (shale-corrected)
```
PHIE = PHIT - (VSH * PHIT_shale)
```
where `PHIT_shale` is PHIT evaluated at the shale baseline point. Clip PHIE to [0, PHIT].
Optionally cross-check with density-neutron combination:
```
PHIE_DN = (PHID + PHIN) / 2
PHIE_DN = sqrt((PHID^2 + PHIN^2) / 2)   # gas-corrected, if PHIN < PHID
```

### 3.4 SWE — Water Saturation (Archie's Equation)
```
Sw = ( (a * Rw) / (PHIE^m * Rt) ) ^ (1/n)
```
Defaults: `a = 1`, `m = 2`, `n = 2`. `Rt` = RESISTIVITY curve. `Rw` (formation water
resistivity) must be a configurable input per well — expose it as a parameter, never
hardcode. Clip SWE to [0, 1].

### 3.5 DPTM — Depth/Time Track
If no checkshot/VSP time-depth table is supplied, approximate by integrating DT:
```
TWT_increment = DT * step_depth / (2 * 3.28084 * 1e6)   # us/ft sonic -> two-way time (s)
DPTM = cumulative_sum(TWT_increment)
```
Build as an optional module; flag clearly in code comments and README as an approximation
pending real checkshot/VSP data.

### 3.6 MD and TVD
For vertical/near-vertical wells (no deviation survey provided): `TVD = MD = DEPT`. Build the
module so a deviation survey (inclination/azimuth vs depth) can later be supplied to compute
true TVD via minimum curvature method. Document this as a placeholder assumption.

### 3.7 PERM_TIXIER — Tixier Permeability
```
K = 250 * ( PHIE^3 / Swirr )^2        # Tixier, medium-gravity oil
```
`Swirr` estimated as `Sw` in the cleanest, highest-PHIE interval, or passed as a config
constant (default 0.2–0.3). Output K in mD.

### 3.8 CORE_PERM_PRED — Predicted Core Permeability
No real core plug data is available yet, so build this as a regression model:
- Train a `RandomForestRegressor` (or simple log-linear model) using PHIE, VSH, and
  PERM_TIXIER as features, target = PERM_TIXIER as a proxy target unless real core
  permeability measurements are supplied later.
- Architect as swappable: `predict_core_perm(df, model_path)` +
  `train_core_perm_model(training_df)`, so real core data can be dropped in later.

### 3.9 VVOLC — Volume of Volcanics
No PEF curve available, so estimate via a density-neutron crossplot heuristic:
- Flag intervals where `RHOB > 2.7 AND NPHI < 0.15 AND GR < GR_shale*0.6` as volcanic.
- Scale fractional `VVOLC` (0–1) linearly within RHOB range 2.7–2.9 → 0–1.
- Document clearly as a heuristic lithology proxy, not true mineralogical decomposition —
  needs calibration against cuttings/core descriptions if available.

### 3.10 ZONES — Reservoir Zonation
```
Reservoir     : VSH < 0.4  AND PHIE > 0.08 AND SWE < 0.65
Pay           : Reservoir AND SWE < 0.5
Non-reservoir : everything else
```
Make all cutoffs configurable per well/field. Output categorical `ZONES` curve
(1 = pay, 2 = reservoir non-pay, 3 = non-reservoir) + human-readable label.

> Every constant above (GR percentiles, matrix density, Rw, Archie a/m/n, Swirr, cutoffs)
> must live in a single `config/petrophysics_config.yaml` per well — never hardcode inline.

---

## 4. Backend API (FastAPI)

- `POST /wells/upload` — upload one or more raw LAS files
- `GET /wells` — list all processed wells with summary stats
- `GET /wells/{well_id}/curves` — full processed curve data (raw + computed) as JSON
- `GET /wells/{well_id}/zones` — zonation summary table
- `GET /wells/{well_id}/crossplot?x=NPHI&y=RHOB&color=VSH` — generic crossplot data endpoint
  (must support any curve pair + optional color-by curve)
- `GET /dashboard/summary` — aggregated multi-well statistics for the dashboard
- `POST /chat` — Anthropic agent endpoint (see section 5)
- `GET /wells/{well_id}/export` — export processed LAS/CSV of the interpreted log

Use Pydantic response models for all endpoints, with 404/422 error handling.

---

## 5. Anthropic Agent / Chatbot

Build a `/chat` endpoint that:
1. Accepts `{ message: string, well_id?: string, conversation_history: [] }`
2. Calls the Anthropic Messages API
3. Gives Claude tool/function-calling access to backend functions:
   `get_well_summary(well_id)`, `get_curve_values(well_id, curve_name, depth_range)`,
   `get_zone_breakdown(well_id)`, `compare_wells(well_ids, metric)` — so the agent pulls
   real computed data rather than hallucinating values.
4. Streams responses back to the frontend chat UI (SSE or WebSocket).
5. System prompt: act as a petrophysics assistant, explain interpretations in plain
   language when asked, always ground numeric answers in tool results, flag when a
   cutoff/assumption (Rw, Swirr, etc.) may need SME review.

Frontend: a persistent **light-themed** chat panel (collapsible sidebar), available on both
the dashboard and single-well view.

---

## 6. Frontend — Dashboard (Multi-Well View)
*(Light UI: white/near-white background, dark-gray/black text, soft neutral borders,
accent color used sparingly for highlights — no dark cards.)*

- Field-wide summary cards: number of wells, total footage logged, average VSH/PHIE/SWE,
  net pay thickness by well
- Bar/column chart comparing key metrics across wells (light chart background, gridlines
  in light gray, not white-on-black)
- Table of all wells with sortable columns (depth range, avg PHIE, avg SWE, zone counts)
- Field-wide chatbot panel for cross-well questions

## 7. Frontend — Single Well View

- Classic multi-track log display (GR/VSH, Resistivity, RHOB-NPHI overlay, DT, computed
  PHIE/PHIT/SWE track, ZONES color column) on a **light background** — reuse a log-track
  component, don't rebuild per well
- **Scatter plots / cross-plots module**, minimum set:
  - Neutron–Density crossplot (NPHI vs RHOB), colored by VSH or ZONES
  - Pickett plot (log Rt vs log PHIE) for Sw QC
  - PHIE vs PERM_TIXIER (log-scale y-axis)
  - VSH vs Depth, PHIE vs Depth trend plots
  - Generic "pick any two curves + optional color-by third curve" scatter builder
    (dropdowns, not hardcoded pairs)
- Zone summary table per well (thickness, avg PHIE/SWE per zone)
- Export button (CSV/LAS of interpreted curves, PNG of any chart)
- Well-scoped chatbot panel

---

## 8. Project Structure
```
/backend
  /app
    las_loader.py
    petrophysics.py
    config/petrophysics_config.yaml
    routers/ (wells.py, chat.py, dashboard.py)
    services/anthropic_agent.py
    models/schemas.py
  main.py
  requirements.txt
/frontend
  /src
    /components (LogTrackViewer, CrossplotBuilder, ChatPanel, Dashboard, WellTable)
    /pages (DashboardPage, WellDetailPage)
    /styles (light-theme tokens: colors, spacing — no dark mode variables)
    /api (client.ts)
  package.json
AGENT_BRIEF.md   <- this file
README.md        <- setup + how formulas/config work
```

---

## 9. How to work through this in Zed
Work in stages, reviewing each diff before moving on:
1. Backend calculation engine (`petrophysics.py`) + unit tests for every formula in
   section 3, using the raw LAS sample data for realistic ranges
2. LAS loader + FastAPI endpoints (section 4)
3. Anthropic agent integration (section 5) — ask me for the API key / env var setup here
4. React dashboard (section 6) — **light mode UI, confirm the color tokens with me before
   building every page**
5. Single-well view + cross-plot builder (section 7)
6. README documenting every assumption/default constant (Rw, Swirr, cutoffs, matrix
   density) so a petrophysicist can tune them without touching code

Ask me for field-specific constants (Rw, matrix density, cutoffs) before finalizing defaults,
and confirm the light-mode color palette before generating frontend components.

---

## Implementation status (updated by the agent)

All 9 sections above have been implemented in this repository. See `README.md` in the repo
root for setup instructions, the full list of configurable assumptions/defaults, and how to
tune them without touching code. Key notes for future sessions:

- Backend lives in `backend/`, frontend in `frontend/`. Both are scaffolded and complete.
- `backend/app/petrophysics.py` implements every formula in section 3, unit tested in
  `backend/tests/test_petrophysics.py` against synthetic (not yet real) well data, since no
  real Z-02..Z-08 LAS files were present in the repo at the time of the initial build. Drop
  real files into `backend/data/raw/` and run `python scripts/bulk_load_wells.py`.
- All tunable constants live in `backend/app/config/petrophysics_config.yaml` (field-wide
  `defaults` + optional per-well `wells.<WELL_ID>` overrides) -- nothing is hardcoded inline.
- The Anthropic agent (`backend/app/services/anthropic_agent.py`) defaults to the
  `claude-sonnet-5` model ID, overridable via the `ANTHROPIC_MODEL` env var.
- The frontend enforces light mode only (no dark theme, no `dark:` classes) per the
  non-negotiable UI requirement -- see `frontend/tailwind.config.js` and
  `frontend/src/styles/tokens.ts`.
- Local `pip install` / `npm install` could not be run in the environment that built this
  (network restrictions), so dependencies have NOT been installed or test-run locally.
  Install and run them in CI/CD or on a machine with network access before relying on this
  code -- see README.md "Getting Started" for exact commands.
- The app was renamed from "PetroInterp" to **"RawReservoirClassifier"** partway through
  (see `frontend/src/App.tsx`, `frontend/index.html`, `backend/main.py`).
- **Seismic module added** (post-initial-build, user request): raw SEG-Y upload/parsing via
  `segyio` (`backend/app/segy_loader.py`), a seismic attribute engine
  (`backend/app/seismic_attributes.py`) computing RMS amplitude, envelope, and dominant
  frequency, plus explicitly-flagged **uncalibrated heuristic** VSH/PHIE/SWE seismic proxies
  (amplitude-based, NOT measured rock properties -- see README.md "Seismic module caveats").
  Mirrors the well pipeline's architecture exactly: its own repository
  (`seismic_repository.py`), service layer (`seismic_service.py`), router (`routers/seismic.py`),
  config file (`config/seismic_config.yaml`), unit tests (`tests/test_seismic_attributes.py`),
  and bulk-load script (`scripts/bulk_load_seismic.py`). Frontend: a dedicated `/seismic` page
  (raw amplitude heatmap + attribute trend charts) plus a compact summary module on the
  dashboard, both linked via a new "Seismic" nav item and 2 new Anthropic agent tools
  (`list_seismic_datasets`, `get_seismic_summary`).
- **Well/seismic surface coordinates added** (post-well-tie-feature, user request): the raw
  `Z-02_raw.las` .. `Z-08_raw.las` files now carry `XWELL`/`YWELL` surface coordinates in
  their `~Well` header section (`las_loader.py` resolves those or common aliases into
  `WellMetadata.well_x/well_y`, exposed on `WellSummary`). `segy_loader.py` now extracts
  per-trace coordinates from SEG-Y trace headers (`CDP_X`/`CDP_Y`, falling back to
  `SourceX`/`SourceY`, with the coordinate scalar applied) into `LoadedSegy.trace_x/trace_y`,
  persisted by `seismic_repository.py`. `tie_service.py` now prefers a real Euclidean
  nearest-trace match (`well_seismic_tie.find_nearest_trace_index`) whenever both sides have
  coordinates, reporting the true `distance_m` and `tie_method="nearest_trace"`; it falls back
  to the pre-existing manually configured `trace_index` (`tie_config.yaml`,
  `tie_method="manual_override"`) when either side lacks coordinates. See README.md
  "Well-to-seismic tie" for details.
- **Seismic Visualization added** (post-well-tie-feature, user request): reads the raw SEG-Y
  volume directly (`backend/app/services/seismic_processor.py`, `SegyVolume`) rather than the
  upload pipeline, since inline/crossline sectioning needs geometry that pipeline never
  stores. This file's inline/crossline live at non-standard trace header bytes 9-12/13-16 (not
  segyio's `INLINE_3D`/`CROSSLINE_3D`), read explicitly and cross-checked at open time. New
  `/api/seismic/*` router (`routers/seismic_viz.py`, kept separate from the existing
  `routers/seismic.py` to avoid colliding with it) serves inline/crossline sections, time
  slices, an amplitude spectrum, and a well tie (reusing `well_seismic_tie.py` +
  `well_service.py`). Frontend: 5 new components under `components/Seismic/`
  (`SeismicPanel.tsx` tabs the other 4), wired into the bottom of `/seismic`.
- **Real well coordinates from the well database, rescaled into the SEG-Y's footprint**
  (post-Seismic-Visualization, user request): the user supplied real X/Y for Z-02..Z-08 from a
  GeoGraphix export, but those coordinates are in a visibly different CRS/projection than
  `origional.segy`'s trace coordinates (confirmed by the two datasets' coordinate ranges not
  scaling proportionally -- not just a constant offset). Without the source/target CRS
  definitions to do a real conversion, `XWELL`/`YWELL` in the 7 LAS files were set to
  placeholder coordinates *inside* the real survey's footprint, linearly rescaled from the
  real well database so relative field geometry is preserved -- functional for exercising the
  well-tie feature end-to-end, but not geophysically correct until the real CRS conversion is
  done. See README.md "Current state of Z-02..Z-08's coordinates" for how to replace them once
  the CRS is known.
- **Spectral decomposition added** (post-Seismic-Visualization, user request): extends
  `seismic_processor.SegyVolume` (not a new module -- builds on the existing class) with STFT
  and hand-rolled complex-Morlet CWT time-frequency decomposition (`scipy.signal.cwt`/`morlet2`
  are gone in this project's scipy 1.17+, so CWT is hand-rolled the same way
  `well_seismic_tie.ricker_wavelet()` already is). Two new endpoints on the *same*
  `routers/seismic_viz.py` router: `/api/seismic/spectral-decomp/inline/{n}` (full volume, or a
  single-frequency section slice -- cached per (inline, method) so a frontend frequency slider
  doesn't recompute the whole decomposition per drag) and `/api/seismic/spectral-decomp/trace`
  (single trace). Frontend: new `SpectralDecompView.tsx` as a 5th `SeismicPanel.tsx` tab,
  reusing `SeismicSectionView`'s Plotly heatmap pattern with a debounced frequency slider and
  STFT/CWT toggle. See README.md "Seismic Visualization" for the window/wavelet parameter
  tradeoffs.
