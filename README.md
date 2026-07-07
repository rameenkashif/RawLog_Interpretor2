# Multi-Well Petrophysical Interpretation Platform

A full-stack application that reads raw LAS well logs (DEPT, GR, RESISTIVITY, RHOB, NPHI, DT),
computes standard petrophysical interpretation curves (VSH, PHIT, PHIE, SWE, PERM_TIXIER,
CORE_PERM_PRED, VVOLC, ZONES, ...), exposes an Anthropic Claude-powered petrophysics
assistant, and presents everything through a light-themed dashboard with cross-well and
single-well views, log tracks, and crossplots.

> **Status:** Code-complete per `AGENT_BRIEF.md`. Dependencies have **not** been installed or
> run locally in this environment (no network access for `pip`/`npm`). Install and validate
> on a machine/CI with network access using the steps below before relying on this in
> production.

---

## 1. Project layout

```
backend/
  app/
    config/petrophysics_config.yaml   <- every tunable constant lives here
    config_loader.py                  <- merges field defaults + per-well overrides
    las_loader.py                     <- raw LAS -> validated DataFrame + metadata
    petrophysics.py                   <- every calculation in section 3, one function each
    repository.py                     <- storage layer (Parquet/JSON today, swappable for Postgres)
    models/schemas.py                 <- Pydantic request/response models
    routers/
      wells.py                        <- upload/list/curves/zones/crossplot/export
      dashboard.py                    <- field-wide summary
      chat.py                         <- SSE-streaming Anthropic agent endpoint
    services/
      well_service.py                 <- application service layer (routers call this)
      anthropic_agent.py              <- Claude Messages API + tool-calling agent loop
  scripts/
    bulk_load_wells.py                <- CLI to process backend/data/raw/*.las
    train_core_perm_model.py          <- trains the CORE_PERM_PRED proxy model
  data/
    raw/                              <- put Z-02.las ... Z-08.las here
    processed/                        <- Parquet + JSON cache (gitignored, regenerable)
    models/                           <- trained sklearn models (gitignored, regenerable)
  tests/
    conftest.py                       <- synthetic 3-zone well fixture
    test_petrophysics.py              <- unit tests for every formula in section 3
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
  `get_zone_breakdown(well_id)`, `compare_wells(well_ids, metric)`.
- System prompt instructs Claude to ground every numeric claim in a tool result and to flag
  when an answer depends on a tunable assumption (Rw, Swirr, matrix density, Archie
  exponents, zone cutoffs) that should be reviewed by an SME.
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
