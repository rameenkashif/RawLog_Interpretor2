"""
anthropic_agent.py
-------------------
Wires the Anthropic Messages API into a petrophysics assistant with
tool/function-calling access to real backend data (section 5 of the brief).

The model is never allowed to "make up" a numeric answer -- every tool
below returns real computed values pulled from the processed wells (via
well_service) or seismic datasets (via seismic_service), so the system
prompt instructs Claude to always ground numeric claims in tool results.

Model selection: defaults to `claude-sonnet-5` (current recommended
general-purpose model at time of writing -- confirmed via Anthropic's
model docs). Override via the ANTHROPIC_MODEL env var if a newer model
should be used without a code change.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic

from app.services import well_service

# The seismic tools depend on the optional seismic module (segyio, scipy). If
# those packages aren't installed, degrade gracefully: the chat agent still
# works for wells, just without seismic tools, rather than crashing the whole
# app (chat.py -> anthropic_agent.py -> here, at import time).
try:
    from app.services import seismic_service

    _SEISMIC_AVAILABLE = True
except Exception:  # noqa: BLE001
    seismic_service = None  # type: ignore[assignment]
    _SEISMIC_AVAILABLE = False

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = """You are a petrophysics assistant embedded in a well log and seismic \
interpretation platform. You help geoscientists and engineers understand computed \
petrophysical curves (VSH, PHIT, PHIE, SWE, PERM_TIXIER, CORE_PERM_PRED, VVOLC, ZONES) \
across a set of wells (Z-02 through Z-08), as well as seismic attribute data derived from \
uploaded SEG-Y datasets.

Rules you must follow:
1. ALWAYS ground numeric answers in tool results. Never estimate, guess, or recall a number \
   from general petrophysics knowledge when a tool can retrieve the real, computed value \
   for this dataset. If a tool call fails or returns no data, say so plainly instead of \
   filling in a plausible-sounding number.
2. Explain interpretations in plain language when asked "why" or "what does this mean" -- \
   e.g. why a zone was classified as pay, or what a high VSH implies about reservoir quality.
3. Flag explicitly whenever an answer depends on an assumption or cutoff that a subject-matter \
   expert (SME) should review -- in particular: Rw (formation water resistivity), Swirr \
   (irreducible water saturation), matrix density, Archie a/m/n exponents, and the VSH/PHIE/SWE \
   zone cutoffs. These are configurable defaults, not measured constants, and can materially \
   change the interpretation.
4. Several well curves are explicitly heuristic/proxy calculations, not direct measurements: \
   VVOLC (density-neutron crossplot heuristic, uncalibrated against cuttings/core), \
   CORE_PERM_PRED (a regression trained on PERM_TIXIER as a proxy target, not real core plugs), \
   and DPTM (sonic-integration approximation pending real checkshot/VSP data). Mention this \
   caveat when discussing those curves.
5. The seismic VSH_SEISMIC_PROXY, PHIE_SEISMIC_PROXY, and SWE_SEISMIC_PROXY attributes \
   returned by list_seismic_datasets and get_seismic_summary are UNCALIBRATED, amplitude-based \
   heuristics -- NOT measured shale volume, porosity, or water saturation. They require a real \
   well tie before being used for interpretation. ALWAYS state this caveat explicitly whenever \
   you report or discuss a value from those two tools specifically, and never conflate them with \
   the log-derived VSH/PHIE/SWE curves from wells.
6. Be concise. Use units (m, API, ohm.m, g/cc, v/v, mD, ms, Hz) when quoting values.
7. get_well_seismic_tie, get_synthetic_seismogram, get_spectral_decomposition, and \
   get_survey_info return direct computed results (real cross-correlation searches, real \
   spectral analysis, real survey geometry) -- the rule 5 heuristic caveat does NOT apply to \
   them. Instead, the tie/synthetic tool results carry a low_confidence flag (spectral/survey \
   results carry an available flag). ALWAYS check this flag and state it plainly when it's true \
   -- e.g. "the tie for well X has low confidence (correlation 0.21, below the 0.3 threshold)" \
   or "the shift search pinned to its boundary, which usually means a spurious match, not a \
   genuine tie" -- rather than reporting the correlation number alone or narrating around a bad \
   result. The user is normally asking about whichever well is currently active in the UI (see \
   the "(The user is currently viewing well ...)" note appended to their message); use that \
   well_id unless they name a different one.
"""

TOOLS = [
    {
        "name": "get_well_summary",
        "description": (
            "Get summary statistics for one well: depth range, sample count, average VSH/PHIE/SWE, "
            "and net pay thickness."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"well_id": {"type": "string", "description": "e.g. 'Z-02'"}},
            "required": ["well_id"],
        },
    },
    {
        "name": "get_curve_values",
        "description": (
            "Get raw values for a specific curve in a well, optionally restricted to a depth range. "
            "Use this to answer questions about specific depth intervals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "well_id": {"type": "string"},
                "curve_name": {
                    "type": "string",
                    "description": (
                        "One of: DEPT, GR, RESISTIVITY, RHOB, NPHI, DT, VSH, PHIT, PHIE, PHIE_DN, "
                        "SWE, PERM_TIXIER, CORE_PERM_PRED, VVOLC, ZONES, ZONES_LABEL, DPTM, MD, TVD"
                    ),
                },
                "depth_min": {
                    "type": "number",
                    "description": "Optional minimum depth (m)",
                },
                "depth_max": {
                    "type": "number",
                    "description": "Optional maximum depth (m)",
                },
            },
            "required": ["well_id", "curve_name"],
        },
    },
    {
        "name": "get_zone_breakdown",
        "description": (
            "Get the reservoir zonation breakdown for a well: thickness and average PHIE/SWE/VSH "
            "for Pay, Reservoir (non-pay), and Non-reservoir zones."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"well_id": {"type": "string"}},
            "required": ["well_id"],
        },
    },
    {
        "name": "compare_wells",
        "description": "Compare a single metric (e.g. avg_phie, avg_swe, avg_vsh, net_pay_thickness) across multiple wells.",
        "input_schema": {
            "type": "object",
            "properties": {
                "well_ids": {"type": "array", "items": {"type": "string"}},
                "metric": {
                    "type": "string",
                    "description": "One of: avg_vsh, avg_phie, avg_swe, net_pay_thickness, footage_logged",
                },
            },
            "required": ["well_ids", "metric"],
        },
    },
]

# Seismic tools are only advertised to Claude when the seismic module
# actually loaded successfully (see the try/except import above) -- this
# keeps the tool list honest about what the agent can actually do.
SEISMIC_TOOLS = [
    {
        "name": "list_seismic_datasets",
        "description": (
            "List all processed seismic (SEG-Y) datasets with summary stats, including the "
            "uncalibrated VSH/PHIE/SWE seismic proxies. Use this to discover what seismic data "
            "is available before calling get_seismic_summary."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_seismic_summary",
        "description": (
            "Get summary statistics for one seismic dataset: trace/sample counts, sample interval, "
            "duration, average RMS amplitude, and the average uncalibrated VSH/PHIE/SWE seismic "
            "proxies. Always caveat these proxies as uncalibrated amplitude heuristics, not "
            "measured rock properties, when reporting them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "e.g. 'LINE_001'"}
            },
            "required": ["dataset_id"],
        },
    },
]

# Distinct subsystem from list_seismic_datasets/get_seismic_summary above
# (deliberately not merged with them, see dashboard_upload_service.py's
# docstring) -- these read the well-processing cache populated by the
# dashboard's combined upload background job (cache-first, live-compute
# fallback for wells not uploaded through that flow), never recomputing a
# tie/synthetic seismogram inside a chat turn when a cached result exists.
TIE_SYNTHETIC_TOOLS = [
    {
        "name": "get_well_seismic_tie",
        "description": (
            "Get the well-to-seismic tie for one well: correlation, winning wavelet frequency, "
            "polarity, bulk time shift, and trace location. A direct computed result (a real "
            "frequency/polarity/shift cross-correlation search against the real seismic trace), "
            "not a heuristic proxy. ALWAYS check and state the low_confidence flag plainly -- "
            "true means the correlation is weak (below 0.3) or the shift search pinned to its "
            "search boundary (likely a spurious match), and must be reported as such rather than "
            "narrated around."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"well_id": {"type": "string", "description": "e.g. 'Z-02'"}},
            "required": ["well_id"],
        },
    },
    {
        "name": "get_synthetic_seismogram",
        "description": (
            "Get the synthetic seismogram / well-tie summary for one well: correlation, "
            "polarity, best time shift, the depth-time datum plausibility check, and washout "
            "interval count. A direct computed result, not a heuristic. ALWAYS check and state "
            "the low_confidence flag plainly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"well_id": {"type": "string", "description": "e.g. 'Z-02'"}},
            "required": ["well_id"],
        },
    },
    {
        "name": "get_spectral_decomposition",
        "description": (
            "Get the dominant frequency, -3dB bandwidth, and S/N proxy of the seismic amplitude "
            "spectrum at the inline nearest to one well's tie point. A direct computed result "
            "from the real seismic trace, not a heuristic. Requires a tie to already exist for "
            "the well (available=false with an error otherwise -- call get_well_seismic_tie "
            "first if unsure)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"well_id": {"type": "string", "description": "e.g. 'Z-02'"}},
            "required": ["well_id"],
        },
    },
    {
        "name": "get_survey_info",
        "description": (
            "Get geometry metadata for the currently active SEG-Y survey (the single volume "
            "backing Seismic Visualization and the Synthetic Seismogram page): trace/sample "
            "counts, inline/crossline range, sample interval, and two-way-time range."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]

if _SEISMIC_AVAILABLE:
    TOOLS = TOOLS + SEISMIC_TOOLS + TIE_SYNTHETIC_TOOLS


# -----------------------------------------------------------------------------
# Tool implementations -- these call into well_service / seismic_service,
# the same service layers the REST routers use, so the agent and the UI
# never disagree about the underlying numbers.
# -----------------------------------------------------------------------------
def _tool_get_well_summary(well_id: str) -> dict[str, Any]:
    summary = well_service.get_well_summary(well_id)
    return summary.model_dump()


def _tool_get_curve_values(
    well_id: str,
    curve_name: str,
    depth_min: float | None = None,
    depth_max: float | None = None,
) -> dict[str, Any]:
    _, df = well_service.get_well_df(well_id)
    if curve_name not in df.columns:
        return {
            "error": f"Curve '{curve_name}' not found. Available: {list(df.columns)}"
        }

    subset = df
    if depth_min is not None:
        subset = subset[subset["DEPT"] >= depth_min]
    if depth_max is not None:
        subset = subset[subset["DEPT"] <= depth_max]

    values = subset[curve_name].dropna()
    if values.empty:
        return {"well_id": well_id, "curve_name": curve_name, "count": 0}

    return {
        "well_id": well_id,
        "curve_name": curve_name,
        "count": int(len(values)),
        "mean": float(values.mean()) if values.dtype.kind in "fi" else None,
        "min": float(values.min()) if values.dtype.kind in "fi" else None,
        "max": float(values.max()) if values.dtype.kind in "fi" else None,
        # Cap the returned sample so we don't blow the context window on long logs.
        "sample_values": values.head(50).tolist(),
    }


def _tool_get_zone_breakdown(well_id: str) -> dict[str, Any]:
    zones = well_service.get_well_zones(well_id)
    return zones.model_dump()


def _tool_compare_wells(well_ids: list[str], metric: str) -> dict[str, Any]:
    results = {}
    for well_id in well_ids:
        try:
            summary = well_service.get_well_summary(well_id)
            results[well_id] = getattr(summary, metric, None)
        except well_service.WellNotFoundError:
            results[well_id] = None
    return {"metric": metric, "values": results}


def _tool_list_seismic_datasets() -> dict[str, Any]:
    summaries = seismic_service.list_seismic_summaries()
    return {"datasets": [s.model_dump() for s in summaries]}


def _tool_get_seismic_summary(dataset_id: str) -> dict[str, Any]:
    summary = seismic_service.get_seismic_summary(dataset_id)
    return summary.model_dump()


def _tool_get_well_seismic_tie(well_id: str) -> dict[str, Any]:
    from app.services import dashboard_upload_service

    return dashboard_upload_service.get_tie_summary(well_id)


def _tool_get_synthetic_seismogram(well_id: str) -> dict[str, Any]:
    from app.services import dashboard_upload_service

    return dashboard_upload_service.get_synthetic_summary(well_id)


def _tool_get_spectral_decomposition(well_id: str) -> dict[str, Any]:
    from app.services import dashboard_upload_service

    return dashboard_upload_service.get_spectral_summary(well_id)


def _tool_get_survey_info() -> dict[str, Any]:
    from app.services import seismic_processor as sp

    try:
        info = sp.get_segy_volume().survey_info()
        return {"available": True, **vars(info)}
    except sp.SegyFileNotFoundError as exc:
        return {"available": False, "error": str(exc)}


TOOL_DISPATCH = {
    "get_well_summary": _tool_get_well_summary,
    "get_curve_values": _tool_get_curve_values,
    "get_zone_breakdown": _tool_get_zone_breakdown,
    "compare_wells": _tool_compare_wells,
}

if _SEISMIC_AVAILABLE:
    TOOL_DISPATCH["list_seismic_datasets"] = _tool_list_seismic_datasets
    TOOL_DISPATCH["get_seismic_summary"] = _tool_get_seismic_summary
    TOOL_DISPATCH["get_well_seismic_tie"] = _tool_get_well_seismic_tie
    TOOL_DISPATCH["get_synthetic_seismogram"] = _tool_get_synthetic_seismogram
    TOOL_DISPATCH["get_spectral_decomposition"] = _tool_get_spectral_decomposition
    TOOL_DISPATCH["get_survey_info"] = _tool_get_survey_info


def _run_tool(name: str, tool_input: dict[str, Any]) -> Any:
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'"}
    try:
        return fn(**tool_input)
    except well_service.WellNotFoundError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 -- surface tool errors to the model, not a 500
        if _SEISMIC_AVAILABLE and isinstance(
            exc, seismic_service.SeismicDatasetNotFoundError
        ):
            return {"error": str(exc)}
        return {"error": f"Tool '{name}' failed: {exc}"}


def _build_context_message(well_id: str | None) -> str:
    if not well_id:
        return ""
    return f"\n\n(The user is currently viewing well '{well_id}' in the UI.)"


def _get_client() -> AsyncAnthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it in backend/.env or your shell before starting the server."
        )
    return AsyncAnthropic(api_key=api_key)


async def stream_chat_response(
    message: str,
    well_id: str | None,
    conversation_history: list[dict[str, str]],
) -> AsyncIterator[dict[str, Any]]:
    """Run one turn of the agent loop (with tool calling), streaming events
    back as they occur. Yields dicts of the shape:
        {"type": "text_delta", "text": "..."}
        {"type": "tool_call", "name": "...", "input": {...}, "output": {...}}
        {"type": "done"}
        {"type": "error", "message": "..."}
    Intended to be adapted directly onto an SSE response by routers/chat.py.
    """
    client = _get_client()

    messages: list[dict[str, Any]] = [
        {"role": m["role"], "content": m["content"]} for m in conversation_history
    ]
    messages.append(
        {"role": "user", "content": message + _build_context_message(well_id)}
    )

    try:
        # Agent loop: keep going while Claude asks for tool calls.
        while True:
            assistant_text = ""
            tool_uses: list[dict[str, Any]] = []

            async with client.messages.stream(
                model=DEFAULT_MODEL,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            ) as stream:
                async for event in stream:
                    if (
                        event.type == "content_block_delta"
                        and event.delta.type == "text_delta"
                    ):
                        assistant_text += event.delta.text
                        yield {"type": "text_delta", "text": event.delta.text}

                final_message = await stream.get_final_message()

            for block in final_message.content:
                if block.type == "tool_use":
                    tool_uses.append(
                        {"id": block.id, "name": block.name, "input": block.input}
                    )

            messages.append({"role": "assistant", "content": final_message.content})

            if final_message.stop_reason != "tool_use":
                break

            tool_results = []
            for tool_use in tool_uses:
                output = _run_tool(tool_use["name"], tool_use["input"])
                yield {
                    "type": "tool_call",
                    "name": tool_use["name"],
                    "input": tool_use["input"],
                    "output": output,
                }
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use["id"],
                        "content": json.dumps(output, default=str),
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        yield {"type": "done"}

    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": str(exc)}
