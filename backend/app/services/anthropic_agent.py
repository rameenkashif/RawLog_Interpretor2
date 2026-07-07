"""
anthropic_agent.py
-------------------
Wires the Anthropic Messages API into a petrophysics assistant with
tool/function-calling access to real backend data (section 5 of the brief).

The model is never allowed to "make up" a numeric answer -- every tool
below returns real computed values pulled from the processed wells via
well_service, so the system prompt instructs Claude to always ground
numeric claims in tool results.

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

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = """You are a petrophysics assistant embedded in a well log interpretation \
platform. You help geoscientists and engineers understand computed petrophysical curves \
(VSH, PHIT, PHIE, SWE, PERM_TIXIER, CORE_PERM_PRED, VVOLC, ZONES) across a set of wells \
(Z-02 through Z-08).

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
4. Several curves are explicitly heuristic/proxy calculations, not direct measurements: \
   VVOLC (density-neutron crossplot heuristic, uncalibrated against cuttings/core), \
   CORE_PERM_PRED (a regression trained on PERM_TIXIER as a proxy target, not real core plugs), \
   and DPTM (sonic-integration approximation pending real checkshot/VSP data). Mention this \
   caveat when discussing those curves.
5. Be concise. Use units (m, API, ohm.m, g/cc, v/v, mD) when quoting values.
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


# -----------------------------------------------------------------------------
# Tool implementations -- these call into well_service, which is the same
# service layer the REST routers use, so the agent and the UI never
# disagree about the underlying numbers.
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


TOOL_DISPATCH = {
    "get_well_summary": _tool_get_well_summary,
    "get_curve_values": _tool_get_curve_values,
    "get_zone_breakdown": _tool_get_zone_breakdown,
    "compare_wells": _tool_compare_wells,
}


def _run_tool(name: str, tool_input: dict[str, Any]) -> Any:
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'"}
    try:
        return fn(**tool_input)
    except well_service.WellNotFoundError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 -- surface tool errors to the model, not a 500
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
