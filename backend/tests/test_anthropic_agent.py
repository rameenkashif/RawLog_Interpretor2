"""
test_anthropic_agent.py
--------------------------
Tests for services/anthropic_agent.py's tool wiring -- specifically
list_wells, added after a real production incident: the chat agent had no
way to discover valid well_ids, only a hardcoded system-prompt example
("Z-02 through Z-08") that drifted from the real IDs LAS filenames
actually produce (e.g. 'Z-02_RAW', from las_loader._well_id_from_filename
uppercasing the file stem) -- causing the agent to repeatedly guess wrong
well_ids with no way to self-correct. These tests guard against that
regressing silently again.

Not a full agent-loop test (that needs a live Anthropic API call) -- just
the tool registration and dispatch, which is what actually broke.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.repository import FileWellRepository
from app.services import anthropic_agent, well_service

RAW_LAS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
Z02_PATH = RAW_LAS_DIR / "Z-02_raw.las"
Z03_PATH = RAW_LAS_DIR / "Z-03_raw.las"


@pytest.fixture
def well_repo(tmp_path):
    return FileWellRepository(base_dir=tmp_path / "wells")


class TestListWellsTool:
    def test_registered_in_tools_and_dispatch(self):
        tool_names = {t["name"] for t in anthropic_agent.TOOLS}
        assert "list_wells" in tool_names
        assert "list_wells" in anthropic_agent.TOOL_DISPATCH
        assert anthropic_agent.TOOL_DISPATCH["list_wells"] is anthropic_agent._tool_list_wells

    def test_takes_no_required_arguments(self):
        tool = next(t for t in anthropic_agent.TOOLS if t["name"] == "list_wells")
        assert tool["input_schema"].get("required", []) == []

    def test_returns_every_loaded_well_id(self, well_repo, monkeypatch):
        well_service.process_and_store_las_bytes(Z02_PATH.read_bytes(), "Z-02_raw.las", repo=well_repo)
        well_service.process_and_store_las_bytes(Z03_PATH.read_bytes(), "Z-03_raw.las", repo=well_repo)
        monkeypatch.setattr(
            well_service, "list_well_summaries",
            lambda repo=None, _f=well_service.list_well_summaries: _f(repo=well_repo),
        )

        result = anthropic_agent._tool_list_wells()

        well_ids = {w["well_id"] for w in result["wells"]}
        assert well_ids == {"Z-02_RAW", "Z-03_RAW"}

    def test_returns_empty_list_rather_than_erroring_when_nothing_loaded(self, well_repo, monkeypatch):
        monkeypatch.setattr(
            well_service, "list_well_summaries",
            lambda repo=None, _f=well_service.list_well_summaries: _f(repo=well_repo),
        )
        result = anthropic_agent._tool_list_wells()
        assert result == {"wells": []}


class TestSystemPromptDoesNotHardcodeWellIds:
    def test_no_stale_z02_through_z08_example(self):
        # The exact regression: this literal string used to appear in
        # SYSTEM_PROMPT and drifted from the real well_id format
        # (LAS files are named Z-0X_raw.las -> well_id 'Z-0X_RAW', not
        # 'Z-0X'), silently misleading the agent into guessing wrong IDs
        # with no tool to self-correct. Must never come back verbatim.
        assert "Z-02 through Z-08" not in anthropic_agent.SYSTEM_PROMPT

    def test_instructs_calling_list_wells_before_guessing(self):
        assert "list_wells" in anthropic_agent.SYSTEM_PROMPT


class TestGetFieldOverviewTool:
    def test_registered_in_tools_and_dispatch(self):
        tool_names = {t["name"] for t in anthropic_agent.TOOLS}
        assert "get_field_overview" in tool_names
        assert anthropic_agent.TOOL_DISPATCH["get_field_overview"] is anthropic_agent._tool_get_field_overview

    def test_takes_no_required_arguments(self):
        tool = next(t for t in anthropic_agent.TOOLS if t["name"] == "get_field_overview")
        assert tool["input_schema"].get("required", []) == []

    def test_dispatches_to_dashboard_upload_service(self, monkeypatch):
        from app.services import dashboard_upload_service

        monkeypatch.setattr(dashboard_upload_service, "get_field_overview", lambda: {"wells": ["sentinel"]})
        assert anthropic_agent._tool_get_field_overview() == {"wells": ["sentinel"]}


class TestSystemPromptReasoningWorkflow:
    """The agent used to answer interpretive questions off a single tool
    call; these guard the reasoning-workflow guidance that tells it to
    gather independent evidence, weigh agreement/conflict, and use
    get_field_overview instead of looping per-well calls itself."""

    def test_mentions_get_field_overview_for_cross_well_questions(self):
        assert "get_field_overview" in anthropic_agent.SYSTEM_PROMPT

    def test_instructs_weighing_agreement_or_conflict_between_evidence(self):
        prompt_lower = anthropic_agent.SYSTEM_PROMPT.lower()
        assert "agree" in prompt_lower and "conflict" in prompt_lower

    def test_forbids_hiding_synthesis_behind_an_invented_score(self):
        assert "composite score" in anthropic_agent.SYSTEM_PROMPT.lower()
