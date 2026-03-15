"""Tests for autoevolve — multi-agent competition orchestrator."""

import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "auto-evolve"))

from auto_evolve import (
    AgentConfig,
    AgentStatus,
    EvolveConfig,
    Experiment,
    STRATEGIES,
    _build_hints_content,
    _compute_improvements,
    _generate_program_md,
    _parse_results_tsv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exp(commit: str, val_bpb: float, status: str = "keep", desc: str = "test") -> Experiment:
    return Experiment(
        commit=commit,
        val_bpb=val_bpb,
        memory_gb=45.0,
        status=status,
        description=desc,
    )


VALID_TSV = (
    "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
    "abc1234\t4.02\t45.2\tkeep\tBaseline\n"
    "def5678\t3.95\t45.5\tkeep\tIncrease depth\n"
    "ghi9012\t4.10\t46.0\tdiscard\tBad batch size\n"
    "jkl3456\t3.90\t44.8\tkeep\tTune LR\n"
)


# ---------------------------------------------------------------------------
# Parse results TSV
# ---------------------------------------------------------------------------

class TestParseResultsTsv:
    def test_valid_with_header(self):
        exps = _parse_results_tsv(VALID_TSV)
        assert len(exps) == 4
        assert exps[0].commit == "abc1234"
        assert exps[0].val_bpb == 4.02
        assert exps[0].status == "keep"

    def test_skips_header(self):
        exps = _parse_results_tsv(VALID_TSV)
        assert not any(e.commit == "commit" for e in exps)

    def test_empty_string(self):
        assert _parse_results_tsv("") == []

    def test_malformed_rows_skipped(self):
        tsv = (
            "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
            "abc\t3.9\t45.0\tkeep\tgood\n"
            "bad\tnotanumber\n"
            "def\t3.8\t44.0\tkeep\talso good\n"
        )
        exps = _parse_results_tsv(tsv)
        assert len(exps) == 2

    def test_description_with_tabs(self):
        tsv = (
            "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
            "abc\t3.9\t45.0\tkeep\tMulti\tword\tdesc\n"
        )
        exps = _parse_results_tsv(tsv)
        assert len(exps) == 1
        assert "Multi" in exps[0].description


# ---------------------------------------------------------------------------
# Compute improvements
# ---------------------------------------------------------------------------

class TestComputeImprovements:
    def test_consecutive_keeps_with_improvement(self):
        exps = [
            _exp("a", 4.0),
            _exp("b", 3.9),
            _exp("c", 3.8),
        ]
        improvements = _compute_improvements(exps)
        assert len(improvements) == 2
        assert abs(improvements[0][1] - 0.1) < 1e-6
        assert abs(improvements[1][1] - 0.1) < 1e-6

    def test_no_keeps(self):
        exps = [
            _exp("a", 4.0, "discard"),
            _exp("b", 4.1, "discard"),
        ]
        assert _compute_improvements(exps) == []

    def test_regression_not_counted(self):
        exps = [
            _exp("a", 4.0),
            _exp("b", 4.1),
        ]
        improvements = _compute_improvements(exps)
        assert len(improvements) == 0

    def test_mixed_keep_discard(self):
        exps = [
            _exp("a", 4.0),
            _exp("b", 4.2, "discard"),
            _exp("c", 3.8),
        ]
        improvements = _compute_improvements(exps)
        assert len(improvements) == 1
        assert abs(improvements[0][1] - 0.2) < 1e-6

    def test_crash_skipped(self):
        exps = [
            _exp("a", 4.0),
            _exp("b", 0.0, "crash"),
            _exp("c", 3.9),
        ]
        improvements = _compute_improvements(exps)
        assert len(improvements) == 1


# ---------------------------------------------------------------------------
# Build hints content
# ---------------------------------------------------------------------------

class TestBuildHintsContent:
    def _make_config(self) -> EvolveConfig:
        return EvolveConfig(
            tag="test",
            base_branch="main",
            base_commit="abc123",
            created_at="2025-01-01T00:00:00+00:00",
            agents=[
                AgentConfig(id=1, branch="evolve/test-agent-1", strategy="architecture-first"),
            ],
        )

    def _make_leader(self) -> AgentStatus:
        return AgentStatus(
            agent=AgentConfig(id=1, branch="evolve/test-agent-1", strategy="architecture-first"),
            experiments=[_exp("a", 4.0), _exp("b", 3.8)],
            best_val_bpb=3.8,
            best_experiment=_exp("b", 3.8, desc="best experiment"),
            keep_count=2,
        )

    def test_contains_leader_info(self):
        config = self._make_config()
        leader = self._make_leader()
        impactful = [(_exp("b", 3.8, desc="Increase depth"), 0.2)]
        content = _build_hints_content(config, leader, impactful)
        assert "Agent 1" in content
        assert "3.8" in content
        assert "Architecture First" in content

    def test_lists_impactful_experiments(self):
        config = self._make_config()
        leader = self._make_leader()
        impactful = [
            (_exp("b", 3.8, desc="Increase depth"), 0.2),
            (_exp("c", 3.7, desc="Tune LR"), 0.1),
        ]
        content = _build_hints_content(config, leader, impactful)
        assert "Increase depth" in content
        assert "Tune LR" in content

    def test_empty_impactful(self):
        config = self._make_config()
        leader = self._make_leader()
        content = _build_hints_content(config, leader, [])
        assert "No impactful" in content


# ---------------------------------------------------------------------------
# Generate program MD
# ---------------------------------------------------------------------------

class TestGenerateProgramMd:
    def test_contains_strategy(self):
        strategy = STRATEGIES[0]
        md = _generate_program_md(strategy, agent_id=1, tag="mar15")
        assert strategy["label"] in md
        assert strategy["guidance"] in md

    def test_contains_agent_id(self):
        md = _generate_program_md(STRATEGIES[0], agent_id=3, tag="test")
        assert "Agent 3" in md

    def test_contains_tag(self):
        md = _generate_program_md(STRATEGIES[0], agent_id=1, tag="mar15")
        assert "mar15" in md

    def test_contains_rules(self):
        md = _generate_program_md(STRATEGIES[0], agent_id=1, tag="test")
        assert "results.tsv" in md
        assert "train.py" in md


# ---------------------------------------------------------------------------
# Strategies constant
# ---------------------------------------------------------------------------

class TestStrategies:
    def test_has_six_strategies(self):
        assert len(STRATEGIES) == 6

    def test_all_have_required_keys(self):
        for s in STRATEGIES:
            assert "key" in s
            assert "label" in s
            assert "guidance" in s

    def test_unique_keys(self):
        keys = [s["key"] for s in STRATEGIES]
        assert len(keys) == len(set(keys))
