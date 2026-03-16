"""
Behavior-focused tests for auto_steer.

Tests cover: classification, TSV parsing, category statistics,
suggestion generation, priority scoring, deduplication, and the CLI interface.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pytest
from click.testing import CliRunner

# auto-steer lives outside the standard package tree; add it to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "auto-steer"))

from auto_steer import (
    KNOWN_DIRECTIONS,
    Category,
    CategoryStats,
    Experiment,
    KnownDirection,
    ParseResult,
    RiskLevel,
    Status,
    Strategy,
    Suggestion,
    SuggestionKind,
    _compute_priority_score,
    _direction_already_tried,
    classify_experiment,
    cli,
    compute_category_stats,
    generate_suggestions,
    parse_results_tsv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exp(
    desc: str = "some experiment",
    status: str = "keep",
    val_bpb: float = 3.9,
    category: Category = Category.OTHER,
    diff_text: str = "",
    commit: str = "abc1234",
) -> Experiment:
    """Build an Experiment with sensible defaults so tests stay terse."""
    return Experiment(
        commit=commit,
        val_bpb=val_bpb,
        memory_gb=45.0,
        status=Status(status),
        description=desc,
        category=category,
        diff_text=diff_text,
    )


def _baseline(val_bpb: float = 4.0) -> Experiment:
    """Convenience: the first experiment is always the baseline."""
    return _exp(desc="Baseline: default config", commit="base000", val_bpb=val_bpb)


def _write_tsv(path: Path, rows: list[list[str]], header: Optional[list[str]] = None) -> None:
    """Write a TSV file with given header and rows."""
    if header is None:
        header = ["commit", "val_bpb", "memory_gb", "status", "description"]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n")


def _empty_stats() -> dict[Category, CategoryStats]:
    """Return zeroed-out stats for every category."""
    return {
        cat: CategoryStats(category=cat, total=0, keeps=0, discards=0, crashes=0, avg_improvement_pct=0.0)
        for cat in Category
    }


def _extract_json(output: str) -> dict:
    """Extract JSON from CLI output, skipping any non-JSON prefix lines (e.g. git warnings)."""
    idx = output.find("{")
    if idx == -1:
        raise ValueError(f"No JSON found in output: {output!r}")
    return json.loads(output[idx:])


VALID_TSV = (
    "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
    "abc1234\t4.02\t45.2\tkeep\tBaseline\n"
    "def5678\t3.95\t45.5\tkeep\tIncrease depth to 10\n"
    "ghi9012\t4.10\t46.0\tdiscard\tBad batch size\n"
    "jkl3456\t3.90\t44.8\tkeep\tTune matrix LR\n"
)


# ===================================================================
# 1. Classification
# ===================================================================


class TestClassifyExperiment:
    """classify_experiment returns the correct Category based on keywords."""

    # ---- architecture keywords ----

    def test_architecture_keyword_layer(self) -> None:
        assert classify_experiment("Add extra layer", "") == Category.ARCHITECTURE

    def test_architecture_keyword_attention(self) -> None:
        assert classify_experiment("Modify attention heads", "") == Category.ARCHITECTURE

    def test_architecture_keyword_mlp(self) -> None:
        assert classify_experiment("Wider mlp expansion", "") == Category.ARCHITECTURE

    def test_architecture_keyword_transformer(self) -> None:
        assert classify_experiment("Different transformer block", "") == Category.ARCHITECTURE

    def test_architecture_keyword_gqa(self) -> None:
        assert classify_experiment("Enable gqa grouping", "") == Category.ARCHITECTURE

    def test_architecture_keyword_depth(self) -> None:
        assert classify_experiment("increase depth to 12 layers", "") == Category.ARCHITECTURE

    def test_architecture_keyword_window(self) -> None:
        assert classify_experiment("change window pattern", "") == Category.ARCHITECTURE

    # ---- hyperparams keywords ----

    def test_hyperparams_keyword_learning_rate(self) -> None:
        assert classify_experiment("tune learning rate to 3e-4", "") == Category.HYPERPARAMS

    def test_hyperparams_keyword_batch_size(self) -> None:
        assert classify_experiment("Larger batch_size", "") == Category.HYPERPARAMS

    def test_hyperparams_keyword_warmup(self) -> None:
        assert classify_experiment("Add warmup schedule", "") == Category.HYPERPARAMS

    def test_hyperparams_keyword_warmdown(self) -> None:
        assert classify_experiment("Adjust warmdown ratio", "") == Category.HYPERPARAMS

    def test_hyperparams_keyword_schedule(self) -> None:
        assert classify_experiment("cosine schedule decay", "") == Category.HYPERPARAMS

    # ---- optimizer keywords ----

    def test_optimizer_keyword_muon(self) -> None:
        assert classify_experiment("adjust muon momentum", "") == Category.OPTIMIZER

    def test_optimizer_keyword_adam(self) -> None:
        assert classify_experiment("Adjust adam betas", "") == Category.OPTIMIZER

    def test_optimizer_keyword_momentum(self) -> None:
        assert classify_experiment("Higher momentum", "") == Category.OPTIMIZER

    def test_optimizer_keyword_newton_schulz(self) -> None:
        assert classify_experiment("tune newton-schulz iterations", "") == Category.OPTIMIZER

    # ---- activation keywords ----

    def test_activation_keyword_gelu(self) -> None:
        assert classify_experiment("Switch to gelu", "") == Category.ACTIVATION

    def test_activation_keyword_swiglu(self) -> None:
        assert classify_experiment("switch to swiglu activation", "") == Category.ACTIVATION

    def test_activation_keyword_relu(self) -> None:
        assert classify_experiment("Use relu variant", "") == Category.ACTIVATION

    def test_activation_keyword_silu(self) -> None:
        assert classify_experiment("silu instead of relu", "") == Category.ACTIVATION

    # ---- regularization keywords ----

    def test_regularization_keyword_dropout(self) -> None:
        assert classify_experiment("Add dropout", "") == Category.REGULARIZATION

    def test_regularization_keyword_z_loss_hyphen(self) -> None:
        assert classify_experiment("add z-loss regularization", "") == Category.REGULARIZATION

    def test_regularization_keyword_z_loss_underscore(self) -> None:
        assert classify_experiment("Add z_loss", "") == Category.REGULARIZATION

    def test_optimizer_keyword_weight_decay(self) -> None:
        assert classify_experiment("Tune weight_decay", "") == Category.OPTIMIZER

    def test_regularization_keyword_softcap(self) -> None:
        assert classify_experiment("Adjust softcap value", "") == Category.REGULARIZATION

    # ---- embedding keywords ----

    def test_embedding_keyword_rope(self) -> None:
        assert classify_experiment("tune rope base frequency", "") == Category.EMBEDDING

    def test_embedding_keyword_vocab(self) -> None:
        assert classify_experiment("Larger vocab size", "") == Category.EMBEDDING

    def test_embedding_keyword_rotary(self) -> None:
        assert classify_experiment("change rotary position", "") == Category.EMBEDDING

    def test_embedding_keyword_embed(self) -> None:
        assert classify_experiment("new embed strategy", "") == Category.EMBEDDING

    # ---- efficiency keywords ----

    def test_efficiency_keyword_memory(self) -> None:
        assert classify_experiment("add gradient checkpoint for memory", "") == Category.EFFICIENCY

    def test_efficiency_keyword_compile(self) -> None:
        assert classify_experiment("Faster compile pass", "") == Category.EFFICIENCY

    def test_efficiency_keyword_bf16(self) -> None:
        assert classify_experiment("Switch to bf16", "") == Category.EFFICIENCY

    def test_efficiency_keyword_flash(self) -> None:
        assert classify_experiment("Use flash attention kernel", "") == Category.EFFICIENCY

    # ---- word boundary matching for short keywords ----

    def test_short_keyword_lr_matches_standalone(self) -> None:
        """Short keyword 'lr' should match when standalone (e.g. 'lr=0.05')."""
        assert classify_experiment("set lr=0.05", "") == Category.HYPERPARAMS

    def test_short_keyword_lr_does_not_match_inside_word(self) -> None:
        """Short keyword 'lr' must NOT match inside longer words like 'clearly'."""
        result = classify_experiment("clearly a better approach", "")
        assert result != Category.HYPERPARAMS

    def test_short_keyword_lr_matches_with_underscore_boundary(self) -> None:
        """lr should match when bounded by underscores like 'matrix_lr'."""
        assert classify_experiment("adjust matrix_lr to 0.06", "") == Category.HYPERPARAMS

    def test_short_keyword_at_end_of_string(self) -> None:
        """Short keywords should match at end of string."""
        assert classify_experiment("changed the lr", "") == Category.HYPERPARAMS

    def test_short_keyword_at_start_of_string(self) -> None:
        """Short keywords should match at start of string."""
        assert classify_experiment("lr is too high", "") == Category.HYPERPARAMS

    def test_short_keyword_glu_via_swiglu(self) -> None:
        """'swiglu' (6 chars) is matched via substring, not word boundary."""
        assert classify_experiment("Try swiglu", "") == Category.ACTIVATION

    # ---- diff text classification ----

    def test_diff_text_classifies_when_description_is_generic(self) -> None:
        assert classify_experiment("some change", "MATRIX_LR = 0.06") == Category.HYPERPARAMS

    def test_diff_text_with_activation_change(self) -> None:
        diff = "-activation = relu\n+activation = gelu"
        assert classify_experiment("Minor tweak", diff) == Category.ACTIVATION

    def test_diff_text_with_optimizer_change(self) -> None:
        diff = "- ns_steps = 5\n+ ns_steps = 6"
        assert classify_experiment("iteration count", diff) == Category.OPTIMIZER

    # ---- fallback to OTHER ----

    def test_unknown_description_returns_other(self) -> None:
        assert classify_experiment("try something completely novel", "") == Category.OTHER

    def test_empty_description_and_diff_returns_other(self) -> None:
        assert classify_experiment("", "") == Category.OTHER

    def test_fix_typo_is_other(self) -> None:
        assert classify_experiment("Fix a typo in the readme", "") == Category.OTHER

    # ---- case insensitivity ----

    def test_classification_is_case_insensitive(self) -> None:
        assert classify_experiment("SWIGLU ACTIVATION", "") == Category.ACTIVATION

    # ---- priority ordering ----

    def test_activation_before_architecture_when_both_present(self) -> None:
        """Activation keywords are checked before architecture, so 'relu' wins over 'layer'."""
        result = classify_experiment("Change relu in layer", "")
        assert result == Category.ACTIVATION

    def test_optimizer_before_hyperparams(self) -> None:
        """Optimizer is checked before hyperparams in priority, so 'adam' wins over 'schedule'."""
        result = classify_experiment("adam schedule tuning", "")
        assert result == Category.OPTIMIZER


# ===================================================================
# 2. Parsing
# ===================================================================


class TestParseResultsTsv:
    """parse_results_tsv reads TSV files into ParseResult with proper error handling."""

    def test_valid_tsv_returns_ok(self, tmp_path: Path) -> None:
        f = tmp_path / "results.tsv"
        f.write_text(VALID_TSV)
        result = parse_results_tsv(f)
        assert result.ok
        assert len(result.experiments) == 4

    def test_valid_tsv_parses_fields_correctly(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        _write_tsv(tsv, [
            ["abc1234", "1.050000", "45.2", "keep", "Baseline: default config"],
        ])
        result = parse_results_tsv(tsv)
        exp = result.experiments[0]
        assert exp.commit == "abc1234"
        assert exp.val_bpb == pytest.approx(1.05)
        assert exp.memory_gb == pytest.approx(45.2)
        assert exp.status == Status.KEEP
        assert exp.description == "Baseline: default config"

    def test_missing_file_returns_error(self, tmp_path: Path) -> None:
        result = parse_results_tsv(tmp_path / "nope.tsv")
        assert not result.ok
        assert "not found" in result.error.lower()
        assert result.experiments == []

    def test_empty_file_returns_error(self, tmp_path: Path) -> None:
        f = tmp_path / "results.tsv"
        f.write_text("")
        result = parse_results_tsv(f)
        assert not result.ok
        assert "empty" in result.error.lower()

    def test_header_only_returns_error(self, tmp_path: Path) -> None:
        tsv = tmp_path / "header_only.tsv"
        tsv.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
        result = parse_results_tsv(tsv)
        assert not result.ok
        assert "no experiments" in result.error.lower()

    def test_missing_required_columns_returns_error(self, tmp_path: Path) -> None:
        f = tmp_path / "results.tsv"
        f.write_text("commit\tval_bpb\nabc\t3.5\n")
        result = parse_results_tsv(f)
        assert not result.ok
        assert "missing columns" in result.error.lower()

    def test_invalid_status_defaults_to_discard(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        _write_tsv(tsv, [
            ["abc1234", "1.05", "45.0", "unknown_status", "Some experiment"],
        ])
        result = parse_results_tsv(tsv)
        assert result.ok is True
        assert result.experiments[0].status == Status.DISCARD

    def test_invalid_val_bpb_defaults_to_zero(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        _write_tsv(tsv, [
            ["abc1234", "not_a_number", "45.0", "keep", "Experiment"],
        ])
        result = parse_results_tsv(tsv)
        assert result.ok is True
        assert result.experiments[0].val_bpb == 0.0

    def test_invalid_memory_gb_defaults_to_zero(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        _write_tsv(tsv, [
            ["abc1234", "1.05", "bad_mem", "keep", "Experiment"],
        ])
        result = parse_results_tsv(tsv)
        assert result.ok is True
        assert result.experiments[0].memory_gb == 0.0

    def test_missing_memory_gb_column_defaults_to_zero(self, tmp_path: Path) -> None:
        """memory_gb is optional -- the parser uses .get() with a default."""
        tsv = tmp_path / "results.tsv"
        header = ["commit", "val_bpb", "status", "description"]
        rows = [["abc1234", "1.05", "keep", "Experiment"]]
        lines = ["\t".join(header)]
        for row in rows:
            lines.append("\t".join(row))
        tsv.write_text("\n".join(lines) + "\n")
        result = parse_results_tsv(tsv)
        assert result.ok is True
        assert result.experiments[0].memory_gb == 0.0

    def test_crash_status_parsed_correctly(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        _write_tsv(tsv, [
            ["abc1234", "0.0", "45.0", "crash", "OOM crash"],
        ])
        result = parse_results_tsv(tsv)
        assert result.experiments[0].status == Status.CRASH

    def test_status_is_case_insensitive(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        _write_tsv(tsv, [
            ["abc1234", "1.05", "45.0", "KEEP", "Experiment"],
        ])
        result = parse_results_tsv(tsv)
        assert result.experiments[0].status == Status.KEEP

    def test_whitespace_in_fields_is_stripped(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        lines = [
            "commit\tval_bpb\tmemory_gb\tstatus\tdescription",
            "  abc1234  \t1.05\t45.0\t  keep  \t  Experiment  ",
        ]
        tsv.write_text("\n".join(lines) + "\n")
        result = parse_results_tsv(tsv)
        exp = result.experiments[0]
        assert exp.commit == "abc1234"
        assert exp.status == Status.KEEP
        assert exp.description == "Experiment"

    def test_multiple_experiments_parsed_in_order(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        _write_tsv(tsv, [
            ["a1", "1.05", "45.0", "keep", "First"],
            ["b2", "1.04", "45.0", "discard", "Second"],
            ["c3", "1.03", "45.0", "crash", "Third"],
        ])
        result = parse_results_tsv(tsv)
        assert [e.commit for e in result.experiments] == ["a1", "b2", "c3"]
        assert [e.status for e in result.experiments] == [Status.KEEP, Status.DISCARD, Status.CRASH]


# ===================================================================
# 3. Category statistics
# ===================================================================


class TestComputeCategoryStats:
    """compute_category_stats aggregates experiments by category correctly."""

    def test_empty_experiments_returns_all_zeroed(self) -> None:
        stats = compute_category_stats([], baseline_bpb=None)
        for cat in Category:
            assert stats[cat].total == 0
            assert stats[cat].keeps == 0
            assert stats[cat].avg_improvement_pct == 0.0

    def test_single_baseline_only_all_zeroed(self) -> None:
        """First experiment is treated as baseline and skipped."""
        exps = [_baseline()]
        stats = compute_category_stats(exps, baseline_bpb=4.0)
        assert all(s.total == 0 for s in stats.values())

    def test_multiple_categories_counted(self) -> None:
        exps = [
            _exp("baseline", category=Category.OTHER),
            _exp("depth change", category=Category.ARCHITECTURE),
            _exp("bad depth", "discard", category=Category.ARCHITECTURE, val_bpb=4.1),
            _exp("tune lr", category=Category.HYPERPARAMS),
        ]
        stats = compute_category_stats(exps, baseline_bpb=4.0)
        assert stats[Category.ARCHITECTURE].total == 2
        assert stats[Category.ARCHITECTURE].keeps == 1
        assert stats[Category.ARCHITECTURE].discards == 1
        assert stats[Category.HYPERPARAMS].total == 1

    def test_crashes_counted_separately(self) -> None:
        exps = [
            _baseline(),
            _exp("OOM crash", "crash", category=Category.EFFICIENCY, val_bpb=0.0),
        ]
        stats = compute_category_stats(exps, baseline_bpb=4.0)
        assert stats[Category.EFFICIENCY].crashes == 1
        assert stats[Category.EFFICIENCY].total == 1

    def test_avg_improvement_calculation(self) -> None:
        """Average improvement is ((baseline - val_bpb) / baseline) * 100 for kept experiments."""
        baseline_bpb = 1.00
        exps = [
            _baseline(val_bpb=baseline_bpb),
            _exp("a", category=Category.HYPERPARAMS, val_bpb=0.95),  # 5% improvement
            _exp("b", category=Category.HYPERPARAMS, val_bpb=0.90),  # 10% improvement
        ]
        stats = compute_category_stats(exps, baseline_bpb=baseline_bpb)
        # Average: (5 + 10) / 2 = 7.5%
        assert stats[Category.HYPERPARAMS].avg_improvement_pct == pytest.approx(7.5)

    def test_avg_improvement_ignores_discarded(self) -> None:
        baseline_bpb = 1.00
        exps = [
            _baseline(val_bpb=baseline_bpb),
            _exp("good", category=Category.HYPERPARAMS, val_bpb=0.90),                    # 10%
            _exp("bad", "discard", category=Category.HYPERPARAMS, val_bpb=1.10),  # ignored
        ]
        stats = compute_category_stats(exps, baseline_bpb=baseline_bpb)
        assert stats[Category.HYPERPARAMS].avg_improvement_pct == pytest.approx(10.0)

    def test_avg_improvement_zero_when_no_keeps(self) -> None:
        exps = [
            _baseline(),
            _exp("bad", "discard", category=Category.OPTIMIZER, val_bpb=4.1),
        ]
        stats = compute_category_stats(exps, baseline_bpb=4.0)
        assert stats[Category.OPTIMIZER].avg_improvement_pct == 0.0

    def test_avg_improvement_zero_when_no_baseline(self) -> None:
        exps = [
            _baseline(),
            _exp("good", category=Category.OPTIMIZER, val_bpb=3.8),
        ]
        stats = compute_category_stats(exps, baseline_bpb=None)
        assert stats[Category.OPTIMIZER].avg_improvement_pct == 0.0

    def test_success_rate_property(self) -> None:
        cs = CategoryStats(
            category=Category.ARCHITECTURE, total=4, keeps=1, discards=2, crashes=1, avg_improvement_pct=5.0
        )
        assert cs.success_rate_pct == pytest.approx(25.0)

    def test_success_rate_zero_total(self) -> None:
        cs = CategoryStats(
            category=Category.OTHER, total=0, keeps=0, discards=0, crashes=0, avg_improvement_pct=0.0
        )
        assert cs.success_rate_pct == 0.0

    def test_success_rate_two_thirds(self) -> None:
        """2/3 success rate = ~66.7%."""
        exps = [
            _exp("baseline"),
            _exp("a", category=Category.ARCHITECTURE),
            _exp("b", category=Category.ARCHITECTURE),
            _exp("c", "discard", category=Category.ARCHITECTURE, val_bpb=4.1),
        ]
        stats = compute_category_stats(exps, baseline_bpb=4.0)
        rate = stats[Category.ARCHITECTURE].success_rate_pct
        assert 60.0 < rate < 70.0  # 2/3 = ~66.7%

    def test_experiments_with_zero_val_bpb_excluded_from_improvement(self) -> None:
        """Experiments with val_bpb == 0 (e.g. crashes) are excluded from improvement calc."""
        baseline_bpb = 1.00
        exps = [
            _baseline(val_bpb=baseline_bpb),
            _exp("crashed", category=Category.ACTIVATION, val_bpb=0.0),   # invalid, excluded
            _exp("good", category=Category.ACTIVATION, val_bpb=0.95),     # 5% improvement
        ]
        stats = compute_category_stats(exps, baseline_bpb=baseline_bpb)
        # Only the 0.95 experiment contributes
        assert stats[Category.ACTIVATION].avg_improvement_pct == pytest.approx(5.0)

    def test_skips_first_experiment_as_baseline(self) -> None:
        """Verify the baseline (index 0) is excluded from counts."""
        exps = [
            _exp("first", category=Category.ARCHITECTURE, val_bpb=3.9),
            _exp("second", category=Category.ARCHITECTURE, val_bpb=3.8),
        ]
        stats = compute_category_stats(exps, baseline_bpb=4.0)
        # Only the second experiment should be counted
        assert stats[Category.ARCHITECTURE].total == 1


# ===================================================================
# 4. Direction already tried
# ===================================================================


class TestDirectionAlreadyTried:
    """_direction_already_tried matches known direction keywords against experiment history."""

    def test_returns_true_when_multiple_keywords_match(self) -> None:
        direction = KnownDirection(
            category=Category.ARCHITECTURE,
            title="Test",
            description="test",
            risk=RiskLevel.LOW,
            expected_range=(0.1, 0.3),
            keywords=("depth", "width"),
        )
        exps = [_exp("increase depth and width ratio")]
        assert _direction_already_tried(direction, exps) is True

    def test_returns_false_when_only_one_keyword_matches_with_three_keywords(self) -> None:
        direction = KnownDirection(
            category=Category.ARCHITECTURE,
            title="Test",
            description="Desc",
            risk=RiskLevel.LOW,
            expected_range=(0.1, 0.3),
            keywords=("aspect_ratio", "depth", "width"),
        )
        # Only 'width' matches -- needs min(2, 3) = 2 matches
        exps = [_exp("Changed width only")]
        assert _direction_already_tried(direction, exps) is False

    def test_single_keyword_direction_matches_with_one_hit(self) -> None:
        """When a direction has only one keyword, threshold is min(2, 1) = 1."""
        direction = KnownDirection(
            category=Category.REGULARIZATION,
            title="Test",
            description="Desc",
            risk=RiskLevel.LOW,
            expected_range=(0.1, 0.3),
            keywords=("softcap",),
        )
        exps = [_exp("Adjust softcap value")]
        assert _direction_already_tried(direction, exps) is True

    def test_returns_false_when_no_keywords_match(self) -> None:
        direction = KnownDirection(
            category=Category.ARCHITECTURE,
            title="Test",
            description="test",
            risk=RiskLevel.LOW,
            expected_range=(0.1, 0.3),
            keywords=("swiglu", "silu"),
        )
        exps = [_exp("increase depth")]
        assert not _direction_already_tried(direction, exps)

    def test_underscore_normalization(self) -> None:
        """Keywords with underscores should match experiment text with spaces and vice versa."""
        direction = KnownDirection(
            category=Category.HYPERPARAMS,
            title="Test",
            description="Desc",
            risk=RiskLevel.LOW,
            expected_range=(0.1, 0.3),
            keywords=("batch_size", "total_batch"),
        )
        exps = [_exp("Increase total batch and batch size")]
        assert _direction_already_tried(direction, exps) is True

    def test_diff_text_also_considered(self) -> None:
        direction = KnownDirection(
            category=Category.OPTIMIZER,
            title="Test",
            description="Desc",
            risk=RiskLevel.LOW,
            expected_range=(0.1, 0.3),
            keywords=("ns_steps", "newton"),
        )
        exps = [_exp("Some tweak", diff_text="- ns_steps = 5\n+ ns_steps = 6\n# newton method")]
        assert _direction_already_tried(direction, exps) is True

    def test_empty_experiment_list_returns_false(self) -> None:
        direction = KNOWN_DIRECTIONS[0]
        assert _direction_already_tried(direction, []) is False

    def test_case_insensitive_matching(self) -> None:
        direction = KnownDirection(
            category=Category.ACTIVATION,
            title="Test",
            description="Desc",
            risk=RiskLevel.LOW,
            expected_range=(0.1, 0.3),
            keywords=("swiglu", "gated"),
        )
        exps = [_exp("Tried SWIGLU with GATED linear unit")]
        assert _direction_already_tried(direction, exps) is True

    def test_multiple_experiments_checked(self) -> None:
        """If any single experiment matches, direction is considered tried."""
        direction = KnownDirection(
            category=Category.HYPERPARAMS,
            title="Test",
            description="Desc",
            risk=RiskLevel.LOW,
            expected_range=(0.1, 0.3),
            keywords=("warmup", "warmup_ratio"),
        )
        exps = [
            _exp("Unrelated change"),
            _exp("Add warmup with warmup_ratio=0.05"),
        ]
        assert _direction_already_tried(direction, exps) is True


# ===================================================================
# 5. Priority scoring
# ===================================================================


class TestComputePriorityScore:
    """_compute_priority_score adjusts scores based on strategy, risk, and history."""

    def _make_direction(
        self,
        category: Category = Category.ARCHITECTURE,
        risk: RiskLevel = RiskLevel.MEDIUM,
        expected_range: tuple = (0.3, 0.7),
    ) -> KnownDirection:
        return KnownDirection(
            category=category,
            title="Test direction",
            description="Description",
            risk=risk,
            expected_range=expected_range,
            keywords=("test",),
        )

    def _make_stats(self, total: int = 0, keeps: int = 0) -> dict[Category, CategoryStats]:
        return {
            cat: CategoryStats(cat, total, keeps, total - keeps, 0, 0.0)
            for cat in Category
        }

    def test_base_score_is_midpoint_times_risk_multiplier(self) -> None:
        """With empty stats and AUTO strategy at 0 experiments, check the base calculation."""
        direction = self._make_direction(risk=RiskLevel.MEDIUM, expected_range=(0.4, 0.6))
        stats = _empty_stats()
        score = _compute_priority_score(direction, stats, Strategy.AUTO, total_experiments=0)
        # midpoint = 0.5, risk_multiplier[MEDIUM] = 1.0
        # AUTO with 0 experiments: exploration_weight = max(0.2, 1.0 - 0/20) = 1.0
        # not category_tried -> score *= 1.0 + 1.0 = 2.0
        expected = 0.5 * 1.0 * 2.0
        assert score == pytest.approx(expected)

    def test_low_risk_gets_higher_multiplier(self) -> None:
        stats = _empty_stats()
        low_dir = self._make_direction(risk=RiskLevel.LOW, expected_range=(0.5, 0.5))
        med_dir = self._make_direction(risk=RiskLevel.MEDIUM, expected_range=(0.5, 0.5))
        high_dir = self._make_direction(risk=RiskLevel.HIGH, expected_range=(0.5, 0.5))

        low_score = _compute_priority_score(low_dir, stats, Strategy.EXPLORE, total_experiments=0)
        med_score = _compute_priority_score(med_dir, stats, Strategy.EXPLORE, total_experiments=0)
        high_score = _compute_priority_score(high_dir, stats, Strategy.EXPLORE, total_experiments=0)
        assert low_score > med_score > high_score

    def test_explore_boosts_untried_categories(self) -> None:
        stats_untried = self._make_stats(total=0, keeps=0)
        direction = KnownDirection(
            category=Category.ACTIVATION,
            title="Test",
            description="",
            risk=RiskLevel.MEDIUM,
            expected_range=(0.5, 1.0),
            keywords=(),
        )
        score_explore = _compute_priority_score(direction, stats_untried, Strategy.EXPLORE, 10)
        score_exploit = _compute_priority_score(direction, stats_untried, Strategy.EXPLOIT, 10)
        assert score_explore > score_exploit

    def test_exploit_boosts_successful_categories(self) -> None:
        direction = KnownDirection(
            category=Category.ARCHITECTURE,
            title="Test",
            description="",
            risk=RiskLevel.LOW,
            expected_range=(0.3, 0.5),
            keywords=(),
        )
        stats = {
            cat: CategoryStats(cat, 5, 4, 1, 0, 1.0) if cat == Category.ARCHITECTURE
            else CategoryStats(cat, 0, 0, 0, 0, 0.0)
            for cat in Category
        }
        score_exploit = _compute_priority_score(direction, stats, Strategy.EXPLOIT, 10)
        score_explore = _compute_priority_score(direction, stats, Strategy.EXPLORE, 10)
        assert score_exploit > score_explore

    def test_risk_adjustment(self) -> None:
        low_risk = KnownDirection(Category.OTHER, "A", "", RiskLevel.LOW, (0.5, 1.0), ())
        high_risk = KnownDirection(Category.OTHER, "B", "", RiskLevel.HIGH, (0.5, 1.0), ())
        stats = self._make_stats()
        score_low = _compute_priority_score(low_risk, stats, Strategy.AUTO, 5)
        score_high = _compute_priority_score(high_risk, stats, Strategy.AUTO, 5)
        assert score_low > score_high

    def test_exploit_penalizes_tried_and_failed_categories(self) -> None:
        stats = _empty_stats()
        stats[Category.OPTIMIZER] = CategoryStats(
            category=Category.OPTIMIZER, total=2, keeps=0, discards=2, crashes=0, avg_improvement_pct=0.0
        )
        direction = self._make_direction(category=Category.OPTIMIZER)
        score = _compute_priority_score(direction, stats, Strategy.EXPLOIT, total_experiments=5)
        # Should get 0.3x penalty for tried-and-failed
        midpoint = (direction.expected_range[0] + direction.expected_range[1]) / 2.0
        base = midpoint * 1.0  # MEDIUM risk
        expected = base * 0.3
        assert score == pytest.approx(expected)

    def test_explore_gives_failed_categories_a_second_look(self) -> None:
        stats = _empty_stats()
        stats[Category.OPTIMIZER] = CategoryStats(
            category=Category.OPTIMIZER, total=2, keeps=0, discards=2, crashes=0, avg_improvement_pct=0.0
        )
        direction = self._make_direction(category=Category.OPTIMIZER)
        score = _compute_priority_score(direction, stats, Strategy.EXPLORE, total_experiments=5)
        midpoint = (direction.expected_range[0] + direction.expected_range[1]) / 2.0
        base = midpoint * 1.0
        expected = base * 0.8
        assert score == pytest.approx(expected)

    def test_auto_mode_shifts_from_explore_to_exploit_with_more_experiments(self) -> None:
        """AUTO strategy should favor exploration early and exploitation later."""
        stats = _empty_stats()
        direction = self._make_direction(category=Category.EMBEDDING)
        score_early = _compute_priority_score(direction, stats, Strategy.AUTO, total_experiments=2)
        score_late = _compute_priority_score(direction, stats, Strategy.AUTO, total_experiments=18)
        assert score_early > score_late

    def test_auto_mode_with_successful_category_gets_bonus(self) -> None:
        stats = _empty_stats()
        stats[Category.ARCHITECTURE] = CategoryStats(
            category=Category.ARCHITECTURE, total=5, keeps=4, discards=1, crashes=0, avg_improvement_pct=3.0
        )
        direction = self._make_direction(category=Category.ARCHITECTURE)
        score = _compute_priority_score(direction, stats, Strategy.AUTO, total_experiments=10)
        # exploitation_weight = 1.0 - max(0.2, 1.0 - 10/20) = 0.5
        # success_rate = 80%, bonus = 0.5 * 0.8 = 0.4, so *= 1.4
        midpoint = (direction.expected_range[0] + direction.expected_range[1]) / 2.0
        base = midpoint * 1.0
        expected = base * 1.4
        assert score == pytest.approx(expected)

    def test_auto_mode_failed_category_penalized(self) -> None:
        stats = _empty_stats()
        stats[Category.REGULARIZATION] = CategoryStats(
            category=Category.REGULARIZATION, total=3, keeps=0, discards=3, crashes=0, avg_improvement_pct=0.0
        )
        direction = self._make_direction(category=Category.REGULARIZATION)
        score = _compute_priority_score(direction, stats, Strategy.AUTO, total_experiments=5)
        midpoint = (direction.expected_range[0] + direction.expected_range[1]) / 2.0
        base = midpoint * 1.0
        expected = base * 0.5
        assert score == pytest.approx(expected)

    def test_exploit_success_rate_affects_boost(self) -> None:
        """Higher success rates should produce higher exploit scores."""
        direction = self._make_direction(category=Category.ARCHITECTURE)
        stats_high = _empty_stats()
        stats_high[Category.ARCHITECTURE] = CategoryStats(
            category=Category.ARCHITECTURE, total=10, keeps=9, discards=1, crashes=0, avg_improvement_pct=5.0
        )
        stats_low = _empty_stats()
        stats_low[Category.ARCHITECTURE] = CategoryStats(
            category=Category.ARCHITECTURE, total=10, keeps=2, discards=8, crashes=0, avg_improvement_pct=1.0
        )
        score_high = _compute_priority_score(direction, stats_high, Strategy.EXPLOIT, total_experiments=10)
        score_low = _compute_priority_score(direction, stats_low, Strategy.EXPLOIT, total_experiments=10)
        assert score_high > score_low


# ===================================================================
# 6. Suggestion generation
# ===================================================================


class TestGenerateSuggestions:
    """generate_suggestions produces ranked, deduplicated suggestions."""

    def test_returns_requested_count(self) -> None:
        stats = {cat: CategoryStats(cat, 0, 0, 0, 0, 0.0) for cat in Category}
        suggestions = generate_suggestions([], stats, Strategy.AUTO, 3)
        assert len(suggestions) == 3

    def test_suggestions_are_ranked_by_priority_descending(self) -> None:
        stats = {cat: CategoryStats(cat, 0, 0, 0, 0, 0.0) for cat in Category}
        suggestions = generate_suggestions([], stats, Strategy.AUTO, 10)
        for i in range(len(suggestions) - 1):
            assert suggestions[i].priority_score >= suggestions[i + 1].priority_score

    def test_suggestion_ranks_are_sequential(self) -> None:
        stats = _empty_stats()
        suggestions = generate_suggestions([], stats, Strategy.AUTO, num_suggestions=5)
        assert [s.rank for s in suggestions] == [1, 2, 3, 4, 5]

    def test_already_tried_directions_are_excluded(self) -> None:
        """Experiments matching a direction's keywords should prevent that direction from appearing."""
        exps = [_exp("Changed ns_steps from 5 to 6 in newton update")]
        stats = _empty_stats()
        suggestions = generate_suggestions(exps, stats, Strategy.EXPLORE, num_suggestions=20)
        titles = [s.title for s in suggestions]
        assert "Tune Muon orthogonalization steps" not in titles

    def test_explore_mode_all_untried_are_explore_kind(self) -> None:
        stats = {cat: CategoryStats(cat, 0, 0, 0, 0, 0.0) for cat in Category}
        suggestions = generate_suggestions([], stats, Strategy.EXPLORE, 5)
        assert all(s.kind == SuggestionKind.EXPLORE for s in suggestions)

    def test_exploit_kind_when_category_has_successes(self) -> None:
        stats = _empty_stats()
        stats[Category.ARCHITECTURE] = CategoryStats(
            category=Category.ARCHITECTURE, total=3, keeps=2, discards=1, crashes=0, avg_improvement_pct=5.0
        )
        suggestions = generate_suggestions([], stats, Strategy.EXPLOIT, num_suggestions=20)
        arch_suggestions = [s for s in suggestions if s.category == Category.ARCHITECTURE]
        for s in arch_suggestions:
            assert s.kind == SuggestionKind.EXPLOIT

    def test_all_suggestions_have_valid_fields(self) -> None:
        stats = _empty_stats()
        suggestions = generate_suggestions([], stats, Strategy.AUTO, num_suggestions=5)
        for s in suggestions:
            assert isinstance(s.rank, int) and s.rank >= 1
            assert isinstance(s.kind, SuggestionKind)
            assert len(s.title) > 0
            assert isinstance(s.category, Category)
            assert isinstance(s.risk, RiskLevel)
            assert len(s.expected_range) == 2
            assert s.expected_range[0] <= s.expected_range[1]
            assert len(s.reasoning) > 0
            assert s.priority_score > 0

    def test_exploit_strategy_ranks_successful_categories_higher(self) -> None:
        """In exploit mode, directions from a successful category should rank above untried ones."""
        stats = _empty_stats()
        stats[Category.HYPERPARAMS] = CategoryStats(
            category=Category.HYPERPARAMS, total=5, keeps=4, discards=1, crashes=0, avg_improvement_pct=8.0
        )
        suggestions = generate_suggestions([], stats, Strategy.EXPLOIT, num_suggestions=10)
        first_hp_rank = next(
            (s.rank for s in suggestions if s.category == Category.HYPERPARAMS), None
        )
        first_other_rank = next(
            (s.rank for s in suggestions if s.category != Category.HYPERPARAMS), None
        )
        if first_hp_rank is not None and first_other_rank is not None:
            assert first_hp_rank < first_other_rank

    def test_explore_strategy_ranks_untried_categories_higher(self) -> None:
        """In explore mode, directions from untried categories should rank above tried ones."""
        stats = _empty_stats()
        stats[Category.HYPERPARAMS] = CategoryStats(
            category=Category.HYPERPARAMS, total=5, keeps=4, discards=1, crashes=0, avg_improvement_pct=8.0
        )
        suggestions = generate_suggestions([], stats, Strategy.EXPLORE, num_suggestions=10)
        first_hp_rank = next(
            (s.rank for s in suggestions if s.category == Category.HYPERPARAMS), None
        )
        first_untried_rank = next(
            (s.rank for s in suggestions if stats[s.category].total == 0), None
        )
        if first_hp_rank is not None and first_untried_rank is not None:
            assert first_untried_rank < first_hp_rank

    def test_zero_suggestions_requested_returns_empty(self) -> None:
        stats = _empty_stats()
        suggestions = generate_suggestions([], stats, Strategy.AUTO, num_suggestions=0)
        assert suggestions == []

    def test_more_suggestions_than_available_returns_all_available(self) -> None:
        stats = _empty_stats()
        suggestions = generate_suggestions([], stats, Strategy.AUTO, num_suggestions=100)
        assert len(suggestions) <= len(KNOWN_DIRECTIONS)
        assert len(suggestions) > 0


# ===================================================================
# 7. CLI
# ===================================================================


class TestCLI:
    """Test the Click CLI via CliRunner for realistic end-to-end behavior."""

    def test_basic_invocation_with_valid_tsv(self, tmp_path: Path) -> None:
        runner = CliRunner()
        tsv = tmp_path / "results.tsv"
        tsv.write_text(VALID_TSV)
        result = runner.invoke(cli, ["--results", str(tsv), "--repo-dir", str(tmp_path), "--no-color"])
        assert result.exit_code == 0
        assert "suggestions" in result.output.lower()

    def test_json_format_produces_valid_json(self, tmp_path: Path) -> None:
        runner = CliRunner()
        tsv = tmp_path / "results.tsv"
        tsv.write_text(VALID_TSV)
        result = runner.invoke(cli, [
            "--results", str(tsv),
            "--repo-dir", str(tmp_path),
            "--format", "json",
            "--no-color",
        ])
        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert "strategy" in data
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)

    def test_json_format_has_expected_suggestion_fields(self, tmp_path: Path) -> None:
        runner = CliRunner()
        tsv = tmp_path / "results.tsv"
        tsv.write_text(VALID_TSV)
        result = runner.invoke(cli, [
            "--results", str(tsv),
            "--repo-dir", str(tmp_path),
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = _extract_json(result.output)
        if data["suggestions"]:
            s = data["suggestions"][0]
            assert "rank" in s
            assert "kind" in s
            assert "title" in s
            assert "category" in s
            assert "risk" in s
            assert "expected_range_pct" in s
            assert "reasoning" in s
            assert "priority_score" in s

    def test_quiet_mode_produces_minimal_output(self, tmp_path: Path) -> None:
        runner = CliRunner()
        tsv = tmp_path / "results.tsv"
        tsv.write_text(VALID_TSV)
        result = runner.invoke(cli, [
            "--results", str(tsv),
            "--repo-dir", str(tmp_path),
            "--quiet",
            "--no-color",
        ])
        assert result.exit_code == 0
        # Filter out git warning lines, then check suggestion lines
        lines = [line for line in result.output.strip().split("\n") if line and not line.startswith("Note:")]
        for line in lines:
            assert line[0].isdigit(), f"Expected line to start with rank number: {line!r}"

    def test_strategy_explore_accepted(self, tmp_path: Path) -> None:
        runner = CliRunner()
        tsv = tmp_path / "results.tsv"
        tsv.write_text(VALID_TSV)
        result = runner.invoke(cli, [
            "--results", str(tsv),
            "--repo-dir", str(tmp_path),
            "--strategy", "explore",
            "--format", "json",
            "--no-color",
        ])
        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert "explore" in data["strategy"].lower()

    def test_strategy_exploit_accepted(self, tmp_path: Path) -> None:
        runner = CliRunner()
        tsv = tmp_path / "results.tsv"
        tsv.write_text(VALID_TSV)
        result = runner.invoke(cli, [
            "--results", str(tsv),
            "--repo-dir", str(tmp_path),
            "--strategy", "exploit",
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert "exploit" in data["strategy"].lower()

    def test_missing_results_file_exits_with_code_1(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, [
            "--results", str(tmp_path / "nonexistent.tsv"),
            "--repo-dir", str(tmp_path),
            "--no-color",
        ])
        assert result.exit_code == 1

    def test_missing_results_file_still_produces_suggestions(self, tmp_path: Path) -> None:
        """Even when the results file is missing, the CLI should still output suggestions."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "--results", str(tmp_path / "nonexistent.tsv"),
            "--repo-dir", str(tmp_path),
            "--format", "json",
        ])
        data = _extract_json(result.output)
        assert len(data["suggestions"]) > 0

    def test_num_suggestions_option(self, tmp_path: Path) -> None:
        runner = CliRunner()
        tsv = tmp_path / "results.tsv"
        tsv.write_text(VALID_TSV)
        result = runner.invoke(cli, [
            "--results", str(tsv),
            "--repo-dir", str(tmp_path),
            "--num-suggestions", "3",
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert len(data["suggestions"]) == 3

    def test_json_category_stats_present(self, tmp_path: Path) -> None:
        runner = CliRunner()
        tsv = tmp_path / "results.tsv"
        tsv.write_text(VALID_TSV)
        result = runner.invoke(cli, [
            "--results", str(tsv),
            "--repo-dir", str(tmp_path),
            "--format", "json",
        ])
        data = _extract_json(result.output)
        assert "category_stats" in data
        assert len(data["category_stats"]) == len(Category)

    def test_version_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output

    def test_no_color_flag_suppresses_ansi(self, tmp_path: Path) -> None:
        runner = CliRunner()
        tsv = tmp_path / "results.tsv"
        tsv.write_text(VALID_TSV)
        result = runner.invoke(cli, [
            "--results", str(tsv),
            "--repo-dir", str(tmp_path),
            "--no-color",
        ])
        assert result.exit_code == 0
        assert "\x1b[" not in result.output

    def test_quiet_with_json_format_json_wins(self, tmp_path: Path) -> None:
        """When both --quiet and --format json are given, JSON format takes precedence (checked first in CLI)."""
        runner = CliRunner()
        tsv = tmp_path / "results.tsv"
        tsv.write_text(VALID_TSV)
        result = runner.invoke(cli, [
            "--results", str(tsv),
            "--repo-dir", str(tmp_path),
            "--quiet",
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = _extract_json(result.output)
        assert "suggestions" in data

    def test_baseline_and_best_bpb_in_json(self, tmp_path: Path) -> None:
        runner = CliRunner()
        tsv = tmp_path / "results.tsv"
        _write_tsv(tsv, [
            ["abc1234", "1.050000", "45.2", "keep", "Baseline: default config"],
            ["def5678", "1.030000", "45.5", "keep", "Better model"],
        ])
        result = runner.invoke(cli, [
            "--results", str(tsv),
            "--repo-dir", str(tmp_path),
            "--format", "json",
        ])
        data = _extract_json(result.output)
        assert data["baseline_bpb"] is not None
        assert data["best_bpb"] is not None
        assert data["best_bpb"] <= data["baseline_bpb"]

    def test_text_output_mentions_strategy(self, tmp_path: Path) -> None:
        runner = CliRunner()
        tsv = tmp_path / "results.tsv"
        tsv.write_text(VALID_TSV)
        result = runner.invoke(cli, [
            "--results", str(tsv),
            "--repo-dir", str(tmp_path),
            "--strategy", "explore",
            "--no-color",
        ])
        assert result.exit_code == 0
        assert "explore" in result.output.lower()


# ===================================================================
# 8. Edge cases and integration
# ===================================================================


class TestEdgeCases:
    """Miscellaneous edge cases that span multiple functions."""

    def test_experiment_dataclass_is_frozen(self) -> None:
        exp = _exp()
        with pytest.raises(AttributeError):
            exp.commit = "changed"  # type: ignore[misc]

    def test_category_stats_is_frozen(self) -> None:
        cs = CategoryStats(Category.OTHER, 0, 0, 0, 0, 0.0)
        with pytest.raises(AttributeError):
            cs.total = 5  # type: ignore[misc]

    def test_suggestion_is_frozen(self) -> None:
        s = Suggestion(
            rank=1,
            kind=SuggestionKind.EXPLORE,
            title="Test",
            category=Category.OTHER,
            risk=RiskLevel.LOW,
            expected_range=(0.1, 0.3),
            reasoning="Because",
            priority_score=1.0,
        )
        with pytest.raises(AttributeError):
            s.rank = 2  # type: ignore[misc]

    def test_parse_result_is_frozen(self) -> None:
        pr = ParseResult(ok=True, experiments=[])
        with pytest.raises(AttributeError):
            pr.ok = False  # type: ignore[misc]

    def test_all_known_directions_have_valid_fields(self) -> None:
        """Sanity check the knowledge base."""
        for d in KNOWN_DIRECTIONS:
            assert isinstance(d.category, Category)
            assert isinstance(d.risk, RiskLevel)
            assert len(d.title) > 0
            assert len(d.description) > 0
            assert len(d.keywords) > 0
            assert d.expected_range[0] <= d.expected_range[1]

    def test_full_pipeline_with_sample_data(self, tmp_path: Path) -> None:
        """End-to-end: parse real-ish TSV, compute stats, generate suggestions."""
        tsv = tmp_path / "results.tsv"
        _write_tsv(tsv, [
            ["base000", "1.050000", "45.0", "keep", "Baseline: default config"],
            ["aaa1111", "1.040000", "45.2", "keep", "Increase depth to 10"],
            ["bbb2222", "1.060000", "46.0", "discard", "Try larger batch size"],
            ["ccc3333", "1.035000", "44.8", "keep", "Tune matrix LR to 0.05"],
            ["ddd4444", "0.000000", "44.0", "crash", "OOM with big model"],
        ])
        parsed = parse_results_tsv(tsv)
        assert parsed.ok is True
        assert len(parsed.experiments) == 5

        baseline_bpb = parsed.experiments[0].val_bpb
        enriched = []
        for exp in parsed.experiments:
            cat = classify_experiment(exp.description, exp.diff_text)
            enriched.append(Experiment(
                commit=exp.commit,
                val_bpb=exp.val_bpb,
                memory_gb=exp.memory_gb,
                status=exp.status,
                description=exp.description,
                category=cat,
                diff_text=exp.diff_text,
            ))

        stats = compute_category_stats(enriched, baseline_bpb)
        suggestions = generate_suggestions(enriched, stats, Strategy.AUTO, num_suggestions=5)

        assert len(suggestions) == 5
        assert suggestions[0].rank == 1
        assert suggestions[0].priority_score >= suggestions[-1].priority_score

    def test_strategy_enum_values_match_cli_choices(self) -> None:
        """Strategy enum values must match the CLI's click.Choice options."""
        assert Strategy.AUTO.value == "auto"
        assert Strategy.EXPLORE.value == "explore"
        assert Strategy.EXPLOIT.value == "exploit"

    def test_category_enum_completeness(self) -> None:
        """Verify all expected categories exist."""
        expected = {"architecture", "hyperparams", "optimizer", "regularization",
                    "activation", "embedding", "efficiency", "other"}
        actual = {c.value for c in Category}
        assert actual == expected

    def test_status_enum_completeness(self) -> None:
        expected = {"keep", "discard", "crash"}
        actual = {s.value for s in Status}
        assert actual == expected
