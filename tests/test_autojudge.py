"""Comprehensive behavior-focused tests for auto_judge.

Tests are organized by category:
1. Parsing (TSV input handling)
2. Noise floor estimation
3. Streak computation
4. Pareto frontier
5. Verdict computation
6. Full analysis pipeline
7. Output formatting (JSON and quiet)
8. CLI integration via Click's CliRunner
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Optional

import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "auto-judge"))

from auto_judge import (
    Err,
    Experiment,
    JudgmentResult,
    Ok,
    ParetoPoint,
    ParseError,
    Status,
    Verdict,
    analyze,
    cli,
    compute_pareto_frontier,
    compute_streaks,
    compute_verdict,
    estimate_noise_floor,
    format_json,
    format_quiet,
    parse_results_tsv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_experiment(
    index: int = 1,
    commit: str = "abc1234",
    val_bpb: float = 1.05,
    memory_gb: float = 45.0,
    status: Status = Status.KEEP,
    description: str = "test experiment",
) -> Experiment:
    """Build a single Experiment with sensible defaults."""
    return Experiment(
        index=index,
        commit=commit,
        val_bpb=val_bpb,
        memory_gb=memory_gb,
        status=status,
        description=description,
    )


def make_keep_series(
    bpb_values: list,
    memory_gb: float = 45.0,
    start_index: int = 1,
) -> list:
    """Build a series of KEEP experiments from BPB values."""
    return [
        make_experiment(
            index=start_index + i,
            commit=f"commit_{start_index + i:03d}",
            val_bpb=bpb,
            memory_gb=memory_gb,
            status=Status.KEEP,
            description=f"experiment {start_index + i}",
        )
        for i, bpb in enumerate(bpb_values)
    ]


def write_tsv(path: Path, rows: list, header: Optional[list] = None) -> None:
    """Write a TSV file from rows, with optional custom header."""
    if header is None:
        header = ["commit", "val_bpb", "memory_gb", "status", "description"]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(row))
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Parsing
# ---------------------------------------------------------------------------


class TestParseResultsTsv:
    """Tests for parse_results_tsv: valid data, missing columns, empty file, etc."""

    def test_valid_tsv_returns_ok_with_experiments(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        write_tsv(tsv, [
            ["abc1234", "1.050000", "45.2", "keep", "Baseline config"],
            ["def5678", "1.040000", "45.5", "keep", "Increase depth"],
        ])

        result = parse_results_tsv(tsv)

        assert isinstance(result, Ok)
        experiments = result.value
        assert len(experiments) == 2
        assert experiments[0].index == 1
        assert experiments[0].commit == "abc1234"
        assert experiments[0].val_bpb == pytest.approx(1.05)
        assert experiments[0].memory_gb == pytest.approx(45.2)
        assert experiments[0].status == Status.KEEP
        assert experiments[0].description == "Baseline config"
        assert experiments[1].index == 2

    def test_valid_tsv_all_statuses_parsed(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        write_tsv(tsv, [
            ["aaa", "1.05", "45.0", "keep", "kept"],
            ["bbb", "1.06", "45.0", "discard", "discarded"],
            ["ccc", "0.00", "0.0", "crash", "crashed"],
        ])

        result = parse_results_tsv(tsv)

        assert isinstance(result, Ok)
        assert result.value[0].status == Status.KEEP
        assert result.value[1].status == Status.DISCARD
        assert result.value[2].status == Status.CRASH

    def test_missing_file_returns_err(self, tmp_path: Path) -> None:
        result = parse_results_tsv(tmp_path / "nonexistent.tsv")

        assert isinstance(result, Err)
        assert "not found" in result.error.message.lower()

    def test_empty_file_returns_err(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        tsv.write_text("", encoding="utf-8")

        result = parse_results_tsv(tsv)

        assert isinstance(result, Err)
        assert "empty" in result.error.message.lower()

    def test_header_only_returns_err(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        tsv.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n", encoding="utf-8")

        result = parse_results_tsv(tsv)

        assert isinstance(result, Err)
        assert "no data rows" in result.error.message.lower()

    def test_missing_columns_returns_err(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        tsv.write_text("commit\tval_bpb\n" "abc\t1.05\n", encoding="utf-8")

        result = parse_results_tsv(tsv)

        assert isinstance(result, Err)
        assert "missing columns" in result.error.message.lower()

    def test_malformed_val_bpb_returns_err(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        write_tsv(tsv, [
            ["abc", "not_a_number", "45.0", "keep", "bad bpb"],
        ])

        result = parse_results_tsv(tsv)

        assert isinstance(result, Err)
        assert "val_bpb" in result.error.message.lower()

    def test_malformed_memory_gb_returns_err(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        write_tsv(tsv, [
            ["abc", "1.05", "nope", "keep", "bad memory"],
        ])

        result = parse_results_tsv(tsv)

        assert isinstance(result, Err)
        assert "memory_gb" in result.error.message.lower()

    def test_invalid_status_returns_err(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        write_tsv(tsv, [
            ["abc", "1.05", "45.0", "unknown_status", "bad status"],
        ])

        result = parse_results_tsv(tsv)

        assert isinstance(result, Err)
        assert "invalid status" in result.error.message.lower()

    def test_too_few_fields_in_row_returns_err(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        tsv.write_text(
            "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n" "abc\t1.05\n",
            encoding="utf-8",
        )

        result = parse_results_tsv(tsv)

        assert isinstance(result, Err)
        # Should mention the line or field count problem
        assert isinstance(result.error, ParseError)

    def test_directory_path_returns_err(self, tmp_path: Path) -> None:
        result = parse_results_tsv(tmp_path)

        assert isinstance(result, Err)
        assert "not a file" in result.error.message.lower()

    def test_column_order_does_not_matter(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        tsv.write_text(
            "description\tstatus\tmemory_gb\tval_bpb\tcommit\n"
            "Baseline\tkeep\t45.2\t1.05\tabc1234\n",
            encoding="utf-8",
        )

        result = parse_results_tsv(tsv)

        assert isinstance(result, Ok)
        exp = result.value[0]
        assert exp.commit == "abc1234"
        assert exp.val_bpb == pytest.approx(1.05)
        assert exp.memory_gb == pytest.approx(45.2)
        assert exp.status == Status.KEEP
        assert exp.description == "Baseline"

    def test_parse_error_includes_source_path(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        tsv.write_text("", encoding="utf-8")

        result = parse_results_tsv(tsv)

        assert isinstance(result, Err)
        assert result.error.source == str(tsv)

    def test_experiments_indexed_sequentially_starting_at_one(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        write_tsv(tsv, [
            ["a", "1.05", "45.0", "keep", "first"],
            ["b", "1.04", "45.0", "keep", "second"],
            ["c", "1.03", "45.0", "keep", "third"],
        ])

        result = parse_results_tsv(tsv)

        assert isinstance(result, Ok)
        indices = [e.index for e in result.value]
        assert indices == [1, 2, 3]

    def test_whitespace_in_fields_stripped(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        tsv.write_text(
            "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
            "  abc1234 \t 1.050 \t 45.2 \t keep \t Baseline config \n",
            encoding="utf-8",
        )

        result = parse_results_tsv(tsv)

        assert isinstance(result, Ok)
        exp = result.value[0]
        assert exp.commit == "abc1234"
        assert exp.val_bpb == pytest.approx(1.05)
        assert exp.status == Status.KEEP

    def test_case_insensitive_status(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        write_tsv(tsv, [
            ["abc", "1.05", "45.0", "KEEP", "upper case status"],
        ])

        result = parse_results_tsv(tsv)

        assert isinstance(result, Ok)
        assert result.value[0].status == Status.KEEP

    def test_case_insensitive_header(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        tsv.write_text(
            "Commit\tVal_BPB\tMemory_GB\tStatus\tDescription\n"
            "abc\t1.05\t45.0\tkeep\ttest\n",
            encoding="utf-8",
        )

        result = parse_results_tsv(tsv)

        assert isinstance(result, Ok)


# ---------------------------------------------------------------------------
# 2. Noise floor estimation
# ---------------------------------------------------------------------------


class TestEstimateNoiseFloor:
    """Tests for estimate_noise_floor: sufficient data, insufficient data, window."""

    def test_returns_none_with_fewer_than_three_keeps(self) -> None:
        experiments = make_keep_series([1.05, 1.04])
        assert estimate_noise_floor(experiments) is None

    def test_returns_none_with_zero_keeps(self) -> None:
        assert estimate_noise_floor([]) is None

    def test_returns_none_with_no_keep_experiments(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.05, status=Status.DISCARD),
            make_experiment(index=2, val_bpb=1.06, status=Status.DISCARD),
        ]
        assert estimate_noise_floor(experiments) is None

    def test_returns_float_with_three_or_more_keeps(self) -> None:
        experiments = make_keep_series([1.050, 1.045, 1.040, 1.035])
        result = estimate_noise_floor(experiments)
        assert result is not None
        assert isinstance(result, float)
        assert result >= 0.0

    def test_constant_diffs_yield_zero_noise_floor(self) -> None:
        # Constant differences: diffs are all -0.005, variance = 0 => noise = 0
        experiments = make_keep_series([1.050, 1.045, 1.040, 1.035])
        result = estimate_noise_floor(experiments)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_noise_floor_increases_with_variance(self) -> None:
        low_var = make_keep_series([1.050, 1.045, 1.040, 1.035])
        high_var = make_keep_series([1.050, 1.040, 1.048, 1.030])

        low_noise = estimate_noise_floor(low_var)
        high_noise = estimate_noise_floor(high_var)

        assert low_noise is not None
        assert high_noise is not None
        assert high_noise > low_noise

    def test_only_keep_experiments_contribute(self) -> None:
        keeps = make_keep_series([1.050, 1.045, 1.040, 1.035])
        discards = [
            make_experiment(index=5, val_bpb=1.10, status=Status.DISCARD),
            make_experiment(index=6, val_bpb=1.20, status=Status.DISCARD),
        ]
        mixed = keeps + discards

        keeps_only_floor = estimate_noise_floor(keeps)
        mixed_floor = estimate_noise_floor(mixed)

        # Discards are ignored, so floors should be identical
        assert keeps_only_floor == pytest.approx(mixed_floor)

    def test_window_parameter_limits_recent_keeps(self) -> None:
        experiments = make_keep_series([
            1.100, 1.090, 1.080, 1.070, 1.060,
            1.050, 1.045, 1.040, 1.035, 1.030,
        ])

        full_noise = estimate_noise_floor(experiments, window=10)
        recent_noise = estimate_noise_floor(experiments, window=3)

        assert full_noise is not None
        assert recent_noise is not None

    def test_excludes_zero_bpb_keeps(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=0.0, status=Status.KEEP),
            make_experiment(index=2, val_bpb=0.0, status=Status.KEEP),
            make_experiment(index=3, val_bpb=1.05, status=Status.KEEP),
        ]
        # Only 1 keep with val_bpb > 0, so insufficient data
        result = estimate_noise_floor(experiments)
        assert result is None

    def test_noise_floor_is_standard_deviation_of_diffs(self) -> None:
        # Manual calculation: keeps = [1.0, 1.1, 1.05, 1.15]
        # diffs = [0.1, -0.05, 0.1]
        # mean = 0.05, variance = ((0.05)^2 + (-0.1)^2 + (0.05)^2) / 2
        # = (0.0025 + 0.01 + 0.0025) / 2 = 0.0075
        # std = sqrt(0.0075) ~ 0.08660
        experiments = make_keep_series([1.0, 1.1, 1.05, 1.15])
        result = estimate_noise_floor(experiments)
        assert result is not None
        expected = math.sqrt(0.0075)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_exactly_three_keeps_with_two_diffs(self) -> None:
        # Minimum case: 3 keeps -> 2 diffs -> can compute std with n-1=1
        experiments = make_keep_series([1.00, 1.05, 0.98])
        result = estimate_noise_floor(experiments)
        assert result is not None
        assert result > 0.0


# ---------------------------------------------------------------------------
# 3. Streaks
# ---------------------------------------------------------------------------


class TestComputeStreaks:
    """Tests for compute_streaks: trailing streaks, longest streaks, empty list."""

    def test_empty_list_returns_all_zeros(self) -> None:
        result = compute_streaks([])
        assert result["discard_streak"] == 0
        assert result["crash_streak"] == 0
        assert result["keep_streak"] == 0
        assert result["longest_keep_streak"] == 0
        assert result["longest_discard_streak"] == 0

    def test_trailing_keep_streak(self) -> None:
        experiments = [
            make_experiment(index=1, status=Status.DISCARD),
            make_experiment(index=2, status=Status.KEEP),
            make_experiment(index=3, status=Status.KEEP),
            make_experiment(index=4, status=Status.KEEP),
        ]
        result = compute_streaks(experiments)
        assert result["keep_streak"] == 3
        assert result["discard_streak"] == 0
        assert result["crash_streak"] == 0

    def test_trailing_discard_streak(self) -> None:
        experiments = [
            make_experiment(index=1, status=Status.KEEP),
            make_experiment(index=2, status=Status.DISCARD),
            make_experiment(index=3, status=Status.DISCARD),
        ]
        result = compute_streaks(experiments)
        assert result["discard_streak"] == 2
        assert result["keep_streak"] == 0

    def test_trailing_crash_streak(self) -> None:
        experiments = [
            make_experiment(index=1, status=Status.KEEP),
            make_experiment(index=2, status=Status.CRASH, val_bpb=0.0),
            make_experiment(index=3, status=Status.CRASH, val_bpb=0.0),
        ]
        result = compute_streaks(experiments)
        assert result["crash_streak"] == 2
        assert result["keep_streak"] == 0
        assert result["discard_streak"] == 0

    def test_longest_keep_streak_tracked_across_history(self) -> None:
        experiments = [
            make_experiment(index=1, status=Status.KEEP),
            make_experiment(index=2, status=Status.KEEP),
            make_experiment(index=3, status=Status.KEEP),
            make_experiment(index=4, status=Status.KEEP),
            make_experiment(index=5, status=Status.DISCARD),
            make_experiment(index=6, status=Status.KEEP),
            make_experiment(index=7, status=Status.KEEP),
        ]
        result = compute_streaks(experiments)
        assert result["longest_keep_streak"] == 4
        assert result["keep_streak"] == 2  # trailing

    def test_longest_discard_streak_tracked_across_history(self) -> None:
        experiments = [
            make_experiment(index=1, status=Status.DISCARD),
            make_experiment(index=2, status=Status.DISCARD),
            make_experiment(index=3, status=Status.DISCARD),
            make_experiment(index=4, status=Status.KEEP),
            make_experiment(index=5, status=Status.DISCARD),
        ]
        result = compute_streaks(experiments)
        assert result["longest_discard_streak"] == 3
        assert result["discard_streak"] == 1  # trailing

    def test_single_experiment_creates_streak_of_one(self) -> None:
        experiments = [make_experiment(index=1, status=Status.KEEP)]
        result = compute_streaks(experiments)
        assert result["keep_streak"] == 1
        assert result["longest_keep_streak"] == 1

    def test_crash_breaks_longest_keep_streak(self) -> None:
        experiments = [
            make_experiment(index=1, status=Status.KEEP),
            make_experiment(index=2, status=Status.KEEP),
            make_experiment(index=3, status=Status.CRASH, val_bpb=0.0),
            make_experiment(index=4, status=Status.KEEP),
        ]
        result = compute_streaks(experiments)
        assert result["longest_keep_streak"] == 2
        assert result["keep_streak"] == 1

    def test_mixed_statuses_trailing_streak_is_last_contiguous(self) -> None:
        experiments = [
            make_experiment(index=1, status=Status.KEEP),
            make_experiment(index=2, status=Status.DISCARD),
            make_experiment(index=3, status=Status.KEEP),
        ]
        result = compute_streaks(experiments)
        assert result["keep_streak"] == 1
        assert result["discard_streak"] == 0

    def test_all_same_status(self) -> None:
        experiments = [
            make_experiment(index=i, status=Status.DISCARD)
            for i in range(1, 6)
        ]
        result = compute_streaks(experiments)
        assert result["discard_streak"] == 5
        assert result["longest_discard_streak"] == 5
        assert result["keep_streak"] == 0


# ---------------------------------------------------------------------------
# 4. Pareto frontier
# ---------------------------------------------------------------------------


class TestComputeParetoFrontier:
    """Tests for compute_pareto_frontier: domination, single point, no keeps."""

    def test_empty_list_returns_empty_frontier(self) -> None:
        assert compute_pareto_frontier([]) == []

    def test_single_keep_is_on_frontier(self) -> None:
        experiments = [make_experiment(index=1, val_bpb=1.05, memory_gb=45.0)]
        frontier = compute_pareto_frontier(experiments)
        assert len(frontier) == 1
        assert frontier[0].index == 1

    def test_dominated_point_excluded(self) -> None:
        # Point A: bpb=1.04, mem=44 (dominates B)
        # Point B: bpb=1.05, mem=45 (dominated: worse on both axes)
        experiments = [
            make_experiment(index=1, val_bpb=1.04, memory_gb=44.0),
            make_experiment(index=2, val_bpb=1.05, memory_gb=45.0),
        ]
        frontier = compute_pareto_frontier(experiments)
        assert len(frontier) == 1
        assert frontier[0].index == 1

    def test_non_dominated_points_both_on_frontier(self) -> None:
        # Point A: bpb=1.02, mem=50 (better bpb, worse memory)
        # Point B: bpb=1.05, mem=40 (worse bpb, better memory)
        experiments = [
            make_experiment(index=1, val_bpb=1.02, memory_gb=50.0),
            make_experiment(index=2, val_bpb=1.05, memory_gb=40.0),
        ]
        frontier = compute_pareto_frontier(experiments)
        assert len(frontier) == 2

    def test_discards_and_crashes_excluded(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.05, memory_gb=45.0, status=Status.KEEP),
            make_experiment(index=2, val_bpb=1.00, memory_gb=40.0, status=Status.DISCARD),
            make_experiment(index=3, val_bpb=0.0, memory_gb=0.0, status=Status.CRASH),
        ]
        frontier = compute_pareto_frontier(experiments)
        assert len(frontier) == 1
        assert frontier[0].index == 1

    def test_zero_bpb_keeps_excluded(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=0.0, memory_gb=45.0, status=Status.KEEP),
            make_experiment(index=2, val_bpb=1.05, memory_gb=45.0, status=Status.KEEP),
        ]
        frontier = compute_pareto_frontier(experiments)
        assert len(frontier) == 1
        assert frontier[0].index == 2

    def test_frontier_returns_pareto_point_type(self) -> None:
        experiments = [make_experiment(index=1, val_bpb=1.05, memory_gb=45.0)]
        frontier = compute_pareto_frontier(experiments)
        assert isinstance(frontier[0], ParetoPoint)
        assert frontier[0].val_bpb == pytest.approx(1.05)
        assert frontier[0].memory_gb == pytest.approx(45.0)

    def test_multiple_points_on_frontier_with_tradeoffs(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.00, memory_gb=60.0),
            make_experiment(index=2, val_bpb=1.02, memory_gb=50.0),
            make_experiment(index=3, val_bpb=1.05, memory_gb=40.0),
            make_experiment(index=4, val_bpb=1.03, memory_gb=55.0),  # dominated by idx 2
        ]
        frontier = compute_pareto_frontier(experiments)
        frontier_indices = {p.index for p in frontier}
        assert frontier_indices == {1, 2, 3}
        assert 4 not in frontier_indices

    def test_all_non_keep_returns_empty(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.05, status=Status.DISCARD),
            make_experiment(index=2, val_bpb=0.0, status=Status.CRASH),
        ]
        assert compute_pareto_frontier(experiments) == []

    def test_same_bpb_different_memory_keeps_lower_memory(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.05, memory_gb=50.0),
            make_experiment(index=2, val_bpb=1.05, memory_gb=40.0),
        ]
        frontier = compute_pareto_frontier(experiments)
        # Sorted by bpb ascending, then min_memory_so_far check:
        # Both have same bpb, first encountered (idx 1 at bpb 1.05) gets added with mem=50,
        # then idx 2 (bpb 1.05, mem=40) has mem <= 50 so also added
        # The frontier should contain the lower-memory point at minimum
        frontier_indices = {p.index for p in frontier}
        assert 2 in frontier_indices


# ---------------------------------------------------------------------------
# 5. Verdict computation
# ---------------------------------------------------------------------------


class TestComputeVerdict:
    """Tests for compute_verdict: noise-based thresholds and percentage fallbacks."""

    def test_crash_status_returns_crash_verdict(self) -> None:
        latest = make_experiment(status=Status.CRASH, val_bpb=0.0)
        verdict, confidence, reason = compute_verdict(latest, prev_best_bpb=1.05, noise_floor=0.001)
        assert verdict == Verdict.CRASH
        assert confidence == 1.0

    def test_zero_bpb_returns_crash_verdict(self) -> None:
        latest = make_experiment(status=Status.KEEP, val_bpb=0.0)
        verdict, confidence, _ = compute_verdict(latest, prev_best_bpb=1.05, noise_floor=0.001)
        assert verdict == Verdict.CRASH
        assert confidence == 1.0

    def test_negative_bpb_returns_crash_verdict(self) -> None:
        latest = make_experiment(status=Status.KEEP, val_bpb=-1.0)
        verdict, _, _ = compute_verdict(latest, prev_best_bpb=1.05, noise_floor=0.001)
        assert verdict == Verdict.CRASH

    def test_strong_keep_when_signal_above_3x_noise(self) -> None:
        noise_floor = 0.001
        # Improvement of 0.004 = 4x noise
        latest = make_experiment(val_bpb=1.046)
        verdict, confidence, reason = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=noise_floor)
        assert verdict == Verdict.STRONG_KEEP
        assert confidence > 0.9
        assert "statistically significant" in reason.lower()

    def test_keep_when_signal_between_1_5x_and_3x_noise(self) -> None:
        noise_floor = 0.002
        # Improvement of 0.004 = 2x noise
        latest = make_experiment(val_bpb=1.046)
        verdict, confidence, reason = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=noise_floor)
        assert verdict == Verdict.KEEP
        assert "likely real improvement" in reason.lower()

    def test_marginal_when_signal_between_0_5x_and_1_5x_noise(self) -> None:
        noise_floor = 0.004
        # Improvement of 0.004 = 1.0x noise
        latest = make_experiment(val_bpb=1.046)
        verdict, _, reason = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=noise_floor)
        assert verdict == Verdict.MARGINAL
        assert "could be noise" in reason.lower()

    def test_retest_when_signal_below_0_5x_noise(self) -> None:
        noise_floor = 0.010
        # Improvement of 0.001 = 0.1x noise
        latest = make_experiment(val_bpb=1.049)
        verdict, confidence, reason = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=noise_floor)
        assert verdict == Verdict.RETEST
        assert confidence == pytest.approx(0.40)
        assert "retest" in reason.lower()

    def test_discard_on_regression(self) -> None:
        latest = make_experiment(val_bpb=1.060)
        verdict, _, reason = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=0.001)
        assert verdict == Verdict.DISCARD
        assert "regressed" in reason.lower()

    def test_discard_regression_within_noise_still_discard(self) -> None:
        noise_floor = 0.020
        # Regression of 0.005 which is within noise floor
        latest = make_experiment(val_bpb=1.055)
        verdict, _, reason = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=noise_floor)
        assert verdict == Verdict.DISCARD
        assert "within noise floor" in reason.lower()

    def test_discard_regression_within_noise_has_capped_confidence(self) -> None:
        noise_floor = 0.020
        latest = make_experiment(val_bpb=1.055)
        _, confidence, _ = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=noise_floor)
        assert confidence <= 0.90

    def test_confidence_capped_below_one(self) -> None:
        # Large regression should not exceed 0.99 confidence
        latest = make_experiment(val_bpb=2.000)
        _, confidence, _ = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=0.001)
        assert confidence <= 0.99

    def test_strong_keep_confidence_capped_below_one(self) -> None:
        # Very large improvement
        latest = make_experiment(val_bpb=0.500)
        _, confidence, _ = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=0.001)
        assert confidence <= 0.99

    # --- Percentage fallback (no noise floor) ---

    def test_fallback_strong_keep_at_1pct_improvement(self) -> None:
        # 1.05 -> 1.039 is about 1.05% improvement
        latest = make_experiment(val_bpb=1.039)
        verdict, confidence, reason = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=None)
        assert verdict == Verdict.STRONG_KEEP
        assert confidence == pytest.approx(0.85)
        assert "no noise floor" in reason.lower()

    def test_fallback_keep_at_0_3pct_improvement(self) -> None:
        # 1.050 -> 1.0465 is about 0.33% improvement
        latest = make_experiment(val_bpb=1.0465)
        verdict, confidence, _ = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=None)
        assert verdict == Verdict.KEEP
        assert confidence == pytest.approx(0.70)

    def test_fallback_marginal_at_0_1pct_improvement(self) -> None:
        # 1.050 -> 1.0488 is about 0.114% improvement
        latest = make_experiment(val_bpb=1.0488)
        verdict, confidence, _ = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=None)
        assert verdict == Verdict.MARGINAL
        assert confidence == pytest.approx(0.55)

    def test_fallback_retest_at_negligible_improvement(self) -> None:
        # 1.050 -> 1.0499 is about 0.0095% improvement
        latest = make_experiment(val_bpb=1.0499)
        verdict, confidence, _ = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=None)
        assert verdict == Verdict.RETEST
        assert confidence == pytest.approx(0.40)

    def test_tiny_noise_floor_treated_as_none(self) -> None:
        # Noise floor < 1e-10 should fall back to percentage thresholds
        latest = make_experiment(val_bpb=1.039)
        verdict_with_tiny, _, _ = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=1e-12)
        verdict_with_none, _, _ = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=None)
        assert verdict_with_tiny == verdict_with_none

    # --- Boundary / edge cases ---

    def test_no_change_from_prev_best_returns_retest(self) -> None:
        latest = make_experiment(val_bpb=1.050)
        verdict, _, _ = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=0.001)
        # delta == 0 means improvement case with abs_delta == 0 => retest
        assert verdict == Verdict.RETEST

    def test_discard_without_noise_floor(self) -> None:
        latest = make_experiment(val_bpb=1.060)
        verdict, _, reason = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=None)
        assert verdict == Verdict.DISCARD
        assert "regressed" in reason.lower()

    def test_reason_includes_percentage(self) -> None:
        latest = make_experiment(val_bpb=1.040)
        _, _, reason = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=0.001)
        assert "%" in reason

    def test_reason_includes_absolute_delta(self) -> None:
        latest = make_experiment(val_bpb=1.040)
        _, _, reason = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=0.001)
        assert "absolute" in reason.lower()

    def test_exactly_at_3x_noise_boundary_is_strong_keep(self) -> None:
        noise_floor = 0.001
        # Improvement of exactly 0.003 = 3.0x noise
        latest = make_experiment(val_bpb=1.047)
        verdict, _, _ = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=noise_floor)
        assert verdict == Verdict.STRONG_KEEP

    def test_exactly_at_1_5x_noise_boundary_is_keep(self) -> None:
        noise_floor = 0.002
        # Improvement of exactly 0.003 = 1.5x noise
        latest = make_experiment(val_bpb=1.047)
        verdict, _, _ = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=noise_floor)
        assert verdict == Verdict.KEEP

    def test_exactly_at_0_5x_noise_boundary_is_marginal(self) -> None:
        noise_floor = 0.002
        # Improvement of exactly 0.001 = 0.5x noise
        latest = make_experiment(val_bpb=1.049)
        verdict, _, _ = compute_verdict(latest, prev_best_bpb=1.050, noise_floor=noise_floor)
        assert verdict == Verdict.MARGINAL


# ---------------------------------------------------------------------------
# 6. Full analysis pipeline
# ---------------------------------------------------------------------------


class TestAnalyze:
    """Tests for analyze: baseline, multi-experiment, crash, error cases."""

    def test_empty_experiments_returns_err(self) -> None:
        result = analyze([])
        assert isinstance(result, Err)
        assert "no experiments" in result.error.message.lower()

    def test_single_experiment_is_baseline_keep(self) -> None:
        experiments = [make_experiment(index=1, val_bpb=1.05)]
        result = analyze(experiments)

        assert isinstance(result, Ok)
        judgment = result.value
        assert judgment.verdict == Verdict.KEEP
        assert judgment.confidence == 1.0
        assert judgment.delta == pytest.approx(0.0)

    def test_improvement_detected_across_experiments(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.050, status=Status.KEEP),
            make_experiment(index=2, val_bpb=1.040, status=Status.KEEP),
        ]
        result = analyze(experiments)

        assert isinstance(result, Ok)
        judgment = result.value
        assert judgment.delta < 0  # negative = improvement
        assert judgment.val_bpb == pytest.approx(1.040)
        assert judgment.prev_best == pytest.approx(1.050)

    def test_regression_detected(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.040, status=Status.KEEP),
            make_experiment(index=2, val_bpb=1.060, status=Status.KEEP),
        ]
        result = analyze(experiments)

        assert isinstance(result, Ok)
        judgment = result.value
        assert judgment.delta > 0  # positive = regression
        assert judgment.verdict == Verdict.DISCARD

    def test_crash_experiment_detected(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.050, status=Status.KEEP),
            make_experiment(index=2, val_bpb=0.0, status=Status.CRASH),
        ]
        result = analyze(experiments)

        assert isinstance(result, Ok)
        judgment = result.value
        assert judgment.verdict == Verdict.CRASH

    def test_trends_populated_correctly(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.050, status=Status.KEEP),
            make_experiment(index=2, val_bpb=1.060, status=Status.DISCARD),
            make_experiment(index=3, val_bpb=1.045, status=Status.KEEP),
        ]
        result = analyze(experiments)

        assert isinstance(result, Ok)
        trends = result.value.trends
        assert trends.total_experiments == 3
        assert trends.total_keeps == 2
        assert trends.total_discards == 1
        assert trends.total_crashes == 0

    def test_pareto_frontier_included_in_result(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.050, memory_gb=45.0, status=Status.KEEP),
            make_experiment(index=2, val_bpb=1.040, memory_gb=44.0, status=Status.KEEP),
        ]
        result = analyze(experiments)

        assert isinstance(result, Ok)
        assert len(result.value.pareto_frontier) > 0

    def test_analyze_returns_judgment_result_type(self) -> None:
        experiments = [make_experiment(index=1, val_bpb=1.05)]
        result = analyze(experiments)

        assert isinstance(result, Ok)
        assert isinstance(result.value, JudgmentResult)

    def test_efficiency_metrics_present(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.050, memory_gb=44.0, status=Status.KEEP),
            make_experiment(index=2, val_bpb=1.040, memory_gb=46.0, status=Status.KEEP),
        ]
        result = analyze(experiments)

        assert isinstance(result, Ok)
        eff = result.value.efficiency
        assert eff.memory_gb == pytest.approx(46.0)
        assert eff.memory_delta_gb == pytest.approx(2.0)

    def test_noise_floor_computed_with_sufficient_keeps(self) -> None:
        experiments = make_keep_series([1.050, 1.045, 1.042, 1.039, 1.035])
        result = analyze(experiments)

        assert isinstance(result, Ok)
        assert result.value.noise_floor is not None

    def test_noise_floor_none_with_few_keeps(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.050, status=Status.KEEP),
            make_experiment(index=2, val_bpb=1.040, status=Status.KEEP),
        ]
        result = analyze(experiments)

        assert isinstance(result, Ok)
        assert result.value.noise_floor is None

    def test_suggestion_field_populated(self) -> None:
        experiments = [make_experiment(index=1, val_bpb=1.05)]
        result = analyze(experiments)

        assert isinstance(result, Ok)
        assert isinstance(result.value.suggestion, str)
        assert len(result.value.suggestion) > 0

    def test_analyze_with_many_discards_flags_stuck_agent(self) -> None:
        experiments = [make_experiment(index=1, val_bpb=1.050, status=Status.KEEP)]
        for i in range(5):
            experiments.append(
                make_experiment(index=i + 2, val_bpb=1.060, status=Status.DISCARD)
            )
        result = analyze(experiments)

        assert isinstance(result, Ok)
        suggestion_lower = result.value.suggestion.lower()
        assert "stuck" in suggestion_lower or "consecutive" in suggestion_lower

    def test_prev_best_ignores_discards_for_comparison(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.050, status=Status.KEEP),
            make_experiment(index=2, val_bpb=0.900, status=Status.DISCARD),
            make_experiment(index=3, val_bpb=1.040, status=Status.KEEP),
        ]
        result = analyze(experiments)
        assert isinstance(result, Ok)
        # prev_best should be 1.050 (the only prior KEEP), not 0.900 (discard)
        assert result.value.prev_best == pytest.approx(1.050)

    def test_delta_pct_sign_convention(self) -> None:
        # Improvement: delta negative, delta_pct negative
        experiments = [
            make_experiment(index=1, val_bpb=1.050, status=Status.KEEP),
            make_experiment(index=2, val_bpb=1.040, status=Status.KEEP),
        ]
        result = analyze(experiments)
        assert isinstance(result, Ok)
        assert result.value.delta < 0
        assert result.value.delta_pct < 0

    def test_delta_pct_positive_on_regression(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.040, status=Status.KEEP),
            make_experiment(index=2, val_bpb=1.060, status=Status.KEEP),
        ]
        result = analyze(experiments)
        assert isinstance(result, Ok)
        assert result.value.delta > 0
        assert result.value.delta_pct > 0

    def test_experiment_field_references_latest(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.050, commit="first"),
            make_experiment(index=2, val_bpb=1.040, commit="second"),
        ]
        result = analyze(experiments)
        assert isinstance(result, Ok)
        assert result.value.experiment.commit == "second"
        assert result.value.experiment.index == 2

    def test_best_val_bpb_in_trends_is_global_minimum(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.050, status=Status.KEEP),
            make_experiment(index=2, val_bpb=1.030, status=Status.KEEP),
            make_experiment(index=3, val_bpb=1.040, status=Status.KEEP),
        ]
        result = analyze(experiments)
        assert isinstance(result, Ok)
        assert result.value.trends.best_val_bpb == pytest.approx(1.030)
        assert result.value.trends.best_experiment_index == 2

    def test_on_pareto_frontier_for_dominating_experiment(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.050, memory_gb=45.0, status=Status.KEEP),
            make_experiment(index=2, val_bpb=1.020, memory_gb=44.0, status=Status.KEEP),
        ]
        result = analyze(experiments)
        assert isinstance(result, Ok)
        assert result.value.on_pareto_frontier is True

    def test_off_pareto_frontier_for_dominated_experiment(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.020, memory_gb=44.0, status=Status.KEEP),
            make_experiment(index=2, val_bpb=1.050, memory_gb=46.0, status=Status.KEEP),
        ]
        result = analyze(experiments)
        assert isinstance(result, Ok)
        assert result.value.on_pareto_frontier is False

    def test_run_log_none_by_default(self) -> None:
        experiments = [make_experiment(index=1, val_bpb=1.05)]
        result = analyze(experiments)
        assert isinstance(result, Ok)
        assert result.value.run_log is None


# ---------------------------------------------------------------------------
# 7. Output formatting
# ---------------------------------------------------------------------------


class TestFormatJson:
    """Tests for format_json: all required fields present, valid JSON."""

    def _make_judgment(self) -> JudgmentResult:
        experiments = [
            make_experiment(index=1, val_bpb=1.050, memory_gb=45.0, status=Status.KEEP),
            make_experiment(index=2, val_bpb=1.040, memory_gb=44.0, status=Status.KEEP),
        ]
        result = analyze(experiments)
        assert isinstance(result, Ok)
        return result.value

    def test_format_json_is_valid_json(self) -> None:
        judgment = self._make_judgment()
        output = format_json(judgment)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_format_json_has_required_top_level_keys(self) -> None:
        judgment = self._make_judgment()
        parsed = json.loads(format_json(judgment))

        required_keys = {
            "verdict", "confidence", "val_bpb", "prev_best", "delta", "delta_pct",
            "noise_floor", "on_pareto_frontier", "suggestion", "experiment",
            "efficiency", "pareto_frontier", "experiments_analyzed",
            "discard_streak", "crash_streak", "keep_streak",
            "longest_keep_streak", "longest_discard_streak",
            "total_keeps", "total_discards", "total_crashes",
            "best_val_bpb", "improvement_rate_pct",
        }
        assert required_keys.issubset(parsed.keys())

    def test_format_json_experiment_subobject_has_required_keys(self) -> None:
        judgment = self._make_judgment()
        parsed = json.loads(format_json(judgment))

        exp_keys = {"index", "commit", "description", "val_bpb", "memory_gb", "status"}
        assert exp_keys.issubset(parsed["experiment"].keys())

    def test_format_json_efficiency_subobject_has_required_keys(self) -> None:
        judgment = self._make_judgment()
        parsed = json.loads(format_json(judgment))

        eff_keys = {"memory_gb", "memory_delta_gb", "num_params_m", "bpb_per_param_m"}
        assert eff_keys.issubset(parsed["efficiency"].keys())

    def test_format_json_pareto_frontier_is_list(self) -> None:
        judgment = self._make_judgment()
        parsed = json.loads(format_json(judgment))
        assert isinstance(parsed["pareto_frontier"], list)

    def test_format_json_pareto_frontier_items_have_keys(self) -> None:
        judgment = self._make_judgment()
        parsed = json.loads(format_json(judgment))
        for point in parsed["pareto_frontier"]:
            assert "index" in point
            assert "commit" in point
            assert "val_bpb" in point
            assert "memory_gb" in point

    def test_format_json_verdict_is_valid_enum_value(self) -> None:
        judgment = self._make_judgment()
        parsed = json.loads(format_json(judgment))
        assert isinstance(parsed["verdict"], str)
        assert parsed["verdict"] in {v.value for v in Verdict}

    def test_format_json_confidence_is_bounded(self) -> None:
        judgment = self._make_judgment()
        parsed = json.loads(format_json(judgment))
        assert isinstance(parsed["confidence"], (int, float))
        assert 0.0 <= parsed["confidence"] <= 1.0

    def test_format_json_values_match_judgment_fields(self) -> None:
        judgment = self._make_judgment()
        parsed = json.loads(format_json(judgment))
        assert parsed["val_bpb"] == pytest.approx(judgment.val_bpb)
        assert parsed["prev_best"] == pytest.approx(judgment.prev_best)
        assert parsed["experiments_analyzed"] == judgment.trends.total_experiments

    def test_format_json_noise_floor_null_when_none(self) -> None:
        # Only 2 keeps, so noise floor will be None
        experiments = [
            make_experiment(index=1, val_bpb=1.050, status=Status.KEEP),
            make_experiment(index=2, val_bpb=1.040, status=Status.KEEP),
        ]
        result = analyze(experiments)
        assert isinstance(result, Ok)
        parsed = json.loads(format_json(result.value))
        assert parsed["noise_floor"] is None

    def test_format_json_status_in_experiment_is_string(self) -> None:
        judgment = self._make_judgment()
        parsed = json.loads(format_json(judgment))
        assert parsed["experiment"]["status"] in {"keep", "discard", "crash"}


class TestFormatQuiet:
    """Tests for format_quiet: one-line format shape."""

    def _make_judgment(self) -> JudgmentResult:
        experiments = [
            make_experiment(index=1, val_bpb=1.050, memory_gb=45.0, status=Status.KEEP),
            make_experiment(index=2, val_bpb=1.040, memory_gb=44.0, status=Status.KEEP),
        ]
        result = analyze(experiments)
        assert isinstance(result, Ok)
        return result.value

    def test_quiet_format_is_single_line(self) -> None:
        judgment = self._make_judgment()
        output = format_quiet(judgment)
        assert "\n" not in output

    def test_quiet_format_starts_with_verdict(self) -> None:
        judgment = self._make_judgment()
        output = format_quiet(judgment)
        first_word = output.split()[0]
        assert first_word in {v.value for v in Verdict}

    def test_quiet_format_contains_percentage(self) -> None:
        judgment = self._make_judgment()
        output = format_quiet(judgment)
        assert "%" in output

    def test_quiet_format_contains_val_bpb(self) -> None:
        judgment = self._make_judgment()
        output = format_quiet(judgment)
        assert "1.040000" in output

    def test_quiet_format_contains_signed_delta_pct(self) -> None:
        judgment = self._make_judgment()
        output = format_quiet(judgment)
        # Should contain a + or - sign before the delta percentage
        parts = output.split()
        delta_part = parts[-1]  # last part: "+X.XX%" or "-X.XX%"
        assert delta_part.endswith("%")
        assert delta_part[0] in {"+", "-"}

    def test_quiet_format_for_crash(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.05, status=Status.KEEP),
            make_experiment(index=2, val_bpb=0.0, status=Status.CRASH),
        ]
        result = analyze(experiments)
        assert isinstance(result, Ok)
        output = format_quiet(result.value)
        assert output.startswith("CRASH")


# ---------------------------------------------------------------------------
# 8. CLI integration
# ---------------------------------------------------------------------------


class TestCli:
    """Tests for the Click CLI via CliRunner."""

    def test_cli_with_valid_results_exits_zero_for_keep(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        write_tsv(tsv, [
            ["abc1234", "1.050000", "45.2", "keep", "Baseline"],
        ])

        runner = CliRunner()
        result = runner.invoke(cli, ["--results", str(tsv), "--no-color"])

        assert result.exit_code == 0

    def test_cli_json_format_produces_valid_json(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        write_tsv(tsv, [
            ["abc1234", "1.050000", "45.2", "keep", "Baseline"],
            ["def5678", "1.040000", "44.0", "keep", "Improvement"],
        ])

        runner = CliRunner()
        result = runner.invoke(cli, ["--results", str(tsv), "--format", "json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "verdict" in parsed

    def test_cli_quiet_mode_produces_single_line(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        write_tsv(tsv, [
            ["abc1234", "1.050000", "45.2", "keep", "Baseline"],
        ])

        runner = CliRunner()
        result = runner.invoke(cli, ["--results", str(tsv), "--quiet", "--no-color"])

        assert result.exit_code == 0
        lines = result.output.strip().splitlines()
        assert len(lines) == 1

    def test_cli_exit_code_2_for_discard_verdict(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        write_tsv(tsv, [
            ["abc1234", "1.040000", "45.0", "keep", "Baseline"],
            ["def5678", "1.080000", "46.0", "keep", "Regression"],
        ])

        runner = CliRunner()
        result = runner.invoke(cli, ["--results", str(tsv), "--format", "json"])

        assert result.exit_code == 2

    def test_cli_exit_code_2_for_crash_verdict(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        write_tsv(tsv, [
            ["abc1234", "1.050000", "45.0", "keep", "Baseline"],
            ["def5678", "0.000000", "0.0", "crash", "OOM"],
        ])

        runner = CliRunner()
        result = runner.invoke(cli, ["--results", str(tsv), "--format", "json"])

        assert result.exit_code == 2

    def test_cli_exit_code_1_for_missing_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--results", "/nonexistent/results.tsv"])

        assert result.exit_code == 1

    def test_cli_exit_code_1_for_empty_file(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        tsv.write_text("", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(cli, ["--results", str(tsv)])

        assert result.exit_code == 1

    def test_cli_human_format_is_default(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        write_tsv(tsv, [
            ["abc1234", "1.050000", "45.2", "keep", "Baseline"],
        ])

        runner = CliRunner()
        result = runner.invoke(cli, ["--results", str(tsv), "--no-color"])

        assert result.exit_code == 0
        assert "autojudge" in result.output.lower()

    def test_cli_version_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "1.0.0" in result.output

    def test_cli_json_output_matches_verdict_from_analysis(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        write_tsv(tsv, [
            ["abc1234", "1.050000", "45.2", "keep", "Baseline"],
            ["def5678", "1.040000", "44.0", "keep", "Better"],
        ])

        runner = CliRunner()
        result = runner.invoke(cli, ["--results", str(tsv), "--format", "json"])

        parsed = json.loads(result.output)
        tsv_result = parse_results_tsv(tsv)
        assert isinstance(tsv_result, Ok)
        analysis_result = analyze(tsv_result.value)
        assert isinstance(analysis_result, Ok)
        assert parsed["verdict"] == analysis_result.value.verdict.value

    def test_cli_error_output_for_missing_file(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--results", str(tmp_path / "nope.tsv")])

        assert result.exit_code == 1
        # Error message goes to combined output in CliRunner
        assert "error" in result.output.lower()

    def test_cli_error_output_for_malformed_tsv(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        tsv.write_text("commit\tval_bpb\n" "abc\t1.05\n", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(cli, ["--results", str(tsv)])

        assert result.exit_code == 1
        assert "error" in result.output.lower()

    def test_cli_with_many_experiments(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        rows = []
        for i in range(20):
            bpb = 1.050 - i * 0.001
            rows.append([f"commit_{i:03d}", f"{bpb:.6f}", "45.0", "keep", f"exp {i}"])
        write_tsv(tsv, rows)

        runner = CliRunner()
        result = runner.invoke(cli, ["--results", str(tsv), "--format", "json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["experiments_analyzed"] == 20

    def test_cli_quiet_json_flags_independent(self, tmp_path: Path) -> None:
        # --format json should take precedence over --quiet
        tsv = tmp_path / "results.tsv"
        write_tsv(tsv, [
            ["abc1234", "1.050000", "45.2", "keep", "Baseline"],
        ])

        runner = CliRunner()
        result = runner.invoke(cli, ["--results", str(tsv), "--format", "json", "--quiet"])

        assert result.exit_code == 0
        # JSON format should be valid JSON regardless of --quiet
        parsed = json.loads(result.output)
        assert "verdict" in parsed


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and integration scenarios that span multiple functions."""

    def test_single_crash_experiment_as_baseline(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=0.0, status=Status.CRASH),
        ]
        result = analyze(experiments)
        # Single experiment is treated as baseline
        assert isinstance(result, Ok)

    def test_large_number_of_experiments(self) -> None:
        experiments = make_keep_series(
            [1.050 - i * 0.0001 for i in range(1000)],
        )
        result = analyze(experiments)
        assert isinstance(result, Ok)
        assert result.value.trends.total_experiments == 1000

    def test_very_small_improvements_near_noise(self) -> None:
        # Tiny improvement: 1.050000 -> 1.049999
        experiments = [
            make_experiment(index=1, val_bpb=1.050000, status=Status.KEEP),
            make_experiment(index=2, val_bpb=1.049999, status=Status.KEEP),
        ]
        result = analyze(experiments)
        assert isinstance(result, Ok)
        assert result.value.verdict == Verdict.RETEST

    def test_all_discards_after_baseline(self) -> None:
        experiments = [make_experiment(index=1, val_bpb=1.050, status=Status.KEEP)]
        for i in range(3):
            experiments.append(
                make_experiment(index=i + 2, val_bpb=1.060, status=Status.DISCARD)
            )
        result = analyze(experiments)
        assert isinstance(result, Ok)
        assert result.value.verdict == Verdict.DISCARD
        assert result.value.trends.discard_streak == 3

    def test_alternating_keep_discard_pattern(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.050, status=Status.KEEP),
            make_experiment(index=2, val_bpb=1.060, status=Status.DISCARD),
            make_experiment(index=3, val_bpb=1.045, status=Status.KEEP),
            make_experiment(index=4, val_bpb=1.070, status=Status.DISCARD),
            make_experiment(index=5, val_bpb=1.040, status=Status.KEEP),
        ]
        result = analyze(experiments)
        assert isinstance(result, Ok)
        assert result.value.trends.total_keeps == 3
        assert result.value.trends.total_discards == 2
        assert result.value.trends.longest_keep_streak == 1

    def test_multiple_crashes_in_sequence(self) -> None:
        experiments = [
            make_experiment(index=1, val_bpb=1.050, status=Status.KEEP),
            make_experiment(index=2, val_bpb=0.0, status=Status.CRASH),
            make_experiment(index=3, val_bpb=0.0, status=Status.CRASH),
        ]
        result = analyze(experiments)
        assert isinstance(result, Ok)
        suggestion = result.value.suggestion.lower()
        assert "crash" in suggestion or "oom" in suggestion.lower()

    def test_result_type_ok_has_value_attribute(self) -> None:
        ok = Ok(value=42)
        assert ok.value == 42

    def test_result_type_err_has_error_attribute(self) -> None:
        err = Err(error=ParseError(message="test", source="test.tsv"))
        assert err.error.message == "test"
        assert err.error.source == "test.tsv"

    def test_experiment_is_frozen_dataclass(self) -> None:
        exp = make_experiment()
        with pytest.raises(AttributeError):
            exp.val_bpb = 99.0  # type: ignore[misc]

    def test_parse_error_default_source_is_none(self) -> None:
        err = ParseError(message="something went wrong")
        assert err.source is None

    def test_full_round_trip_tsv_to_json(self, tmp_path: Path) -> None:
        """End-to-end: write TSV, parse, analyze, format JSON, verify structure."""
        tsv = tmp_path / "results.tsv"
        write_tsv(tsv, [
            ["abc1234", "4.020000", "45.2", "keep", "Baseline"],
            ["def5678", "3.950000", "45.5", "keep", "Increase depth"],
            ["ghi9012", "4.100000", "46.0", "discard", "Bad batch size"],
            ["jkl3456", "3.900000", "44.8", "keep", "Tune LR"],
            ["mno7890", "3.850000", "44.5", "keep", "SwiGLU"],
        ])

        parse_result = parse_results_tsv(tsv)
        assert isinstance(parse_result, Ok)
        assert len(parse_result.value) == 5

        analysis_result = analyze(parse_result.value)
        assert isinstance(analysis_result, Ok)

        json_output = format_json(analysis_result.value)
        parsed = json.loads(json_output)

        assert parsed["experiments_analyzed"] == 5
        assert parsed["total_keeps"] == 4
        assert parsed["total_discards"] == 1
        assert parsed["experiment"]["commit"] == "mno7890"
        assert parsed["val_bpb"] == pytest.approx(3.85)
