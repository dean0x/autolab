"""auto-judge: Smarter experiment evaluation for autoresearch.

Analyzes experiment results from Karpathy's autoresearch and provides
structured verdicts beyond simple "did val_bpb go down?" checks.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generic, Optional, TypeVar

import click


@dataclass(frozen=True)
class OutputConfig:
    color: bool
    quiet: bool
    def styled(self, text: str, **kwargs) -> str:
        return click.style(text, **kwargs) if self.color else text

# Status symbols
SYM_KEEP = "\u2714"    # ✔
SYM_FAIL = "\u2718"    # ✘
SYM_CRASH = "\u2620"   # ☠
SYM_WARN = "\u26A0"    # ⚠
SYM_ARROW = "\u2192"   # →
SYM_STAR = "\u2605"    # ★


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class Verdict(str, Enum):
    STRONG_KEEP = "STRONG_KEEP"
    KEEP = "KEEP"
    MARGINAL = "MARGINAL"
    RETEST = "RETEST"
    DISCARD = "DISCARD"
    CRASH = "CRASH"


class Status(str, Enum):
    KEEP = "keep"
    DISCARD = "discard"
    CRASH = "crash"


@dataclass(frozen=True)
class Experiment:
    index: int
    commit: str
    val_bpb: float
    memory_gb: float
    status: Status
    description: str


@dataclass(frozen=True)
class RunLogMetrics:
    training_seconds: float
    total_seconds: float
    peak_vram_mb: float
    mfu_percent: float
    total_tokens_m: float
    num_steps: int
    num_params_m: float
    depth: int


@dataclass(frozen=True)
class EfficiencyMetrics:
    memory_gb: float
    memory_delta_gb: float
    num_params_m: Optional[float]
    bpb_per_param_m: Optional[float]


@dataclass(frozen=True)
class TrendMetrics:
    best_val_bpb: float
    best_experiment_index: int
    improvement_rate_pct: Optional[float]
    improvement_rate_window: int
    discard_streak: int
    crash_streak: int
    keep_streak: int
    longest_keep_streak: int
    longest_discard_streak: int
    total_experiments: int
    total_keeps: int
    total_discards: int
    total_crashes: int


@dataclass(frozen=True)
class ParetoPoint:
    index: int
    commit: str
    val_bpb: float
    memory_gb: float


@dataclass(frozen=True)
class JudgmentResult:
    """Structured result of experiment analysis."""

    verdict: Verdict
    confidence: float
    val_bpb: float
    prev_best: float
    delta: float
    delta_pct: float
    noise_floor: Optional[float]
    on_pareto_frontier: bool
    pareto_frontier: list[ParetoPoint]
    efficiency: EfficiencyMetrics
    trends: TrendMetrics
    suggestion: str
    experiment: Experiment
    run_log: Optional[RunLogMetrics]


# ---------------------------------------------------------------------------
# Parsing — parse, don't validate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParseError:
    message: str
    source: Optional[str] = None


T = TypeVar("T")


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T


@dataclass(frozen=True)
class Err:
    error: ParseError


def parse_results_tsv(path: Path) -> Ok[list[Experiment]] | Err:
    """Parse results.tsv into a list of Experiment objects."""
    if not path.exists():
        return Err(ParseError(f"Results file not found: {path}", str(path)))
    if not path.is_file():
        return Err(ParseError(f"Not a file: {path}", str(path)))

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return Err(ParseError("Results file is empty", str(path)))

    lines = text.splitlines()
    if len(lines) < 2:
        return Err(ParseError("Results file has no data rows (only header?)", str(path)))

    # Validate header
    header = lines[0].split("\t")
    expected_columns = {"commit", "val_bpb", "memory_gb", "status", "description"}
    actual_columns = {col.strip().lower() for col in header}
    missing = expected_columns - actual_columns
    if missing:
        return Err(ParseError(
            f"Missing columns in TSV header: {missing}. "
            f"Found: {[col.strip() for col in header]}",
            str(path),
        ))

    # Build column index map from header
    col_map: dict[str, int] = {}
    for i, col in enumerate(header):
        col_map[col.strip().lower()] = i

    experiments: list[Experiment] = []
    for line_num, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) <= max(col_map.values()):
            return Err(ParseError(
                f"Line {line_num}: expected at least {max(col_map.values()) + 1} "
                f"tab-separated fields, got {len(fields)}",
                str(path),
            ))

        commit = fields[col_map["commit"]].strip()

        try:
            val_bpb = float(fields[col_map["val_bpb"]].strip())
        except ValueError:
            return Err(ParseError(
                f"Line {line_num}: invalid val_bpb value: "
                f"{fields[col_map['val_bpb']].strip()!r}",
                str(path),
            ))

        try:
            memory_gb = float(fields[col_map["memory_gb"]].strip())
        except ValueError:
            return Err(ParseError(
                f"Line {line_num}: invalid memory_gb value: "
                f"{fields[col_map['memory_gb']].strip()!r}",
                str(path),
            ))

        status_str = fields[col_map["status"]].strip().lower()
        try:
            status = Status(status_str)
        except ValueError:
            return Err(ParseError(
                f"Line {line_num}: invalid status: {status_str!r}. "
                f"Expected one of: keep, discard, crash",
                str(path),
            ))

        description = fields[col_map["description"]].strip()

        experiments.append(Experiment(
            index=len(experiments) + 1,
            commit=commit,
            val_bpb=val_bpb,
            memory_gb=memory_gb,
            status=status,
            description=description,
        ))

    return Ok(experiments)


def parse_run_log(path: Path) -> Ok[RunLogMetrics] | Err:
    """Parse run.log to extract training metrics from the --- block."""
    if not path.exists():
        return Err(ParseError(f"Run log not found: {path}", str(path)))

    text = path.read_text(encoding="utf-8")

    # Find the metrics block after "---"
    separator_idx = text.rfind("\n---\n")
    if separator_idx == -1:
        # Try at start of file
        if text.startswith("---\n"):
            separator_idx = 0
        else:
            return Err(ParseError(
                "No '---' separator found in run log. "
                "Expected metrics block at end of file.",
                str(path),
            ))

    metrics_text = text[separator_idx:]
    metrics: dict[str, str] = {}
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line == "---":
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        metrics[key.strip().lower()] = value.strip()

    required_keys = {
        "training_seconds", "total_seconds", "peak_vram_mb",
        "mfu_percent", "total_tokens_m", "num_steps",
        "num_params_m", "depth",
    }
    # Normalize key names (run.log uses val_bpb too but we get that from TSV)
    missing = required_keys - set(metrics.keys())
    if missing:
        return Err(ParseError(
            f"Missing metrics in run log: {missing}. Found: {list(metrics.keys())}",
            str(path),
        ))

    try:
        result = RunLogMetrics(
            training_seconds=float(metrics["training_seconds"]),
            total_seconds=float(metrics["total_seconds"]),
            peak_vram_mb=float(metrics["peak_vram_mb"]),
            mfu_percent=float(metrics["mfu_percent"]),
            total_tokens_m=float(metrics["total_tokens_m"]),
            num_steps=int(metrics["num_steps"]),
            num_params_m=float(metrics["num_params_m"]),
            depth=int(metrics["depth"]),
        )
    except (ValueError, KeyError) as exc:
        return Err(ParseError(f"Failed to parse run log metrics: {exc}", str(path)))

    return Ok(result)


# ---------------------------------------------------------------------------
# Analysis functions — pure computation, no side effects
# ---------------------------------------------------------------------------


def estimate_noise_floor(experiments: list[Experiment], window: int = 5) -> Optional[float]:
    """Estimate noise floor from step-to-step variance of recent 'keep' experiments.

    Computes pairwise differences between consecutive keeps and returns
    the standard deviation of those differences. This isolates measurement
    noise from any underlying improvement trend.

    Requires at least 3 keeps (which gives at least 2 pairwise diffs).
    Returns None if insufficient data.
    """
    keeps = [e for e in experiments if e.status == Status.KEEP and e.val_bpb > 0.0]
    if len(keeps) < 3:
        return None

    recent_keeps = keeps[-window:]
    if len(recent_keeps) < 2:
        return None

    diffs = [
        recent_keeps[i].val_bpb - recent_keeps[i - 1].val_bpb
        for i in range(1, len(recent_keeps))
    ]
    n = len(diffs)
    if n < 2:
        return None

    mean = sum(diffs) / n
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    return math.sqrt(variance)


def compute_improvement_rate(
    experiments: list[Experiment], window: int = 5,
) -> tuple[Optional[float], int]:
    """Compute average improvement rate over the last `window` experiments.

    Returns (rate_pct, actual_window). Rate is negative when regressing.
    """
    keeps = [e for e in experiments if e.status == Status.KEEP and e.val_bpb > 0.0]
    if len(keeps) < 2:
        return None, 0

    recent = keeps[-window:]
    if len(recent) < 2:
        return None, 0

    # Linear improvement rate: (first - last) / first * 100 / (n-1)
    # Positive means improving (val_bpb going down)
    first = recent[0].val_bpb
    last = recent[-1].val_bpb
    if first == 0.0:
        return None, len(recent)

    total_improvement_pct = (first - last) / first * 100.0
    rate_per_step = total_improvement_pct / (len(recent) - 1)
    return rate_per_step, len(recent)


def compute_streaks(experiments: list[Experiment]) -> dict[str, int]:
    """Compute current and longest streaks for each status."""
    if not experiments:
        return {
            "discard_streak": 0, "crash_streak": 0, "keep_streak": 0,
            "longest_keep_streak": 0, "longest_discard_streak": 0,
        }

    # Current trailing streak (from the end)
    discard_streak = 0
    crash_streak = 0
    keep_streak = 0

    for exp in reversed(experiments):
        if exp.status == Status.DISCARD:
            if crash_streak > 0 or keep_streak > 0:
                break
            discard_streak += 1
        elif exp.status == Status.CRASH:
            if discard_streak > 0 or keep_streak > 0:
                break
            crash_streak += 1
        elif exp.status == Status.KEEP:
            if discard_streak > 0 or crash_streak > 0:
                break
            keep_streak += 1

    # Longest streaks ever
    longest_keep = 0
    longest_discard = 0
    current_keep = 0
    current_discard = 0

    for exp in experiments:
        if exp.status == Status.KEEP:
            current_keep += 1
            current_discard = 0
        elif exp.status == Status.DISCARD:
            current_discard += 1
            current_keep = 0
        else:
            current_keep = 0
            current_discard = 0
        longest_keep = max(longest_keep, current_keep)
        longest_discard = max(longest_discard, current_discard)

    return {
        "discard_streak": discard_streak,
        "crash_streak": crash_streak,
        "keep_streak": keep_streak,
        "longest_keep_streak": longest_keep,
        "longest_discard_streak": longest_discard,
    }


def compute_pareto_frontier(experiments: list[Experiment]) -> list[ParetoPoint]:
    """Compute Pareto frontier of val_bpb (lower is better) vs memory_gb (lower is better).

    Only considers 'keep' experiments with val_bpb > 0.
    """
    candidates = [
        e for e in experiments
        if e.status == Status.KEEP and e.val_bpb > 0.0
    ]
    if not candidates:
        return []

    # Sort by val_bpb ascending (best first)
    sorted_by_bpb = sorted(candidates, key=lambda e: e.val_bpb)

    frontier: list[ParetoPoint] = []
    min_memory_so_far = float("inf")

    for exp in sorted_by_bpb:
        if exp.memory_gb <= min_memory_so_far:
            frontier.append(ParetoPoint(
                index=exp.index,
                commit=exp.commit,
                val_bpb=exp.val_bpb,
                memory_gb=exp.memory_gb,
            ))
            min_memory_so_far = exp.memory_gb

    return frontier


def is_on_pareto_frontier(
    experiment: Experiment, frontier: list[ParetoPoint],
) -> bool:
    """Check if the experiment lies on the Pareto frontier."""
    return any(
        p.index == experiment.index
        for p in frontier
    )


def compute_verdict(
    latest: Experiment,
    prev_best_bpb: float,
    noise_floor: Optional[float],
) -> tuple[Verdict, float, str]:
    """Determine verdict, confidence, and reason for the latest experiment.

    Returns (verdict, confidence, reason).
    """
    if latest.status == Status.CRASH:
        return Verdict.CRASH, 1.0, "Experiment crashed (OOM or runtime error)"

    val_bpb = latest.val_bpb
    if val_bpb <= 0.0:
        return Verdict.CRASH, 1.0, "val_bpb is zero or negative, likely a crash"

    delta = val_bpb - prev_best_bpb  # positive = regression, negative = improvement
    abs_delta = abs(delta)

    if prev_best_bpb > 0.0:
        delta_pct = (delta / prev_best_bpb) * 100.0
    else:
        delta_pct = 0.0

    # Regression case
    if delta > 0.0:
        if noise_floor is not None and abs_delta < noise_floor:
            confidence = 0.6 + 0.3 * (abs_delta / noise_floor)
            return (
                Verdict.DISCARD,
                min(confidence, 0.90),
                f"val_bpb regressed by {abs(delta_pct):.2f}% ({abs_delta:.6f} absolute), "
                f"within noise floor ({noise_floor:.6f})",
            )
        confidence = min(0.80 + 0.15 * min(abs(delta_pct), 5.0) / 5.0, 0.99)
        return (
            Verdict.DISCARD,
            confidence,
            f"val_bpb regressed by {abs(delta_pct):.2f}% ({abs_delta:.6f} absolute)",
        )

    # Improvement case (delta <= 0)
    if noise_floor is not None and noise_floor < 1e-10:
        noise_floor = None  # treat as insufficient data, use percentage fallback

    if noise_floor is not None:
        signal_to_noise = abs_delta / noise_floor

        if signal_to_noise >= 3.0:
            return (
                Verdict.STRONG_KEEP,
                min(0.90 + 0.08 * min(signal_to_noise, 10.0) / 10.0, 0.99),
                f"val_bpb improved by {abs(delta_pct):.2f}% ({abs_delta:.6f} absolute), "
                f"{signal_to_noise:.1f}x noise floor — statistically significant",
            )
        elif signal_to_noise >= 1.5:
            return (
                Verdict.KEEP,
                0.70 + 0.15 * (signal_to_noise - 1.5) / 1.5,
                f"val_bpb improved by {abs(delta_pct):.2f}% ({abs_delta:.6f} absolute), "
                f"{signal_to_noise:.1f}x noise floor — likely real improvement",
            )
        elif signal_to_noise >= 0.5:
            return (
                Verdict.MARGINAL,
                0.50 + 0.15 * (signal_to_noise - 0.5),
                f"val_bpb improved by {abs(delta_pct):.2f}% ({abs_delta:.6f} absolute), "
                f"{signal_to_noise:.1f}x noise floor — could be noise",
            )
        else:
            return (
                Verdict.RETEST,
                0.40,
                f"val_bpb improved by {abs(delta_pct):.2f}% ({abs_delta:.6f} absolute), "
                f"only {signal_to_noise:.1f}x noise floor — likely noise, retest recommended",
            )

    # No noise floor data — fall back to percentage thresholds
    if abs(delta_pct) >= 1.0:
        return (
            Verdict.STRONG_KEEP,
            0.85,
            f"val_bpb improved by {abs(delta_pct):.2f}% ({abs_delta:.6f} absolute) "
            f"— substantial improvement (no noise floor estimate available)",
        )
    elif abs(delta_pct) >= 0.3:
        return (
            Verdict.KEEP,
            0.70,
            f"val_bpb improved by {abs(delta_pct):.2f}% ({abs_delta:.6f} absolute) "
            f"— moderate improvement (no noise floor estimate available)",
        )
    elif abs(delta_pct) >= 0.1:
        return (
            Verdict.MARGINAL,
            0.55,
            f"val_bpb improved by {abs(delta_pct):.2f}% ({abs_delta:.6f} absolute) "
            f"— small improvement, unclear significance",
        )
    else:
        return (
            Verdict.RETEST,
            0.40,
            f"val_bpb improved by {abs(delta_pct):.2f}% ({abs_delta:.6f} absolute) "
            f"— negligible change, retest recommended",
        )


def generate_suggestion(
    verdict: Verdict,
    trends: TrendMetrics,
    on_pareto: bool,
    latest: Experiment,
) -> str:
    """Generate a human-readable suggestion based on the analysis."""
    parts: list[str] = []

    fail_streak = trends.discard_streak + trends.crash_streak
    if fail_streak >= 5:
        parts.append(
            f"ALERT: {fail_streak} consecutive failures. "
            "The agent appears stuck. Consider resetting to baseline "
            "and trying a fundamentally different approach."
        )
    elif fail_streak >= 3:
        parts.append(
            f"Warning: {fail_streak} consecutive failure(s). "
            "Consider a different direction."
        )

    if trends.crash_streak >= 2:
        parts.append(
            "Multiple consecutive crashes detected. "
            "Check for OOM issues or reduce model size."
        )

    if verdict == Verdict.DISCARD:
        parts.append("Consider reverting to a different approach.")
        if fail_streak > 0:
            parts.append(
                f"Last {fail_streak} experiment(s) have been "
                f"{'crashes' if trends.crash_streak > 0 else 'discards'}."
            )

    if verdict == Verdict.RETEST:
        parts.append(
            "Improvement is within noise. "
            "Run the experiment again to verify."
        )

    if verdict == Verdict.MARGINAL:
        parts.append(
            "Improvement is marginal. Consider keeping but "
            "watch for diminishing returns."
        )

    if verdict in (Verdict.STRONG_KEEP, Verdict.KEEP):
        if on_pareto:
            parts.append("This experiment is on the Pareto frontier. Good progress.")
        else:
            parts.append(
                "Good improvement in val_bpb, but not on the Pareto frontier "
                "(another experiment achieves better val_bpb with less memory)."
            )

    if (
        trends.improvement_rate_pct is not None
        and trends.improvement_rate_pct < 0.05
        and trends.total_keeps >= 5
    ):
        parts.append(
            "Improvements are diminishing. The research may be approaching "
            "a plateau. Consider a larger architectural change."
        )

    if latest.memory_gb > 0 and trends.total_experiments > 1:
        if latest.memory_gb > 70.0:
            parts.append(
                f"Memory usage ({latest.memory_gb:.1f} GB) is high. "
                "Watch for OOM in future experiments."
            )

    return "\n".join(parts) if parts else "No specific suggestions."


# ---------------------------------------------------------------------------
# Main analysis orchestrator
# ---------------------------------------------------------------------------


def analyze(
    experiments: list[Experiment],
    run_log: Optional[RunLogMetrics] = None,
) -> Ok[JudgmentResult] | Err:
    """Run the full analysis pipeline and return a JudgmentResult."""
    if not experiments:
        return Err(ParseError("No experiments to analyze"))

    latest = experiments[-1]

    # Single experiment is always the baseline — no comparison needed
    is_baseline = len(experiments) == 1

    # Find previous best val_bpb (from all experiments before the latest)
    prior_keeps = [
        e for e in experiments[:-1]
        if e.status == Status.KEEP and e.val_bpb > 0.0
    ]
    if prior_keeps:
        prev_best_bpb = min(e.val_bpb for e in prior_keeps)
    else:
        # No prior keeps — compare against self (first experiment is baseline)
        prev_best_bpb = latest.val_bpb

    # Noise floor
    noise_floor = estimate_noise_floor(experiments)

    # Verdict
    if is_baseline:
        verdict = Verdict.KEEP
        confidence = 1.0
        reason = "Baseline established"
    else:
        verdict, confidence, reason = compute_verdict(latest, prev_best_bpb, noise_floor)

    # Delta (signed: negative = improvement, positive = regression)
    delta = latest.val_bpb - prev_best_bpb
    delta_pct = (delta / prev_best_bpb * 100.0) if prev_best_bpb > 0.0 else 0.0

    # Pareto frontier
    pareto_frontier = compute_pareto_frontier(experiments)
    on_pareto = is_on_pareto_frontier(latest, pareto_frontier)

    # Best overall
    all_valid = [e for e in experiments if e.status == Status.KEEP and e.val_bpb > 0.0]
    if all_valid:
        best_exp = min(all_valid, key=lambda e: e.val_bpb)
        best_val_bpb = best_exp.val_bpb
        best_index = best_exp.index
    else:
        best_val_bpb = latest.val_bpb
        best_index = latest.index

    # Improvement rate
    improvement_rate, rate_window = compute_improvement_rate(experiments)

    # Streaks
    streaks = compute_streaks(experiments)

    # Efficiency
    prev_keeps = [e for e in experiments[:-1] if e.status == Status.KEEP and e.memory_gb > 0]
    prev_memory = prev_keeps[-1].memory_gb if prev_keeps else latest.memory_gb
    num_params_m = run_log.num_params_m if run_log else None
    bpb_per_param_m: Optional[float] = None
    if num_params_m is not None and num_params_m > 0.0 and latest.val_bpb > 0.0:
        bpb_per_param_m = latest.val_bpb / num_params_m

    efficiency = EfficiencyMetrics(
        memory_gb=latest.memory_gb,
        memory_delta_gb=latest.memory_gb - prev_memory,
        num_params_m=num_params_m,
        bpb_per_param_m=bpb_per_param_m,
    )

    # Trends
    trends = TrendMetrics(
        best_val_bpb=best_val_bpb,
        best_experiment_index=best_index,
        improvement_rate_pct=improvement_rate,
        improvement_rate_window=rate_window,
        discard_streak=streaks["discard_streak"],
        crash_streak=streaks["crash_streak"],
        keep_streak=streaks["keep_streak"],
        longest_keep_streak=streaks["longest_keep_streak"],
        longest_discard_streak=streaks["longest_discard_streak"],
        total_experiments=len(experiments),
        total_keeps=sum(1 for e in experiments if e.status == Status.KEEP),
        total_discards=sum(1 for e in experiments if e.status == Status.DISCARD),
        total_crashes=sum(1 for e in experiments if e.status == Status.CRASH),
    )

    # Suggestion
    suggestion = generate_suggestion(verdict, trends, on_pareto, latest)

    return Ok(JudgmentResult(
        verdict=verdict,
        confidence=confidence,
        val_bpb=latest.val_bpb,
        prev_best=prev_best_bpb,
        delta=delta,
        delta_pct=delta_pct,
        noise_floor=noise_floor,
        on_pareto_frontier=on_pareto,
        pareto_frontier=pareto_frontier,
        efficiency=efficiency,
        trends=trends,
        suggestion=suggestion,
        experiment=latest,
        run_log=run_log,
    ))


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_quiet(result: JudgmentResult) -> str:
    """One-line quiet output: VERDICT confidence val_bpb delta_pct"""
    return (
        f"{result.verdict.value} "
        f"{result.confidence:.0%} "
        f"{result.val_bpb:.6f} "
        f"{result.delta_pct:+.2f}%"
    )


def format_human(result: JudgmentResult, cfg: OutputConfig) -> str:
    """Format the judgment as human-readable text with optional color."""
    lines: list[str] = []
    exp = result.experiment

    lines.append(f"== {cfg.styled('autojudge', fg='cyan', bold=True)} verdict ==")
    lines.append(f'Experiment: {exp.commit} "{exp.description}"')

    if result.verdict == Verdict.CRASH:
        lines.append(f"val_bpb: {exp.val_bpb:.6f} (CRASH)")
    else:
        lines.append(f"val_bpb: {exp.val_bpb:.6f} (prev best: {result.prev_best:.6f})")

    lines.append("")

    # Verdict with symbol and color
    verdict_styles: dict[Verdict, tuple[str, dict]] = {
        Verdict.STRONG_KEEP: (SYM_KEEP, {"fg": "green", "bold": True}),
        Verdict.KEEP: (SYM_KEEP, {"fg": "green"}),
        Verdict.MARGINAL: (SYM_WARN, {"fg": "yellow"}),
        Verdict.RETEST: (SYM_WARN, {"fg": "yellow"}),
        Verdict.DISCARD: (SYM_FAIL, {"fg": "red"}),
        Verdict.CRASH: (SYM_CRASH, {"fg": "red", "bold": True}),
    }
    sym, style = verdict_styles[result.verdict]
    lines.append(f"Verdict:    {cfg.styled(sym, **style)} {cfg.styled(result.verdict.value, **style)}")
    lines.append(f"Confidence: {result.confidence:.0%}")

    # Reason line
    if result.verdict == Verdict.CRASH:
        lines.append("Reason:     Experiment crashed")
    elif result.delta > 0:
        lines.append(
            f"Reason:     val_bpb regressed by {abs(result.delta_pct):.2f}% "
            f"({abs(result.delta):.6f} absolute)"
        )
    elif result.delta < 0:
        lines.append(
            f"Reason:     val_bpb improved by {abs(result.delta_pct):.2f}% "
            f"({abs(result.delta):.6f} absolute)"
        )
    else:
        lines.append("Reason:     No change from previous best")

    if result.noise_floor is not None:
        lines.append(f"Noise floor: {result.noise_floor:.6f}")

    # Efficiency
    lines.append("")
    lines.append(cfg.styled("Efficiency:", dim=True))
    eff = result.efficiency
    delta_sign = "+" if eff.memory_delta_gb >= 0 else ""
    if eff.memory_delta_gb == 0.0:
        lines.append(f"  Memory:     {eff.memory_gb:.1f} GB (no change)")
    else:
        lines.append(
            f"  Memory:     {eff.memory_gb:.1f} GB "
            f"({delta_sign}{eff.memory_delta_gb:.1f} GB)"
        )
    if eff.num_params_m is not None:
        lines.append(f"  Params:     {eff.num_params_m:.1f}M")
    if eff.bpb_per_param_m is not None:
        lines.append(f"  BPB/param:  {eff.bpb_per_param_m:.5f} per M params")

    # Run log extras
    if result.run_log is not None:
        rl = result.run_log
        lines.append(f"  MFU:        {rl.mfu_percent:.1f}%")
        lines.append(f"  Tokens:     {rl.total_tokens_m:.1f}M in {rl.training_seconds:.0f}s")
        lines.append(f"  Steps:      {rl.num_steps}")

    # Trends
    lines.append("")
    lines.append(cfg.styled("Trends:", dim=True))
    t = result.trends
    lines.append(f"  Best val_bpb:      {t.best_val_bpb:.6f} (experiment #{t.best_experiment_index})")
    if t.improvement_rate_pct is not None:
        lines.append(
            f"  Improvement rate:  {t.improvement_rate_pct:+.2f}% per experiment "
            f"(last {t.improvement_rate_window})"
        )
    else:
        lines.append("  Improvement rate:  insufficient data")

    if t.discard_streak > 0:
        lines.append(f"  Discard streak:    {t.discard_streak}")
    elif t.crash_streak > 0:
        lines.append(f"  Crash streak:      {t.crash_streak}")
    elif t.keep_streak > 0:
        lines.append(f"  Keep streak:       {t.keep_streak}")

    lines.append(
        f"  Experiments:       {t.total_experiments} total "
        f"({t.total_keeps} kept, {t.total_discards} discarded, {t.total_crashes} crashed)"
    )

    # Pareto
    lines.append("")
    if result.on_pareto_frontier:
        lines.append(f"Pareto:  {cfg.styled(SYM_KEEP, fg='green')} On frontier (best val_bpb for its memory usage)")
    elif result.pareto_frontier:
        dominators = [
            p for p in result.pareto_frontier
            if p.val_bpb <= exp.val_bpb and p.memory_gb <= exp.memory_gb
        ]
        if dominators:
            best_dom = min(dominators, key=lambda p: p.val_bpb)
            lines.append(
                f"Pareto:  Not on frontier (dominated by experiment #{best_dom.index})"
            )
        else:
            lines.append("Pareto:  Not on frontier")
    else:
        lines.append("Pareto:  No frontier computed (no valid kept experiments)")

    # Suggestion
    lines.append("")
    lines.append(f"{cfg.styled(SYM_ARROW, fg='cyan')} {cfg.styled('Suggestion:', fg='cyan')} {result.suggestion}")

    return "\n".join(lines)


def format_json(result: JudgmentResult) -> str:
    """Format the judgment as JSON."""
    output = {
        "verdict": result.verdict.value,
        "confidence": round(result.confidence, 4),
        "val_bpb": result.val_bpb,
        "prev_best": result.prev_best,
        "delta": round(result.delta, 6),
        "delta_pct": round(result.delta_pct, 4),
        "noise_floor": round(result.noise_floor, 6) if result.noise_floor is not None else None,
        "on_pareto_frontier": result.on_pareto_frontier,
        "discard_streak": result.trends.discard_streak,
        "crash_streak": result.trends.crash_streak,
        "keep_streak": result.trends.keep_streak,
        "longest_keep_streak": result.trends.longest_keep_streak,
        "longest_discard_streak": result.trends.longest_discard_streak,
        "total_keeps": result.trends.total_keeps,
        "total_discards": result.trends.total_discards,
        "total_crashes": result.trends.total_crashes,
        "experiments_analyzed": result.trends.total_experiments,
        "best_val_bpb": result.trends.best_val_bpb,
        "improvement_rate_pct": (
            round(result.trends.improvement_rate_pct, 4)
            if result.trends.improvement_rate_pct is not None
            else None
        ),
        "suggestion": result.suggestion,
        "experiment": {
            "index": result.experiment.index,
            "commit": result.experiment.commit,
            "description": result.experiment.description,
            "val_bpb": result.experiment.val_bpb,
            "memory_gb": result.experiment.memory_gb,
            "status": result.experiment.status.value,
        },
        "efficiency": {
            "memory_gb": result.efficiency.memory_gb,
            "memory_delta_gb": round(result.efficiency.memory_delta_gb, 2),
            "num_params_m": result.efficiency.num_params_m,
            "bpb_per_param_m": (
                round(result.efficiency.bpb_per_param_m, 5)
                if result.efficiency.bpb_per_param_m is not None
                else None
            ),
        },
        "pareto_frontier": [
            {
                "index": p.index,
                "commit": p.commit,
                "val_bpb": p.val_bpb,
                "memory_gb": p.memory_gb,
            }
            for p in result.pareto_frontier
        ],
    }

    if result.run_log is not None:
        output["run_log"] = {
            "training_seconds": result.run_log.training_seconds,
            "total_seconds": result.run_log.total_seconds,
            "peak_vram_mb": result.run_log.peak_vram_mb,
            "mfu_percent": result.run_log.mfu_percent,
            "total_tokens_m": result.run_log.total_tokens_m,
            "num_steps": result.run_log.num_steps,
            "num_params_m": result.run_log.num_params_m,
            "depth": result.run_log.depth,
        }

    return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.command(epilog="Exit codes: 0 = success/keep, 1 = error, 2 = discard/crash verdict")
@click.version_option(version="1.0.1", prog_name="autojudge")
@click.option(
    "--results", "results_path",
    default="results.tsv",
    type=click.Path(),
    help="Path to results.tsv (default: results.tsv)",
)
@click.option(
    "--run-log", "run_log_path",
    default=None,
    type=click.Path(),
    help="Path to run.log for additional training metrics",
)
@click.option(
    "--format", "output_format",
    default="human",
    type=click.Choice(["human", "json"], case_sensitive=False),
    help="Output format (default: human)",
)
@click.option(
    "--no-color", "no_color",
    is_flag=True,
    default=False,
    help="Disable colored output",
)
@click.option(
    "--quiet", "-q", "quiet",
    is_flag=True,
    default=False,
    help="Minimal output (one line)",
)
def cli(
    results_path: str,
    run_log_path: Optional[str],
    output_format: str,
    no_color: bool,
    quiet: bool,
) -> None:
    """Smarter experiment evaluation for autoresearch.

    Analyzes the latest result in context of the full experiment history.
    """
    cfg = OutputConfig(color=not no_color and sys.stdout.isatty(), quiet=quiet)
    # Parse results
    results_result = parse_results_tsv(Path(results_path))
    if isinstance(results_result, Err):
        click.echo(f"Error: {results_result.error.message}", err=True)
        sys.exit(1)

    experiments: list[Experiment] = results_result.value

    # Parse optional run log
    run_log: Optional[RunLogMetrics] = None
    if run_log_path is not None:
        run_log_result = parse_run_log(Path(run_log_path))
        if isinstance(run_log_result, Err):
            click.echo(
                f"Warning: Could not parse run log: {run_log_result.error.message}",
                err=True,
            )
        else:
            run_log = run_log_result.value

    # Analyze
    analysis_result = analyze(experiments, run_log)
    if isinstance(analysis_result, Err):
        click.echo(f"Error: {analysis_result.error.message}", err=True)
        sys.exit(1)

    judgment: JudgmentResult = analysis_result.value

    # Output
    if output_format == "json":
        click.echo(format_json(judgment))
    elif cfg.quiet:
        click.echo(format_quiet(judgment))
    else:
        click.echo(format_human(judgment, cfg))

    # Exit with non-zero status for DISCARD/CRASH to support scripting
    if judgment.verdict in (Verdict.DISCARD, Verdict.CRASH):
        sys.exit(2)


def main() -> None:
    """Entry point for the autojudge CLI."""
    cli()


if __name__ == "__main__":
    main()
