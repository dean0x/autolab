"""
auto-steer: Research direction generator for autoresearch.

Analyzes experiment history (results.tsv + git diffs) and generates
smart next-step suggestions for the research agent.

Companion tool for https://github.com/karpathy/autoresearch
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import click

# ---------------------------------------------------------------------------
# Output infrastructure
# ---------------------------------------------------------------------------

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

class Category(Enum):
    ARCHITECTURE = "architecture"
    HYPERPARAMS = "hyperparams"
    OPTIMIZER = "optimizer"
    REGULARIZATION = "regularization"
    ACTIVATION = "activation"
    EMBEDDING = "embedding"
    EFFICIENCY = "efficiency"
    OTHER = "other"


class Status(Enum):
    KEEP = "keep"
    DISCARD = "discard"
    CRASH = "crash"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SuggestionKind(Enum):
    EXPLORE = "EXPLORE"
    EXPLOIT = "EXPLOIT"


class Strategy(Enum):
    AUTO = "auto"
    EXPLORE = "explore"
    EXPLOIT = "exploit"


@dataclass(frozen=True)
class Experiment:
    commit: str
    val_bpb: float
    memory_gb: float
    status: Status
    description: str
    category: Category = Category.OTHER
    diff_text: str = ""


@dataclass(frozen=True)
class CategoryStats:
    category: Category
    total: int
    keeps: int
    discards: int
    crashes: int
    avg_improvement_pct: float  # average % improvement over baseline when kept

    @property
    def success_rate_pct(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.keeps / self.total) * 100.0


@dataclass(frozen=True)
class Suggestion:
    rank: int
    kind: SuggestionKind
    title: str
    category: Category
    risk: RiskLevel
    expected_range: tuple[float, float]  # (low%, high%) improvement
    reasoning: str
    priority_score: float  # internal score for ranking


@dataclass(frozen=True)
class AnalysisResult:
    experiments: list[Experiment]
    stats_by_category: dict[Category, CategoryStats]
    suggestions: list[Suggestion]
    strategy_label: str
    baseline_bpb: Optional[float]
    best_bpb: Optional[float]


# ---------------------------------------------------------------------------
# Known good directions — the knowledge base
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KnownDirection:
    category: Category
    title: str
    description: str
    risk: RiskLevel
    expected_range: tuple[float, float]
    keywords: tuple[str, ...]  # used to check if already tried


KNOWN_DIRECTIONS: tuple[KnownDirection, ...] = (
    # Architecture
    KnownDirection(
        category=Category.ARCHITECTURE,
        title="Adjust depth/width ratio",
        description=(
            "Current ASPECT_RATIO=64 gives dim=512 at depth=8. Try ASPECT_RATIO=96 "
            "for a wider but same-depth model, or DEPTH=12 with ASPECT_RATIO=48 "
            "for deeper but narrower. Width tends to help more on short training runs."
        ),
        risk=RiskLevel.MEDIUM,
        expected_range=(0.3, 0.8),
        keywords=("aspect_ratio", "depth", "width", "model_dim"),
    ),
    KnownDirection(
        category=Category.ARCHITECTURE,
        title="Modify GQA head configuration",
        description=(
            "Adjust HEAD_DIM or number of KV heads. Fewer KV heads (more aggressive "
            "GQA) reduces memory and may allow a larger model. Try HEAD_DIM=64 for "
            "more heads or HEAD_DIM=256 for fewer."
        ),
        risk=RiskLevel.MEDIUM,
        expected_range=(0.2, 0.6),
        keywords=("head_dim", "gqa", "kv_head", "num_head", "query_head"),
    ),
    KnownDirection(
        category=Category.ARCHITECTURE,
        title="Change sliding window pattern",
        description=(
            "Current WINDOW_PATTERN='SSSL'. Try 'SSLL' for more full-attention layers, "
            "or 'SLSL' for alternating. More full-attention layers capture longer "
            "dependencies at the cost of memory."
        ),
        risk=RiskLevel.LOW,
        expected_range=(0.1, 0.4),
        keywords=("window_pattern", "sliding window", "window", "attention pattern"),
    ),
    KnownDirection(
        category=Category.ARCHITECTURE,
        title="Adjust MLP expansion ratio",
        description=(
            "Default MLP is 4x expansion. Try 3x to free parameters for more layers, "
            "or 8/3x (~2.67x) which is the SwiGLU-optimal ratio. The freed parameters "
            "can go toward extra depth."
        ),
        risk=RiskLevel.MEDIUM,
        expected_range=(0.2, 0.5),
        keywords=("mlp", "expansion", "ffn", "feed_forward", "intermediate_size"),
    ),
    # Hyperparams
    KnownDirection(
        category=Category.HYPERPARAMS,
        title="Tune matrix learning rate",
        description=(
            "MATRIX_LR has the most leverage on Muon's behavior. Try MATRIX_LR=0.06 "
            "(50% increase) or MATRIX_LR=0.02 (50% decrease). Muon is sensitive to "
            "this — small changes can have outsized impact."
        ),
        risk=RiskLevel.LOW,
        expected_range=(0.3, 0.5),
        keywords=("matrix_lr", "learning_rate", "muon_lr"),
    ),
    KnownDirection(
        category=Category.HYPERPARAMS,
        title="Tune embedding learning rates",
        description=(
            "EMBEDDING_LR=0.6 and UNEMBEDDING_LR=0.004 are quite asymmetric. Try "
            "reducing EMBEDDING_LR to 0.3 or increasing UNEMBEDDING_LR to 0.008. "
            "The large gap suggests room for tuning."
        ),
        risk=RiskLevel.LOW,
        expected_range=(0.1, 0.3),
        keywords=("embedding_lr", "unembedding_lr", "embed"),
    ),
    KnownDirection(
        category=Category.HYPERPARAMS,
        title="Add warmup schedule",
        description=(
            "Currently WARMUP_RATIO=0.0 (no warmup). Try WARMUP_RATIO=0.05 for "
            "5% warmup — helps with training stability early on, especially with "
            "large learning rates."
        ),
        risk=RiskLevel.LOW,
        expected_range=(0.1, 0.3),
        keywords=("warmup", "warmup_ratio"),
    ),
    KnownDirection(
        category=Category.HYPERPARAMS,
        title="Adjust warmdown schedule",
        description=(
            "Currently WARMDOWN_RATIO=0.5. Try 0.3 for less cooldown (more time at "
            "peak LR) or 0.7 for more gradual decay. With a 5-minute budget, the "
            "warmdown schedule significantly affects final performance."
        ),
        risk=RiskLevel.LOW,
        expected_range=(0.1, 0.4),
        keywords=("warmdown", "warmdown_ratio", "cooldown", "final_lr"),
    ),
    KnownDirection(
        category=Category.HYPERPARAMS,
        title="Increase batch size",
        description=(
            "Current TOTAL_BATCH_SIZE=2**19 (~524K tokens). Try 2**20 (~1M tokens) "
            "for smoother gradients. Larger batches can help with short training runs "
            "by reducing gradient noise. May need to reduce model size to fit."
        ),
        risk=RiskLevel.MEDIUM,
        expected_range=(0.2, 0.5),
        keywords=("batch_size", "total_batch", "device_batch"),
    ),
    # Optimizer
    KnownDirection(
        category=Category.OPTIMIZER,
        title="Tune Muon orthogonalization steps",
        description=(
            "Muon uses Newton-Schulz iterations for polar decomposition (typically "
            "ns_steps=5). Try ns_steps=6 for better orthogonalization quality, or "
            "ns_steps=4 to save compute and allow a slightly larger model."
        ),
        risk=RiskLevel.LOW,
        expected_range=(0.1, 0.3),
        keywords=("ns_steps", "newton", "schulz", "orthogonal", "polar"),
    ),
    KnownDirection(
        category=Category.OPTIMIZER,
        title="Adjust Adam betas",
        description=(
            "Current ADAM_BETAS=(0.8, 0.95). Try (0.9, 0.95) for more momentum "
            "(standard Adam default) or (0.8, 0.99) for longer gradient memory. "
            "Beta1=0.8 is already aggressive — going higher may stabilize training."
        ),
        risk=RiskLevel.LOW,
        expected_range=(0.1, 0.3),
        keywords=("adam_beta", "beta1", "beta2", "momentum"),
    ),
    KnownDirection(
        category=Category.OPTIMIZER,
        title="Tune weight decay",
        description=(
            "Current WEIGHT_DECAY=0.2. Try 0.1 for less regularization (may help "
            "with short training runs where overfitting is not an issue) or 0.3 "
            "for stronger regularization."
        ),
        risk=RiskLevel.LOW,
        expected_range=(0.1, 0.3),
        keywords=("weight_decay",),
    ),
    # Regularization
    KnownDirection(
        category=Category.REGULARIZATION,
        title="Add z-loss regularization",
        description=(
            "Add a small penalty on the log of the softmax partition function. "
            "This stabilizes training and prevents logit drift. Typical coefficient "
            "is 1e-4. Used in PaLM and other large LMs."
        ),
        risk=RiskLevel.LOW,
        expected_range=(0.1, 0.3),
        keywords=("z_loss", "z-loss", "partition", "logit_reg"),
    ),
    KnownDirection(
        category=Category.REGULARIZATION,
        title="Adjust softcap value",
        description=(
            "The current logit softcap prevents extreme values. Try increasing it "
            "(less capping, more expressivity) or decreasing it (more regularization). "
            "Small changes can affect training dynamics significantly."
        ),
        risk=RiskLevel.LOW,
        expected_range=(0.1, 0.3),
        keywords=("softcap",),
    ),
    # Activation
    KnownDirection(
        category=Category.ACTIVATION,
        title="Try SwiGLU activation",
        description=(
            "Replace relu().square() MLP with SwiGLU (gate * silu(x)). Common in "
            "modern LLMs (LLaMA, Gemma). Requires adjusting MLP dimensions to 8/3 "
            "ratio instead of 4x to keep param count similar. Typically gives a "
            "meaningful quality boost."
        ),
        risk=RiskLevel.MEDIUM,
        expected_range=(0.5, 1.0),
        keywords=("swiglu", "silu", "glu", "gated"),
    ),
    KnownDirection(
        category=Category.ACTIVATION,
        title="Try GELU activation",
        description=(
            "Replace ReSquared (relu^2) with GELU. GELU is the standard GPT "
            "activation. ReSquared may underperform GELU in some regimes, though "
            "it has theoretical advantages for feature learning."
        ),
        risk=RiskLevel.MEDIUM,
        expected_range=(0.3, 0.7),
        keywords=("gelu",),
    ),
    # Embedding
    KnownDirection(
        category=Category.EMBEDDING,
        title="Adjust value embedding frequency",
        description=(
            "Value embeddings (ResFormer) currently apply at alternating layers. "
            "Try applying at every layer (more capacity but more parameters) or "
            "every 3rd layer (fewer parameters, may allow larger model)."
        ),
        risk=RiskLevel.LOW,
        expected_range=(0.1, 0.3),
        keywords=("value_embed", "resformer", "embedding_freq", "embed_layer"),
    ),
    KnownDirection(
        category=Category.EMBEDDING,
        title="Tune RoPE base frequency",
        description=(
            "Adjust the RoPE base frequency (theta). Higher theta extends effective "
            "context but may hurt short-range performance. Lower theta sharpens "
            "attention within the local window."
        ),
        risk=RiskLevel.LOW,
        expected_range=(0.1, 0.3),
        keywords=("rope", "theta", "rotary", "base_freq"),
    ),
    # Efficiency
    KnownDirection(
        category=Category.EFFICIENCY,
        title="Optimize memory for larger model",
        description=(
            "Use gradient checkpointing, activation recomputation, or mixed precision "
            "to free VRAM. The saved memory can be reinvested into a larger model "
            "(more layers or wider). Even a small model size increase can improve "
            "val_bpb substantially."
        ),
        risk=RiskLevel.MEDIUM,
        expected_range=(0.3, 0.8),
        keywords=("checkpoint", "recompute", "memory", "vram", "mixed_precision"),
    ),
    KnownDirection(
        category=Category.EFFICIENCY,
        title="Reduce compilation overhead",
        description=(
            "If torch.compile takes significant time, try compiling fewer components "
            "or using mode='reduce-overhead'. The saved startup time becomes training "
            "time within the 5-minute budget."
        ),
        risk=RiskLevel.LOW,
        expected_range=(0.1, 0.3),
        keywords=("compile", "compilation", "torch.compile", "startup"),
    ),
)


# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------

# Keywords for classifying experiment descriptions and diffs into categories.
# Checked in priority order — first match wins.
_CATEGORY_KEYWORDS: tuple[tuple[Category, tuple[str, ...]], ...] = (
    (Category.ACTIVATION, (
        "activation", "relu", "gelu", "silu", "swiglu", "squared",
        "resquared", "relu^2", "glu",
    )),
    (Category.OPTIMIZER, (
        "optimizer", "muon", "adam", "sgd", "momentum", "ns_step",
        "newton", "schulz", "polar", "orthogonal",
    )),
    (Category.REGULARIZATION, (
        "dropout", "z-loss", "z_loss", "softcap", "regulariz",
        "weight_decay", "weight decay",
    )),
    (Category.EMBEDDING, (
        "embedding", "embed", "token embed", "resformer", "rope",
        "rotary", "vocab", "tokeniz", "unembedding",
    )),
    (Category.EFFICIENCY, (
        "memory", "vram", "compile", "checkpoint", "recompute",
        "mixed precision", "fp16", "bf16", "flash", "fused",
    )),
    (Category.HYPERPARAMS, (
        "lr", "learning rate", "batch_size", "batch size", "warmup",
        "warmdown", "cooldown", "schedule", "final_lr",
    )),
    (Category.ARCHITECTURE, (
        "layer", "depth", "width", "head", "attention", "mlp",
        "ffn", "aspect_ratio", "model_dim", "gqa", "window",
        "transformer", "block", "residual",
    )),
)


def classify_experiment(description: str, diff_text: str) -> Category:
    """Classify an experiment into a category based on its description and diff."""
    combined = (description + " " + diff_text).lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        for keyword in keywords:
            # Use word boundary matching for short keywords to avoid false positives
            # e.g. "lr" matching "clearly", "head" matching "overhead"
            if len(keyword) <= 3:
                if re.search(r'(?:^|[\s_])' + re.escape(keyword) + r'(?:[\s_=,.]|$)', combined):
                    return category
            else:
                if keyword in combined:
                    return category
    return Category.OTHER


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParseResult:
    ok: bool
    experiments: list[Experiment]
    error: str = ""


def parse_results_tsv(path: Path) -> ParseResult:
    """Parse a results.tsv file into a list of Experiment records."""
    if not path.exists():
        return ParseResult(ok=False, experiments=[], error=f"File not found: {path}")

    text = path.read_text().strip()
    if not text:
        return ParseResult(ok=False, experiments=[], error=f"Empty file: {path}")

    lines = text.splitlines()
    if len(lines) < 2:
        return ParseResult(
            ok=False, experiments=[],
            error="Only header row found — no experiments yet",
        )

    reader = csv.DictReader(lines, delimiter="\t")
    required_fields = {"commit", "val_bpb", "status", "description"}
    if reader.fieldnames is None or not required_fields.issubset(set(reader.fieldnames)):
        missing = required_fields - set(reader.fieldnames or [])
        return ParseResult(
            ok=False, experiments=[],
            error=f"Missing columns in results.tsv: {missing}",
        )

    experiments: list[Experiment] = []
    for row in reader:
        try:
            status = Status(row["status"].strip().lower())
        except ValueError:
            status = Status.DISCARD

        try:
            val_bpb = float(row["val_bpb"])
        except (ValueError, KeyError):
            val_bpb = 0.0

        try:
            memory_gb = float(row.get("memory_gb", "0.0"))
        except ValueError:
            memory_gb = 0.0

        experiments.append(Experiment(
            commit=row["commit"].strip(),
            val_bpb=val_bpb,
            memory_gb=memory_gb,
            status=status,
            description=row["description"].strip(),
        ))

    return ParseResult(ok=True, experiments=experiments)


# ---------------------------------------------------------------------------
# Git integration
# ---------------------------------------------------------------------------

def get_git_diff(commit: str, repo_dir: Path) -> str:
    """Get the diff for a given commit hash using git."""
    try:
        result = subprocess.run(
            ["git", "diff", f"{commit}~1", commit, "--", "train.py"],
            capture_output=True,
            text=True,
            cwd=str(repo_dir),
            timeout=10,
        )
        return result.stdout if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


_git_repo_warning_shown = False

def enrich_experiments_with_git(
    experiments: list[Experiment],
    repo_dir: Path,
) -> list[Experiment]:
    """Add git diff text and category classification to each experiment."""
    global _git_repo_warning_shown

    # Check if we're in a git repo
    if not _git_repo_warning_shown:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True, text=True,
                cwd=str(repo_dir), timeout=5,
            )
            if result.returncode != 0:
                click.echo(
                    "Note: not in a git repo, category classification based on descriptions only",
                    err=True,
                )
                _git_repo_warning_shown = True
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            click.echo(
                "Note: git not available, category classification based on descriptions only",
                err=True,
            )
            _git_repo_warning_shown = True

    enriched: list[Experiment] = []
    for exp in experiments:
        diff_text = get_git_diff(exp.commit, repo_dir) if exp.commit else ""
        category = classify_experiment(exp.description, diff_text)
        enriched.append(Experiment(
            commit=exp.commit,
            val_bpb=exp.val_bpb,
            memory_gb=exp.memory_gb,
            status=exp.status,
            description=exp.description,
            category=category,
            diff_text=diff_text,
        ))
    return enriched


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_category_stats(
    experiments: list[Experiment],
    baseline_bpb: Optional[float],
) -> dict[Category, CategoryStats]:
    """Compute success rates and average improvements by category."""
    stats: dict[Category, CategoryStats] = {}
    by_cat: dict[Category, list[Experiment]] = {}

    # Skip the baseline (first experiment)
    for exp in experiments[1:]:
        by_cat.setdefault(exp.category, []).append(exp)

    for category in Category:
        cat_exps = by_cat.get(category, [])
        keeps = [e for e in cat_exps if e.status == Status.KEEP]
        discards = [e for e in cat_exps if e.status == Status.DISCARD]
        crashes = [e for e in cat_exps if e.status == Status.CRASH]

        avg_improvement = 0.0
        if keeps and baseline_bpb and baseline_bpb > 0:
            improvements = [
                ((baseline_bpb - e.val_bpb) / baseline_bpb) * 100.0
                for e in keeps
                if e.val_bpb > 0  # only valid bpb values for improvement calc
            ]
            avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0

        stats[category] = CategoryStats(
            category=category,
            total=len(cat_exps),
            keeps=len(keeps),
            discards=len(discards),
            crashes=len(crashes),
            avg_improvement_pct=avg_improvement,
        )

    return stats


def _direction_already_tried(
    direction: KnownDirection,
    experiments: list[Experiment],
) -> bool:
    """Check if a known direction has already been tried based on keyword matching."""
    for exp in experiments:
        combined = (exp.description + " " + exp.diff_text).lower().replace("_", " ")
        normalized_keywords = [kw.replace("_", " ") for kw in direction.keywords]
        matches = sum(1 for kw in normalized_keywords if kw in combined)
        # Require at least 2 keyword matches (or 1 if the direction only has 1 keyword)
        threshold = min(2, len(direction.keywords))
        if matches >= threshold:
            return True
    return False


def _compute_priority_score(
    direction: KnownDirection,
    stats: dict[Category, CategoryStats],
    strategy: Strategy,
    total_experiments: int,
) -> float:
    """Compute a priority score for a suggestion based on strategy and history."""
    cat_stats = stats.get(direction.category, CategoryStats(
        category=direction.category, total=0, keeps=0,
        discards=0, crashes=0, avg_improvement_pct=0.0,
    ))

    # Base score from expected value (midpoint of expected range)
    expected_midpoint = (direction.expected_range[0] + direction.expected_range[1]) / 2.0
    score = expected_midpoint

    # Risk adjustment
    risk_multiplier = {RiskLevel.LOW: 1.2, RiskLevel.MEDIUM: 1.0, RiskLevel.HIGH: 0.7}
    score *= risk_multiplier[direction.risk]

    # Strategy adjustments
    category_tried = cat_stats.total > 0
    category_has_success = cat_stats.keeps > 0

    if strategy == Strategy.EXPLOIT:
        if category_has_success:
            # Boost categories that have worked before
            score *= 1.5 + (cat_stats.success_rate_pct / 100.0)
        elif category_tried:
            # Penalize categories that have been tried and failed
            score *= 0.3
        else:
            # Untried categories get moderate penalty in exploit mode
            score *= 0.5
    elif strategy == Strategy.EXPLORE:
        if not category_tried:
            # Boost untried categories
            score *= 2.0
        elif category_tried and not category_has_success:
            # Even failed categories get a second look in explore mode
            score *= 0.8
        else:
            # Already-successful categories get slight penalty (explore new ground)
            score *= 0.7
    else:
        # AUTO: balance based on experiment count
        exploration_weight = max(0.2, 1.0 - (total_experiments / 20.0))
        exploitation_weight = 1.0 - exploration_weight

        if not category_tried:
            score *= 1.0 + exploration_weight
        elif category_has_success:
            success_bonus = cat_stats.success_rate_pct / 100.0
            score *= 1.0 + (exploitation_weight * success_bonus)
        else:
            score *= 0.5

    return score


def generate_suggestions(
    experiments: list[Experiment],
    stats: dict[Category, CategoryStats],
    strategy: Strategy,
    num_suggestions: int,
) -> list[Suggestion]:
    """Generate ranked experiment suggestions based on history and strategy."""
    total_experiments = sum(s.total for s in stats.values())

    scored: list[tuple[KnownDirection, float, SuggestionKind]] = []
    for direction in KNOWN_DIRECTIONS:
        if _direction_already_tried(direction, experiments):
            continue

        priority = _compute_priority_score(direction, stats, strategy, total_experiments)

        # Determine suggestion kind
        cat_stats = stats.get(direction.category)
        if cat_stats and cat_stats.keeps > 0:
            kind = SuggestionKind.EXPLOIT
        else:
            kind = SuggestionKind.EXPLORE

        scored.append((direction, priority, kind))

    # Sort by priority descending
    scored.sort(key=lambda x: x[1], reverse=True)

    suggestions: list[Suggestion] = []
    for rank, (direction, priority, kind) in enumerate(scored[:num_suggestions], start=1):
        suggestions.append(Suggestion(
            rank=rank,
            kind=kind,
            title=direction.title,
            category=direction.category,
            risk=direction.risk,
            expected_range=direction.expected_range,
            reasoning=direction.description,
            priority_score=priority,
        ))

    return suggestions


def _resolve_strategy_label(strategy: Strategy, total_experiments: int) -> str:
    """Generate a human-readable label for the chosen strategy."""
    if strategy == Strategy.EXPLORE:
        return "explore (favoring new directions)"
    if strategy == Strategy.EXPLOIT:
        return "exploit (favoring what works)"

    # Auto mode: describe the balance
    if total_experiments <= 3:
        return f"auto (balanced — {total_experiments} experiments so far, favoring exploration)"
    elif total_experiments <= 10:
        return f"auto (balanced — {total_experiments} experiments so far, mixed strategy)"
    else:
        return f"auto (balanced — {total_experiments} experiments so far, favoring exploitation)"


def analyze(
    results_path: Path,
    repo_dir: Path,
    strategy: Strategy,
    num_suggestions: int,
) -> AnalysisResult:
    """Run the full analysis pipeline."""
    parsed = parse_results_tsv(results_path)
    if not parsed.ok:
        # File missing or malformed — still generate suggestions with empty history
        empty_stats = {c: CategoryStats(c, 0, 0, 0, 0, 0.0) for c in Category}
        strategy_label = _resolve_strategy_label(strategy, 0)
        suggestions = generate_suggestions([], empty_stats, strategy, num_suggestions)
        return AnalysisResult(
            experiments=[],
            stats_by_category=empty_stats,
            suggestions=suggestions,
            strategy_label=f"{strategy_label} (note: {parsed.error})",
            baseline_bpb=None,
            best_bpb=None,
        )

    experiments = enrich_experiments_with_git(parsed.experiments, repo_dir)

    # Determine baseline (first experiment, typically status=keep)
    baseline_bpb: Optional[float] = None
    best_bpb: Optional[float] = None
    if experiments:
        baseline_bpb = experiments[0].val_bpb if experiments[0].val_bpb > 0 else None
        kept_bpbs = [e.val_bpb for e in experiments if e.status == Status.KEEP and e.val_bpb > 0]
        best_bpb = min(kept_bpbs) if kept_bpbs else baseline_bpb

    stats = compute_category_stats(experiments, baseline_bpb)
    total_experiments = sum(s.total for s in stats.values())
    strategy_label = _resolve_strategy_label(strategy, total_experiments)

    suggestions = generate_suggestions(experiments, stats, strategy, num_suggestions)

    return AnalysisResult(
        experiments=experiments,
        stats_by_category=stats,
        suggestions=suggestions,
        strategy_label=strategy_label,
        baseline_bpb=baseline_bpb,
        best_bpb=best_bpb,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_quiet(result: AnalysisResult) -> str:
    """Minimal output: numbered titles only."""
    lines: list[str] = []
    for s in result.suggestions:
        lines.append(f"{s.rank}. [{s.kind.value}] {s.title}")
    return "\n".join(lines) if lines else "No suggestions available."


def format_text(result: AnalysisResult, cfg: OutputConfig) -> str:
    """Format the analysis result as human-readable text with optional color."""
    lines: list[str] = []
    lines.append(f"== {cfg.styled('autosteer', fg='cyan', bold=True)} suggestions ==")
    lines.append(f"Strategy: {result.strategy_label}")

    if result.baseline_bpb is not None:
        lines.append(f"Baseline val_bpb: {result.baseline_bpb:.6f}")
    if result.best_bpb is not None and result.best_bpb != result.baseline_bpb:
        lines.append(f"Best val_bpb:     {result.best_bpb:.6f}")

    # Category stats
    lines.append("")
    lines.append(cfg.styled("Based on history:", dim=True))
    has_any_experiments = False
    for category in Category:
        stats = result.stats_by_category.get(category)
        if stats is None:
            continue
        if stats.total == 0:
            continue
        has_any_experiments = True
        cat_label = cfg.styled(category.value, fg="blue")
        success_str = f"{stats.success_rate_pct:.0f}% success"
        if stats.avg_improvement_pct > 0:
            success_str += f", avg +{stats.avg_improvement_pct:.2f}%"
        if cfg.color:
            lines.append(
                f"  {cat_label}: {stats.total} tried, "
                f"{stats.keeps} kept ({success_str})"
            )
        else:
            lines.append(
                f"  {category.value + ':':18s} {stats.total} tried, "
                f"{stats.keeps} kept ({success_str})"
            )

    # Show untried categories
    untried = [
        c for c in Category
        if result.stats_by_category.get(c) is not None
        and result.stats_by_category[c].total == 0
    ]
    if untried:
        for cat in untried:
            if cfg.color:
                lines.append(f"  {cfg.styled(cat.value, fg='blue')}: 0 tried")
            else:
                lines.append(f"  {cat.value + ':':18s} 0 tried")

    if not has_any_experiments and not untried:
        lines.append("  (no experiments recorded yet)")

    # Suggestions
    lines.append("")
    if not result.suggestions:
        lines.append("No suggestions available — all known directions have been tried.")
        lines.append("Consider exploring novel ideas outside the standard playbook.")
    else:
        lines.append(cfg.styled("Suggestions (ranked by expected value):", dim=True))
        lines.append("")

        risk_colors = {"low": "green", "medium": "yellow", "high": "red"}

        for s in result.suggestions:
            kind_color = "cyan" if s.kind == SuggestionKind.EXPLORE else "green"
            kind_badge = cfg.styled(f"[{s.kind.value}]", fg=kind_color)
            rank_str = cfg.styled(f"{s.rank}.", bold=True)
            lines.append(f"{rank_str} {kind_badge} {s.title}")

            risk_color = risk_colors.get(s.risk.value, "white")
            cat_str = cfg.styled(s.category.value, fg="blue")
            risk_str = cfg.styled(s.risk.value, fg=risk_color)
            lines.append(
                f"   Category: {cat_str} | "
                f"Risk: {risk_str} | "
                f"Expected: +{s.expected_range[0]:.1f}-{s.expected_range[1]:.1f}%"
            )
            wrapped = _wrap_text(s.reasoning, width=72, indent="   ")
            lines.append(wrapped)
            lines.append("")

    return "\n".join(lines)


def _wrap_text(text: str, width: int, indent: str) -> str:
    """Word-wrap with indent prefix for all lines."""
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def format_json(result: AnalysisResult) -> str:
    """Format the analysis result as JSON."""
    data = {
        "strategy": result.strategy_label,
        "baseline_bpb": result.baseline_bpb,
        "best_bpb": result.best_bpb,
        "category_stats": {
            cat.value: {
                "total": stats.total,
                "keeps": stats.keeps,
                "discards": stats.discards,
                "crashes": stats.crashes,
                "success_rate_pct": round(stats.success_rate_pct, 1),
                "avg_improvement_pct": round(stats.avg_improvement_pct, 2),
            }
            for cat, stats in result.stats_by_category.items()
        },
        "suggestions": [
            {
                "rank": s.rank,
                "kind": s.kind.value,
                "title": s.title,
                "category": s.category.value,
                "risk": s.risk.value,
                "expected_range_pct": [s.expected_range[0], s.expected_range[1]],
                "reasoning": s.reasoning,
                "priority_score": round(s.priority_score, 3),
            }
            for s in result.suggestions
        ],
    }

    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command(epilog="Exit codes: 0 = success, 1 = file error")
@click.version_option(version="1.0.0", prog_name="autosteer")
@click.option(
    "--results", "results_path",
    default="results.tsv",
    type=click.Path(),
    help="Path to results.tsv file (default: results.tsv in current dir)",
)
@click.option(
    "--repo-dir", "repo_dir",
    default=".",
    type=click.Path(exists=True, file_okay=False),
    help="Path to the autoresearch repo (default: current dir)",
)
@click.option(
    "--num-suggestions", "num_suggestions",
    default=5,
    type=click.IntRange(min=1, max=20),
    help="Number of suggestions to generate (default: 5)",
)
@click.option(
    "--strategy",
    default="auto",
    type=click.Choice(["auto", "explore", "exploit"], case_sensitive=False),
    help="Strategy: auto (balanced), explore (new directions), exploit (what works)",
)
@click.option(
    "--format", "output_format",
    default="text",
    type=click.Choice(["text", "json"], case_sensitive=False),
    help="Output format (default: text)",
)
@click.option(
    "--no-color", "no_color",
    is_flag=True,
    default=False,
    help="Disable colored output",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    default=False,
    help="Minimal output (one line per suggestion)",
)
def cli(
    results_path: str,
    repo_dir: str,
    num_suggestions: int,
    strategy: str,
    output_format: str,
    no_color: bool,
    quiet: bool,
) -> None:
    """Analyze autoresearch experiment history and suggest next steps."""
    cfg = OutputConfig(color=not no_color and sys.stdout.isatty(), quiet=quiet)

    resolved_results = Path(results_path)
    if not resolved_results.is_absolute():
        resolved_results = Path(repo_dir) / resolved_results

    result = analyze(
        results_path=resolved_results,
        repo_dir=Path(repo_dir),
        strategy=Strategy(strategy.lower()),
        num_suggestions=num_suggestions,
    )

    if output_format == "json":
        click.echo(format_json(result))
    elif cfg.quiet:
        click.echo(format_quiet(result))
    else:
        click.echo(format_text(result, cfg))

    # Exit with error code if no experiments found and it was due to a file issue
    if not result.experiments and "not found" in result.strategy_label.lower():
        sys.exit(1)


def main() -> None:
    """Entry point for the auto-steer CLI."""
    cli()


if __name__ == "__main__":
    main()
