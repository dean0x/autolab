"""
auto-evolve: Multi-agent research competition orchestrator for autoresearch.

Manages multiple competing autoresearch agents on separate git branches,
with leaderboard tracking and cross-pollination of winning ideas.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Generic, Optional, TypeVar, Union

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
# Result type — all fallible operations return Result instead of raising
# ---------------------------------------------------------------------------

T = TypeVar("T")


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T

    @property
    def ok(self) -> bool:
        return True


@dataclass(frozen=True)
class Err:
    error: str

    @property
    def ok(self) -> bool:
        return False


Result = Union[Ok[T], Err]


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

STRATEGIES: list[dict[str, str]] = [
    {
        "key": "architecture-first",
        "label": "Architecture First",
        "guidance": (
            "Start by exploring model architecture (depth, width, attention patterns, "
            "MLP ratio). Once you find a good architecture, fine-tune hyperparams."
        ),
    },
    {
        "key": "hyperparams-first",
        "label": "Hyperparams First",
        "guidance": (
            "Start by sweeping hyperparameters (learning rates, batch size, "
            "warmup/cooldown). Find optimal training dynamics before changing architecture."
        ),
    },
    {
        "key": "optimizer-first",
        "label": "Optimizer First",
        "guidance": (
            "Start by tuning the optimizer (Muon momentum, ns_steps, AdamW betas, "
            "weight decay schedule). A well-tuned optimizer can unlock gains."
        ),
    },
    {
        "key": "regularization-first",
        "label": "Regularization First",
        "guidance": (
            "Start by exploring regularization (weight decay, dropout, z-loss, softcap "
            "values). Prevent overfitting before scaling up."
        ),
    },
    {
        "key": "efficiency-first",
        "label": "Efficiency First",
        "guidance": (
            "Start by maximizing compute efficiency (larger batch size, better memory "
            "usage, faster iteration). More experiments = more chances."
        ),
    },
    {
        "key": "radical",
        "label": "Radical",
        "guidance": (
            "Try bold, unconventional changes. Large architecture modifications, novel "
            "activation functions, unusual training schedules. Go big or go home."
        ),
    },
]


@dataclass(frozen=True)
class Experiment:
    """A single row from results.tsv."""
    commit: str
    val_bpb: float
    memory_gb: float
    status: str
    description: str


@dataclass
class AgentConfig:
    """Configuration for a single agent in the evolve."""
    id: int
    branch: str
    strategy: str
    status: str = "pending"


@dataclass
class EvolveConfig:
    """Root evolve state persisted to evolve.json."""
    tag: str
    base_branch: str
    base_commit: str
    created_at: str
    agents: list[AgentConfig] = field(default_factory=list)


@dataclass(frozen=True)
class AgentStatus:
    """Runtime status of an agent derived from its results.tsv."""
    agent: AgentConfig
    experiments: list[Experiment]
    best_val_bpb: Optional[float]
    best_experiment: Optional[Experiment]
    keep_count: int


# ---------------------------------------------------------------------------
# Git helpers — thin wrappers around subprocess
# ---------------------------------------------------------------------------

def _run_git(*args: str, check: bool = True, timeout: int = 30) -> Result[str]:
    """Run a git command and return stdout on success, or an Err on failure."""
    cmd = ["git"] + list(args)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
            timeout=timeout,
        )
        return Ok(proc.stdout.strip())
    except subprocess.TimeoutExpired:
        return Err(f"git {' '.join(args)} timed out after {timeout}s")
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else str(exc)
        return Err(f"git {' '.join(args)} failed: {stderr}")


def _git_branch_exists(branch: str) -> bool:
    """Check whether a local branch exists."""
    result = _run_git("rev-parse", "--verify", branch, check=False)
    return result.ok


def _git_current_branch() -> Result[str]:
    """Return the current branch name."""
    return _run_git("rev-parse", "--abbrev-ref", "HEAD")


def _git_head_sha() -> Result[str]:
    """Return the short SHA of HEAD."""
    return _run_git("rev-parse", "--short", "HEAD")


def _git_show_file(branch: str, path: str) -> Result[str]:
    """Read a file from a given branch without checking it out."""
    return _run_git("show", f"{branch}:{path}", check=False)


def _git_working_tree_clean() -> bool:
    """Check that the working tree has no uncommitted changes."""
    result = _run_git("status", "--porcelain", check=False)
    return result.ok and result.value.strip() == ""


def _git_diff_commits(commit_a: str, commit_b: str) -> Result[str]:
    """Get the diff between two commits."""
    return _run_git("diff", commit_a, commit_b)


def _git_log_oneline(branch: str, base_commit: str, max_count: int = 50) -> Result[str]:
    """Get one-line log of commits on branch since base_commit."""
    return _run_git(
        "log", "--oneline", f"--max-count={max_count}",
        f"{base_commit}..{branch}",
        check=False,
    )


# ---------------------------------------------------------------------------
# Evolve config persistence
# ---------------------------------------------------------------------------

EVOLVE_CONFIG_FILE = "evolve.json"


def _evolve_config_path() -> Path:
    """Return the path to evolve.json in the repo root."""
    result = _run_git("rev-parse", "--show-toplevel")
    if not result.ok:
        return Path(EVOLVE_CONFIG_FILE)
    return Path(result.value) / EVOLVE_CONFIG_FILE


def _load_evolve_config() -> Result[EvolveConfig]:
    """Load evolve.json from the repo root."""
    path = _evolve_config_path()
    if not path.exists():
        return Err(
            f"No evolve config found at {path}. "
            "Run 'autoevolve init' first."
        )
    try:
        raw = json.loads(path.read_text())
        agents = [
            AgentConfig(
                id=a["id"],
                branch=a["branch"],
                strategy=a["strategy"],
                status=a.get("status", "pending"),
            )
            for a in raw.get("agents", [])
        ]
        return Ok(EvolveConfig(
            tag=raw["tag"],
            base_branch=raw["base_branch"],
            base_commit=raw["base_commit"],
            created_at=raw["created_at"],
            agents=agents,
        ))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as exc:
        return Err(f"Corrupt evolve.json: {exc}")


def _save_evolve_config(config: EvolveConfig) -> Result[None]:
    """Persist evolve state to evolve.json atomically (not committed to git)."""
    path = _evolve_config_path()
    data = {
        "tag": config.tag,
        "base_branch": config.base_branch,
        "base_commit": config.base_commit,
        "created_at": config.created_at,
        "agents": [asdict(a) for a in config.agents],
    }
    try:
        content = json.dumps(data, indent=2) + "\n"
        tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                f.write(content)
            Path(tmp_path).replace(path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        return Ok(None)
    except OSError as exc:
        return Err(f"Failed to write evolve.json: {exc}")


# ---------------------------------------------------------------------------
# Results.tsv parsing
# ---------------------------------------------------------------------------

def _parse_results_tsv(raw: str) -> list[Experiment]:
    """Parse a results.tsv string into a list of Experiment records.

    Expected header: commit\tval_bpb\tmemory_gb\tstatus\tdescription
    """
    lines = raw.strip().splitlines()
    experiments: list[Experiment] = []

    for idx, line in enumerate(lines):
        # Skip empty lines, and skip the first line if it looks like a header
        if not line.strip():
            continue
        if idx == 0 and line.strip().startswith("commit"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        try:
            experiments.append(Experiment(
                commit=parts[0].strip(),
                val_bpb=float(parts[1].strip()),
                memory_gb=float(parts[2].strip()),
                status=parts[3].strip(),
                description="\t".join(parts[4:]).strip(),
            ))
        except (ValueError, IndexError):
            # Skip malformed rows
            continue

    return experiments


def _read_results_for_agent(agent: AgentConfig) -> str:
    """Read results.tsv for an agent, trying git first then filesystem."""
    # Try git show first (works if results.tsv is committed)
    result = _git_show_file(agent.branch, "results.tsv")
    if result.ok and result.value.strip():
        return result.value

    # Fall back to filesystem if this branch is currently checked out
    current = _git_current_branch()
    if current.ok and current.value == agent.branch:
        repo_root = _run_git("rev-parse", "--show-toplevel")
        if repo_root.ok:
            results_path = Path(repo_root.value) / "results.tsv"
            if results_path.exists():
                return results_path.read_text()

    return ""


def _get_agent_status(agent: AgentConfig) -> AgentStatus:
    """Read results.tsv from an agent's branch and compute status."""
    raw = _read_results_for_agent(agent)
    if not raw.strip():
        return AgentStatus(
            agent=agent,
            experiments=[],
            best_val_bpb=None,
            best_experiment=None,
            keep_count=0,
        )

    experiments = _parse_results_tsv(raw)
    keeps = [e for e in experiments if e.status == "keep"]
    valid_keeps = [e for e in experiments if e.status == "keep" and e.val_bpb > 0]
    best = min(valid_keeps, key=lambda e: e.val_bpb) if valid_keeps else None

    return AgentStatus(
        agent=agent,
        experiments=experiments,
        best_val_bpb=best.val_bpb if best else None,
        best_experiment=best,
        keep_count=len(keeps),
    )


def _compute_improvements(experiments: list[Experiment]) -> list[tuple[Experiment, float]]:
    """Find impactful keep experiments by comparing each to the previous best.

    Each keep is compared against the best val_bpb among all preceding keeps,
    avoiding comparisons to crashed experiments (val_bpb=0.0) or discards.
    """
    improvements: list[tuple[Experiment, float]] = []
    prev_best: Optional[float] = None
    for exp in experiments:
        if exp.status != "keep" or exp.val_bpb <= 0:
            continue
        if prev_best is not None:
            delta = prev_best - exp.val_bpb
            if delta > 0:
                improvements.append((exp, delta))
        if prev_best is None or exp.val_bpb < prev_best:
            prev_best = exp.val_bpb
    return improvements


# ---------------------------------------------------------------------------
# Program.md generation
# ---------------------------------------------------------------------------

def _generate_program_md(strategy: dict[str, str], agent_id: int, tag: str) -> str:
    """Generate a program.md variant for an agent with a specific research strategy."""
    return f"""\
# Autoresearch Program — Evolve {tag}, Agent {agent_id}

## Strategy: {strategy['label']}

{strategy['guidance']}

## Rules

1. Modify `train.py` and commit your changes with a clear description.
2. Run `uv run train.py > run.log 2>&1` — training runs for exactly 5 minutes.
3. Read results: `grep "^val_bpb:\\|^peak_vram_mb:" run.log`
4. Record results in `results.tsv` (tab-separated):
   - commit (short hash), val_bpb, memory_gb (peak_vram_mb / 1024), status, description
5. **Commit results.tsv** after each experiment so evolve tracking works:
   `git add results.tsv && git commit -m "update results"`
6. If val_bpb improved (lower), set status to `keep` and advance the branch.
7. If val_bpb is equal or worse, set status to `discard` and `git reset --hard HEAD~1`
   to revert the train.py changes (but keep results.tsv updated).
8. Repeat indefinitely. Each experiment should build on previous successes.

## Hints

If an `evolve-hints.md` file exists in the repo root, it contains insights from the
leading agent in the evolve. Consider incorporating their successful ideas.

## Goal

Minimize `val_bpb` within the 5-minute time budget per experiment. Lower is better.
"""


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@click.group(epilog="Exit codes: 0 = success, 1 = error")
@click.version_option(version="1.0.0", prog_name="autoevolve")
@click.option("--no-color", is_flag=True, default=False, help="Disable colored output")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Minimal output")
@click.pass_context
def cli(ctx: click.Context, no_color: bool, quiet: bool) -> None:
    """Multi-agent research competition orchestrator for autoresearch."""
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = OutputConfig(
        color=not no_color and sys.stdout.isatty(),
        quiet=quiet,
    )


@cli.command()
@click.option("--agents", "-n", type=int, required=True, help="Number of competing agents")
@click.option("--base-branch", "-b", type=str, default="main", help="Branch to fork from")
@click.option("--tag", "-t", type=str, required=True, help="Evolve tag (e.g. mar15)")
@click.pass_context
def init(ctx: click.Context, agents: int, base_branch: str, tag: str) -> None:
    """Initialize a new evolve with N competing agent branches."""
    cfg = ctx.obj["cfg"]
    if agents < 1:
        click.echo("Error: --agents must be at least 1.", err=True)
        sys.exit(1)

    # Verify we are inside a git repo
    repo_root = _run_git("rev-parse", "--show-toplevel")
    if not repo_root.ok:
        click.echo("Error: not inside a git repository.", err=True)
        sys.exit(1)

    # Ensure no uncommitted changes before checking out branches
    if not _git_working_tree_clean():
        click.echo("Error: working tree has uncommitted changes. Commit or stash first.", err=True)
        sys.exit(1)

    # Verify base branch exists
    if not _git_branch_exists(base_branch):
        click.echo(f"Error: base branch '{base_branch}' does not exist.", err=True)
        sys.exit(1)

    # Check for existing evolve config
    config_path = _evolve_config_path()
    if config_path.exists():
        click.echo(
            f"Error: evolve.json already exists at {config_path}. "
            "Remove it first or use a different repo.",
            err=True,
        )
        sys.exit(1)

    # Get base commit
    base_sha = _run_git("rev-parse", "--short", base_branch)
    if not base_sha.ok:
        click.echo(f"Error: could not resolve base branch: {base_sha.error}", err=True)
        sys.exit(1)

    # Remember current branch to return to it
    current_branch = _git_current_branch()
    if not current_branch.ok:
        click.echo(f"Error: {current_branch.error}", err=True)
        sys.exit(1)

    agent_configs: list[AgentConfig] = []
    created_branches: list[str] = []

    try:
        for i in range(1, agents + 1):
            strategy = STRATEGIES[(i - 1) % len(STRATEGIES)]
            branch_name = f"evolve/{tag}-agent-{i}"

            # Check if branch already exists
            if _git_branch_exists(branch_name):
                click.echo(f"Error: branch '{branch_name}' already exists.", err=True)
                sys.exit(1)

            # Create branch from base
            result = _run_git("checkout", "-b", branch_name, base_branch)
            if not result.ok:
                click.echo(f"Error creating branch {branch_name}: {result.error}", err=True)
                sys.exit(1)
            created_branches.append(branch_name)

            # Write program.md
            program_content = _generate_program_md(strategy, i, tag)
            program_path = Path(repo_root.value) / "program.md"
            program_path.write_text(program_content)

            # Create an initial empty results.tsv with header
            results_path = Path(repo_root.value) / "results.tsv"
            if not results_path.exists():
                results_path.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")

            # Commit program.md and results.tsv to the branch
            _run_git("add", "program.md", "results.tsv")
            commit_result = _run_git(
                "commit", "-m",
                f"evolve({tag}): initialize agent {i} with {strategy['key']} strategy",
            )
            if not commit_result.ok:
                click.echo(f"Warning: commit on {branch_name}: {commit_result.error}", err=True)

            agent_configs.append(AgentConfig(
                id=i,
                branch=branch_name,
                strategy=strategy["key"],
                status="pending",
            ))
    except SystemExit:
        # Clean up created branches on failure
        _run_git("checkout", current_branch.value, check=False)
        for branch in created_branches:
            _run_git("branch", "-D", branch, check=False)
        raise
    finally:
        # Always return to original branch
        _run_git("checkout", current_branch.value, check=False)

    # Save evolve config (not committed)
    evolve = EvolveConfig(
        tag=tag,
        base_branch=base_branch,
        base_commit=base_sha.value,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        agents=agent_configs,
    )
    save_result = _save_evolve_config(evolve)
    if not save_result.ok:
        click.echo(f"Error saving config: {save_result.error}", err=True)
        sys.exit(1)

    # Print summary
    click.echo(f"\n== {cfg.styled('autoevolve', fg='cyan', bold=True)} initialized ==")
    click.echo(f"Evolve: {tag} | Agents: {agents} | Base: {base_branch} ({base_sha.value})")
    click.echo()
    for ac in agent_configs:
        strategy_info = next(s for s in STRATEGIES if s["key"] == ac.strategy)
        click.echo(f"  Agent {ac.id}: {ac.branch} ({strategy_info['label']})")
    click.echo()
    click.echo("To start each agent, check out its branch and run your autoresearch agent:")
    click.echo()
    for ac in agent_configs:
        click.echo(f"  git checkout {ac.branch}")
        click.echo(f"  # Start your AI agent here (e.g., claude, codex, gemini)")
        click.echo()
    click.echo(f"Monitor progress with: autoevolve status")
    click.echo(f"Cross-pollinate ideas with: autoevolve pollinate")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current evolve status and quick leaderboard."""
    cfg = ctx.obj["cfg"]

    config_result = _load_evolve_config()
    if not config_result.ok:
        click.echo(f"Error: {config_result.error}", err=True)
        sys.exit(1)

    config = config_result.value
    statuses = [_get_agent_status(agent) for agent in config.agents]

    # Find overall leader
    agents_with_results = [s for s in statuses if s.best_val_bpb is not None]
    leader = min(agents_with_results, key=lambda s: s.best_val_bpb) if agents_with_results else None

    # Quiet mode
    if cfg.quiet:
        if leader and leader.best_experiment:
            total_exps = sum(len(s.experiments) for s in statuses)
            click.echo(
                f"Leader: Agent {leader.agent.id}, "
                f"best: {leader.best_val_bpb:.6f} "
                f"({total_exps} experiments)"
            )
        else:
            click.echo("No results yet.")
        return

    click.echo(f"\n== {cfg.styled('autoevolve', fg='cyan', bold=True)} status ==")
    click.echo(
        f"Evolve: {config.tag} | Agents: {len(config.agents)} | "
        f"Started: {config.created_at}"
    )
    click.echo()

    for s in statuses:
        strategy_info = next(
            (st for st in STRATEGIES if st["key"] == s.agent.strategy),
            {"label": s.agent.strategy},
        )
        bpb_str = f"{s.best_val_bpb:.6f}" if s.best_val_bpb is not None else "N/A"
        marker = (
            f"  {cfg.styled(SYM_STAR + ' LEADER', fg='yellow', bold=True)}"
            if (leader and s is leader)
            else ""
        )
        click.echo(
            f"Agent {s.agent.id} ({strategy_info['label']}):  "
            f"{len(s.experiments)} experiments, "
            f"best val_bpb: {bpb_str}, "
            f"{s.keep_count} keeps{marker}"
        )

    if leader and leader.best_experiment:
        click.echo()
        exp = leader.best_experiment
        click.echo(
            f"Overall best: Agent {leader.agent.id} at {exp.val_bpb:.6f} "
            f'("{exp.description}")'
        )

    if not agents_with_results:
        click.echo("\nNo experiment results found yet. Agents may not have started.")

    click.echo()


@cli.command()
@click.option("--detailed", is_flag=True, help="Show detailed per-agent trajectory")
@click.pass_context
def leaderboard(ctx: click.Context, detailed: bool) -> None:
    """Show detailed leaderboard comparison across all agents."""
    cfg = ctx.obj["cfg"]

    config_result = _load_evolve_config()
    if not config_result.ok:
        click.echo(f"Error: {config_result.error}", err=True)
        sys.exit(1)

    config = config_result.value
    statuses = [_get_agent_status(agent) for agent in config.agents]

    ranked = sorted(
        statuses,
        key=lambda s: s.best_val_bpb if s.best_val_bpb is not None else float("inf"),
    )

    click.echo(f"\n== {cfg.styled('autoevolve', fg='cyan', bold=True)} leaderboard ==")
    click.echo(
        f"Evolve: {config.tag} | Agents: {len(config.agents)} | "
        f"Base: {config.base_branch} ({config.base_commit})"
    )
    click.echo()

    header = f"{'Rank':<6}{'Agent':<10}{'Strategy':<25}{'Best BPB':<12}{'Exps':<8}{'Keeps':<8}{'Keep %':<8}"
    click.echo(cfg.styled(header, dim=True))
    click.echo("-" * 77)

    for rank, s in enumerate(ranked, 1):
        strategy_info = next(
            (st for st in STRATEGIES if st["key"] == s.agent.strategy),
            {"label": s.agent.strategy},
        )
        bpb_str = f"{s.best_val_bpb:.6f}" if s.best_val_bpb is not None else "N/A"
        total = len(s.experiments)
        keep_pct = f"{(s.keep_count / total * 100):.0f}%" if total > 0 else "N/A"
        click.echo(
            f"{rank:<6}{s.agent.id:<10}{strategy_info['label']:<25}"
            f"{bpb_str:<12}{total:<8}{s.keep_count:<8}{keep_pct:<8}"
        )

    if detailed:
        click.echo()
        click.echo("=" * 77)
        click.echo("DETAILED TRAJECTORIES")
        click.echo("=" * 77)

        for s in ranked:
            strategy_info = next(
                (st for st in STRATEGIES if st["key"] == s.agent.strategy),
                {"label": s.agent.strategy},
            )
            click.echo(
                f"\n--- Agent {s.agent.id}: {strategy_info['label']} "
                f"({s.agent.branch}) ---"
            )

            if not s.experiments:
                click.echo("  No experiments yet.")
                continue

            traj_header = f"  {'#':<4}{'Commit':<10}{'val_bpb':<12}{'Status':<8}{'Best So Far':<14}{'Description'}"
            click.echo(cfg.styled(traj_header, dim=True))
            click.echo(f"  {'-' * 72}")

            running_best = float("inf")
            for idx, exp in enumerate(s.experiments, 1):
                is_new_best = exp.val_bpb < running_best
                if is_new_best:
                    running_best = exp.val_bpb
                marker = f" {cfg.styled(SYM_KEEP, fg='green')}" if is_new_best else ""
                click.echo(
                    f"  {idx:<4}{exp.commit:<10}{exp.val_bpb:<12.6f}"
                    f"{exp.status:<8}{running_best:<14.6f}{exp.description}{marker}"
                )

        click.echo()
        click.echo("=" * 77)
        click.echo("STRATEGY EFFECTIVENESS")
        click.echo("=" * 77)
        click.echo()

        for s in ranked:
            strategy_info = next(
                (st for st in STRATEGIES if st["key"] == s.agent.strategy),
                {"label": s.agent.strategy},
            )
            computed = _compute_improvements(s.experiments)
            if not computed:
                continue

            deltas = [delta for _, delta in computed]
            avg_improvement = sum(deltas) / len(deltas)
            best_improvement = max(deltas)
            click.echo(
                f"  {strategy_info['label']}: "
                f"avg improvement per keep: {avg_improvement:.6f}, "
                f"best single improvement: {best_improvement:.6f}"
            )

    click.echo()


def _find_impactful_experiments(status: AgentStatus) -> list[tuple[Experiment, float]]:
    """Find the most impactful keep experiments for an agent, sorted by delta."""
    improvements = _compute_improvements(status.experiments)
    improvements.sort(key=lambda pair: pair[1], reverse=True)
    return improvements


def _build_hints_content(
    config: EvolveConfig,
    leader: AgentStatus,
    impactful: list[tuple[Experiment, float]],
) -> str:
    """Build the evolve-hints.md content from leader data."""
    leader_strategy = next(
        (st for st in STRATEGIES if st["key"] == leader.agent.strategy),
        {"label": leader.agent.strategy},
    )
    lines: list[str] = [
        f"# Hints from Evolve {config.tag}",
        f"",
        f"Generated by `auto-evolve pollinate` at "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"",
        f"## Leading Agent",
        f"",
        f"Agent {leader.agent.id} ({leader_strategy['label']}) is currently leading "
        f"with best val_bpb: {leader.best_val_bpb:.6f}",
        f"",
        f"## Most Impactful Experiments",
        f"",
    ]

    # Include up to top 5 most impactful experiments
    top_n = min(5, len(impactful))
    if top_n == 0:
        lines.append("No impactful improvements detected yet.")
        lines.append("")
    else:
        for rank, (exp, delta) in enumerate(impactful[:top_n], 1):
            lines.append(f"### {rank}. {exp.description}")
            lines.append(f"")
            lines.append(f"- **Commit**: {exp.commit}")
            lines.append(f"- **val_bpb**: {exp.val_bpb:.6f}")
            lines.append(f"- **Improvement**: {delta:.6f}")
            lines.append(f"- **Memory**: {exp.memory_gb:.1f} GB")
            lines.append(f"")

            # Try to get the diff for this commit
            diff_result = _run_git(
                "diff", f"{exp.commit}~1", exp.commit,
                "--", "train.py",
                check=False,
            )
            if diff_result.ok and diff_result.value.strip():
                lines.append(f"<details><summary>Code changes</summary>")
                lines.append(f"")
                lines.append(f"```diff")
                lines.append(diff_result.value)
                lines.append(f"```")
                lines.append(f"")
                lines.append(f"</details>")
                lines.append(f"")

    lines.append("## Suggestion")
    lines.append("")
    lines.append(
        "Consider incorporating the above successful changes into your experiments. "
        "These modifications produced measurable improvements in val_bpb."
    )
    lines.append("")

    return "\n".join(lines)


@cli.command()
@click.pass_context
def pollinate(ctx: click.Context) -> None:
    """Cross-pollinate: share winning ideas from the best agent with all others."""
    cfg = ctx.obj["cfg"]
    config_result = _load_evolve_config()
    if not config_result.ok:
        click.echo(f"Error: {config_result.error}", err=True)
        sys.exit(1)

    config = config_result.value
    statuses = [_get_agent_status(agent) for agent in config.agents]

    # Find the leader
    agents_with_results = [s for s in statuses if s.best_val_bpb is not None]
    if not agents_with_results:
        click.echo("No experiment results found yet. Nothing to pollinate.", err=True)
        sys.exit(1)

    leader = min(agents_with_results, key=lambda s: s.best_val_bpb)

    # Find the leader's most impactful "keep" experiments
    impactful = _find_impactful_experiments(leader)

    # Build hints content
    hints_content = _build_hints_content(config, leader, impactful)

    # Write to repo root as an untracked file (no checkout needed!)
    repo_root = _run_git("rev-parse", "--show-toplevel")
    if not repo_root.ok:
        click.echo(f"Error: {repo_root.error}", err=True)
        sys.exit(1)

    hints_path = Path(repo_root.value) / "evolve-hints.md"
    hints_path.write_text(hints_content)

    click.echo(f"Hints from Agent {leader.agent.id} written to {hints_path}")
    click.echo("All agents can read this file regardless of their branch.")
    click.echo()


@cli.command()
@click.option(
    "--format", "fmt",
    type=click.Choice(["json", "tsv"]),
    default="json",
    help="Export format",
)
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path")
@click.pass_context
def export(ctx: click.Context, fmt: str, output: Optional[str]) -> None:
    """Export all agent results to a single file for external analysis."""
    cfg = ctx.obj["cfg"]
    config_result = _load_evolve_config()
    if not config_result.ok:
        click.echo(f"Error: {config_result.error}", err=True)
        sys.exit(1)

    config = config_result.value
    statuses = [_get_agent_status(agent) for agent in config.agents]

    if fmt == "json":
        data = {
            "evolve": config.tag,
            "base_branch": config.base_branch,
            "base_commit": config.base_commit,
            "created_at": config.created_at,
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "agents": [],
        }
        for s in statuses:
            agent_data = {
                "id": s.agent.id,
                "branch": s.agent.branch,
                "strategy": s.agent.strategy,
                "best_val_bpb": s.best_val_bpb,
                "keep_count": s.keep_count,
                "total_experiments": len(s.experiments),
                "experiments": [
                    {
                        "commit": e.commit,
                        "val_bpb": e.val_bpb,
                        "memory_gb": e.memory_gb,
                        "status": e.status,
                        "description": e.description,
                    }
                    for e in s.experiments
                ],
            }
            data["agents"].append(agent_data)

        content = json.dumps(data, indent=2) + "\n"

    elif fmt == "tsv":
        header = "agent_id\tagent_strategy\tcommit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        rows: list[str] = [header]
        for s in statuses:
            for e in s.experiments:
                rows.append(
                    f"{s.agent.id}\t{s.agent.strategy}\t{e.commit}\t"
                    f"{e.val_bpb}\t{e.memory_gb}\t{e.status}\t{e.description}\n"
                )
        content = "".join(rows)
    else:
        click.echo(f"Error: unsupported format '{fmt}'.", err=True)
        sys.exit(1)

    if output:
        Path(output).write_text(content)
        click.echo(f"Exported {fmt.upper()} to {output}")
    else:
        click.echo(content, nl=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cli()


if __name__ == "__main__":
    main()
