# Awesome Autoresearch - Companion Tools

Companion tools for [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — the autonomous AI research framework.

## Installation

Each tool can be installed independently from its directory:

```bash
cd auto-judge && uv pip install -e .   # installs `autojudge` CLI
cd auto-steer && uv pip install -e .   # installs `autosteer` CLI
cd auto-arena && uv pip install -e .   # installs `autoarena` CLI
```

Requires Python >= 3.10 (matching autoresearch).

## Usage

Run any tool from inside your autoresearch repo directory:

```bash
# Evaluate the latest experiment
autojudge --results results.tsv

# Get suggestions for what to try next
autosteer --results results.tsv

# Set up a multi-agent competition
autoarena init --agents 3 --tag mar15
autoarena status
autoarena pollinate
```

## Tools

### 1. autojudge — Smarter Experiment Evaluation
Replaces the simple "did val_bpb go down?" check with statistical analysis.
- Estimates noise floor from pairwise differences between consecutive keeps
- Scores improvement confidence (STRONG_KEEP / KEEP / MARGINAL / RETEST / DISCARD)
- Tracks compute efficiency (improvement per VRAM GB, per parameter)
- Detects plateau and diminishing returns
- Pareto frontier analysis (val_bpb vs memory)
- Exit code 0 for keeps, 2 for discards — enables `autojudge && git commit` scripting

### 2. autosteer — Research Direction Generator
Analyzes experiment history and generates smart next-step suggestions.
- Categorizes past experiments by type (architecture, hyperparams, optimizer, etc.)
- 20 built-in research directions specific to GPT pretraining
- Deduplication: won't suggest what's already been tried
- Strategy modes: auto / explore / exploit
- Word-boundary-aware keyword matching to avoid false classifications

### 3. autoarena — Multi-Agent Research Competition
Orchestrates multiple AI agents competing on the same autoresearch problem.
- Creates separate git branches with different research strategies per agent
- Monitors results.tsv across all branches without checkout
- Produces a ranked leaderboard with trajectory analysis
- Cross-pollinates via `arena-hints.md` (no branch checkout, safe for concurrent agents)
- 6 built-in strategies: architecture-first, hyperparams-first, optimizer-first, regularization-first, efficiency-first, radical

All three tools support `--quiet` / `-q` for minimal output and `--no-color` for plain text (auto-disabled when piped):

```bash
# One-line verdict
autojudge --results results.tsv --quiet

# Just numbered titles
autosteer --results results.tsv --quiet

# Leader summary
autoarena status --quiet

# Pipe-safe (color auto-disabled)
autojudge --results results.tsv | cat
```

## Agent Skills

Three skill files in `skills/` let AI agents use the tools autonomously:

### autoresearch-evaluate
Teaches the agent to run `autojudge` after every experiment and interpret verdicts.
Located at `skills/autoresearch-evaluate/SKILL.md`.

### autoresearch-steer
Teaches the agent to use `autosteer` for guided experiment selection, especially when stuck.
Located at `skills/autoresearch-steer/SKILL.md`.

### autoresearch-arena
Teaches the agent to set up and manage multi-agent competitions with `autoarena`.
Located at `skills/autoresearch-arena/SKILL.md`.

## Integration

`skills/program-addon.md` is a drop-in snippet you can append to your autoresearch `program.md`. It modifies the experiment loop to:
1. Evaluate with `autojudge` after each run (replaces manual val_bpb comparison)
2. Call `autosteer` when stuck (3+ consecutive discards)
3. Use exit codes for automated keep/discard decisions

## Future Tools (Planned)

### 4. autodashboard — Real-time Experiment Visualizer
Live web dashboard showing experiment progress, diffs, and trends.

### 5. autoreplay — Experiment Narrative Generator
Replays research history as a shareable narrative with visualizations.

### 6. autoconfig — Hardware-Adaptive Presets
Auto-detects hardware and generates optimal autoresearch configuration.
