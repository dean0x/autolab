# autolab

Companion tools for [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — the autonomous AI research framework for GPT pretraining.

Three CLIs that make the autoresearch experiment loop smarter: evaluate results statistically, get data-driven suggestions for what to try next, and run multi-agent competitions.

[![CI](https://github.com/dean0x/autolab/actions/workflows/ci.yml/badge.svg)](https://github.com/dean0x/autolab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Install

```bash
# All three tools
pip install autolab

# Or individually
pip install autojudge
pip install autosteer
pip install autoarena
```

Requires Python >= 3.10 (matching autoresearch).

## The Tools

### autojudge — Smarter Experiment Evaluation

Replaces eyeballing `val_bpb` with statistical verdicts that account for noise floor, Pareto efficiency, and trend context.

```bash
autojudge --results results.tsv --run-log run.log
```

```
Experiment #14: val_bpb 3.91 → 3.87
  Verdict: KEEP (confidence: 82%)
  Delta: -1.01% (2.0x noise floor)
  Pareto frontier: yes
  Suggestion: Improvement looks real. Commit and continue.
```

Verdicts: `STRONG_KEEP` | `KEEP` | `MARGINAL` | `RETEST` | `DISCARD` | `CRASH`

Exit codes enable scripting: `autojudge --results results.tsv && git commit -m "keep" || git reset --hard HEAD~1`

### autosteer — Research Direction Generator

Analyzes experiment history and suggests what to try next. Stops the random walk.

```bash
autosteer --results results.tsv
```

```
[1] [EXPLOIT] Tune learning rate warmup schedule          risk: low
    Rationale: Learning rate experiments have 3 keeps in 4 attempts.
[2] [EXPLORE] Try rotary position embeddings               risk: medium
    Rationale: Positional encoding category untested. High potential.
[3] [EXPLOIT] Increase batch size from 32 to 48           risk: low
    Rationale: Batch size increase worked well in experiment #12.
```

Strategy modes: `auto` (default) | `explore` (when stuck) | `exploit` (when winning)

### autoarena — Multi-Agent Competition

Run parallel AI agents with different strategies competing on the same problem.

```bash
autoarena init --agents 3 --tag mar15    # Create 3 agent branches
autoarena leaderboard                     # Who's winning?
autoarena pollinate                       # Spread winning ideas
```

6 built-in strategies assigned round-robin: Architecture First, Hyperparams First, Optimizer First, Regularization First, Efficiency First, Radical.

## The Experiment Loop

These tools plug into the standard autoresearch loop:

```
1. autosteer --results results.tsv         # Pick next experiment
2. Implement the suggestion in train.py
3. uv run train.py > run.log 2>&1          # Train
4. autojudge --results results.tsv         # Evaluate
5. Keep or discard based on verdict
6. Repeat
```

For multi-agent competitions, each agent runs this loop independently on its own branch, and `autoarena pollinate` spreads winning ideas between them.

## AI Agent Integration

The `skills/` directory contains [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill definitions that teach AI agents to use these tools autonomously:

| Skill | Purpose |
|-------|---------|
| `autoresearch-evaluate` | Run autojudge after every experiment, interpret verdicts |
| `autoresearch-steer` | Use autosteer for guided experiment selection |
| `autoresearch-arena` | Set up and manage multi-agent competitions |

The `templates/program-addon.md` is a drop-in snippet you can append to your autoresearch `program.md` to integrate all three tools into the experiment loop.

## Quick Reference

```bash
# Evaluate
autojudge --results results.tsv --run-log run.log    # Full evaluation
autojudge --results results.tsv --format json         # JSON output
autojudge --results results.tsv --quiet               # One-line verdict

# Steer
autosteer --results results.tsv                       # 5 suggestions, auto strategy
autosteer --results results.tsv --strategy explore    # Favor new directions
autosteer --results results.tsv --num-suggestions 10  # More suggestions

# Arena
autoarena init --agents 3 --tag TAG                   # Start competition
autoarena status                                      # Quick overview
autoarena leaderboard --detailed                      # Full analysis
autoarena pollinate                                   # Cross-pollinate wins
autoarena export --format json -o results.json        # Export data
```

All tools support `--quiet` for minimal output and `--no-color` for plain text (auto-disabled when piped).

## Development

```bash
git clone https://github.com/dean0x/autolab.git
cd autolab
pip install -e ./auto-judge -e ./auto-steer -e ./auto-arena
pip install pytest ruff
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT](LICENSE)
