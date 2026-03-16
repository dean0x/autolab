# autoevolve

Multi-agent research competition orchestrator for [autoresearch](https://github.com/karpathy/autoresearch). Run parallel AI agents with different strategies and cross-pollinate winning ideas.

## Install

```bash
pip install autoevolve
```

## Usage

```bash
# Initialize a 3-agent competition
autoevolve init --agents 3 --tag mar15

# Check who's winning
autoevolve status
autoevolve leaderboard --detailed

# Spread winning ideas to all agents
autoevolve pollinate

# Export results
autoevolve export --format json -o evolve-results.json
```

## How It Works

1. **init** creates a git worktree per agent in a sibling directory, each with a different research strategy
2. Each agent works independently in its worktree directory using autojudge + autosteer
3. **leaderboard** ranks agents by best val_bpb with keep rate tracking
4. **pollinate** writes the leader's best experiments to `evolve-hints.md` in each agent's worktree
5. Agents incorporate hints and continue competing
6. **cleanup** removes worktrees, branches, and config when done

## Built-in Strategies

| Strategy | Approach |
|----------|----------|
| Architecture First | Explore model structure before tuning |
| Hyperparams First | Sweep learning rates and schedules first |
| Optimizer First | Tune Muon/Adam parameters first |
| Regularization First | Explore weight decay, dropout, z-loss |
| Efficiency First | Maximize compute efficiency to run more experiments |
| Radical | Bold, unconventional changes |

Strategies are assigned round-robin. With 3 agents, you get 3 different strategies competing.

## Commands

| Command | Description |
|---------|-------------|
| `autoevolve init --agents N --tag TAG` | Create N agent worktrees |
| `autoevolve init ... --worktree-dir DIR` | Place worktrees in custom directory |
| `autoevolve status` | Quick overview with current leader |
| `autoevolve leaderboard` | Ranked table with keep rates |
| `autoevolve leaderboard --detailed` | Full trajectories + strategy effectiveness |
| `autoevolve pollinate` | Cross-pollinate winning ideas |
| `autoevolve export --format json\|tsv` | Export results for analysis |
| `autoevolve cleanup` | Remove worktrees, branches, and config |

## Requirements

- Python >= 3.10
- A git repository with autoresearch set up
- Multiple compute environments (one per agent)

## License

MIT
