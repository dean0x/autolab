# autoarena

Multi-agent research competition orchestrator for [autoresearch](https://github.com/karpathy/autoresearch). Run parallel AI agents with different strategies and cross-pollinate winning ideas.

## Install

```bash
pip install autoarena
```

## Usage

```bash
# Initialize a 3-agent competition
autoarena init --agents 3 --tag mar15

# Check who's winning
autoarena status
autoarena leaderboard --detailed

# Spread winning ideas to all agents
autoarena pollinate

# Export results
autoarena export --format json -o arena-results.json
```

## How It Works

1. **init** creates one git branch per agent, each with a different research strategy
2. Each agent works independently on its branch using autojudge + autosteer
3. **leaderboard** ranks agents by best val_bpb with keep rate tracking
4. **pollinate** writes the leader's best experiments to `arena-hints.md` — readable from any branch
5. Agents incorporate hints and continue competing

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
| `autoarena init --agents N --tag TAG` | Create N agent branches |
| `autoarena status` | Quick overview with current leader |
| `autoarena leaderboard` | Ranked table with keep rates |
| `autoarena leaderboard --detailed` | Full trajectories + strategy effectiveness |
| `autoarena pollinate` | Cross-pollinate winning ideas |
| `autoarena export --format json\|tsv` | Export results for analysis |

## Requirements

- Python >= 3.10
- A git repository with autoresearch set up
- Multiple compute environments (one per agent)

## License

MIT
