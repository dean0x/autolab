# autosteer

Research direction generator for [autoresearch](https://github.com/karpathy/autoresearch). Analyzes experiment history and suggests data-driven next steps instead of random-walking through experiment space.

## Install

```bash
pip install autosteer
```

## Usage

```bash
# Get 5 suggestions (default)
autosteer --results results.tsv

# Explore mode — favor untried directions (good when stuck)
autosteer --results results.tsv --strategy explore

# Exploit mode — double down on what works
autosteer --results results.tsv --strategy exploit

# More suggestions
autosteer --results results.tsv --num-suggestions 10

# Quick numbered list
autosteer --results results.tsv --quiet
```

## Strategy Modes

| Mode | When to Use |
|------|-------------|
| `auto` | Default. Balances explore/exploit based on experiment count. |
| `explore` | Early research, or stuck after 3+ discards. Favors untried categories. |
| `exploit` | You have proven wins. Doubles down on what works. |

## Output

Each suggestion includes:
- **Badge**: `[EXPLORE]` or `[EXPLOIT]` indicating category status
- **Category, risk, and expected improvement range**
- **Reasoning**: Why this direction is recommended

```
1. [EXPLOIT] Tune learning rate warmup schedule
   Category: hyperparams | Risk: low | Expected: +0.1-0.3%
   Currently WARMUP_RATIO=0.0 (no warmup). Try WARMUP_RATIO=0.05...

2. [EXPLORE] Tune RoPE base frequency
   Category: embedding | Risk: low | Expected: +0.1-0.3%
   Adjust the RoPE base frequency (theta)...
```

## How It Works

- 20 built-in research directions specific to GPT pretraining
- Categorizes past experiments (architecture, hyperparams, optimizer, etc.)
- Keyword deduplication: won't re-suggest failed directions
- Git integration: reads diffs to classify experiments automatically
- Strategy-weighted scoring that adapts to experiment count

## Requirements

- Python >= 3.10
- A `results.tsv` file from autoresearch
- Git repository (for diff-based experiment classification)

## License

MIT
