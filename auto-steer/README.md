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
- **Risk level**: `low`, `medium`, or `high`
- **Rationale**: Why this was ranked where it is

```
[1] [EXPLOIT] Tune learning rate warmup schedule          risk: low
    Rationale: Learning rate experiments have 3 keeps in 4 attempts.
[2] [EXPLORE] Try rotary position embeddings               risk: medium
    Rationale: Positional encoding category untested. High potential.
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
