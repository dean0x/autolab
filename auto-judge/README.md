# autojudge

Smarter experiment evaluation for [autoresearch](https://github.com/karpathy/autoresearch). Replaces eyeballing `val_bpb` with statistical verdicts that account for noise floor, Pareto efficiency, and trend context.

## Install

```bash
pip install autojudge
```

## Usage

```bash
# Evaluate the latest experiment
autojudge --results results.tsv --run-log run.log

# JSON output for scripting
autojudge --results results.tsv --format json | jq '.verdict'

# One-line verdict
autojudge --results results.tsv --quiet
```

## Verdicts

| Verdict | Meaning | Exit Code |
|---------|---------|-----------|
| STRONG_KEEP | Improvement well above noise floor (3x+) | 0 |
| KEEP | Improvement likely real (1.5-3x noise) | 0 |
| MARGINAL | Improvement within noise (0.5-1.5x) | 0 |
| RETEST | Indistinguishable from noise | 0 |
| DISCARD | Regression detected | 2 |
| CRASH | OOM or runtime error | 2 |

Exit code 1 = input error (file not found, parse failure).

## Scripting

```bash
# Auto-evaluate and commit or revert
if autojudge --results results.tsv --run-log run.log; then
    git commit -m "keep"
else
    git reset --hard HEAD~1
fi
```

## JSON Output

```json
{
  "verdict": "KEEP",
  "confidence": 0.82,
  "val_bpb": 3.87,
  "prev_best": 3.91,
  "delta": -0.04,
  "delta_pct": -1.01,
  "noise_floor": 0.02,
  "on_pareto_frontier": true,
  "suggestion": "Good improvement in val_bpb. This experiment is on the Pareto frontier. Good progress.",
  "...": "plus efficiency, trends, pareto_frontier, experiment details"
}
```

## How It Works

- Estimates noise floor from pairwise differences between consecutive keeps
- Scores improvement confidence as a ratio of delta to noise floor
- Tracks Pareto frontier (val_bpb vs memory efficiency)
- Detects streaks, plateaus, and diminishing returns
- Parses `run.log` for OOM warnings, memory pressure, and training metrics

## Requirements

- Python >= 3.10
- A `results.tsv` file from autoresearch

## License

MIT
