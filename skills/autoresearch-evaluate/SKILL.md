---
name: autoresearch-evaluate
description: >-
  This skill should be used when the user asks to "evaluate experiment",
  "check val_bpb", "decide keep or discard", or needs to determine if a
  training run improved. Provides systematic experiment evaluation via autojudge.
user-invocable: false
allowed-tools: Bash, Read
---

# Autoresearch Evaluate

Systematic experiment evaluation using autojudge. Never eyeball val_bpb — let autojudge account for noise floor, Pareto efficiency, and trend context.

---

## Iron Law

> **EVALUATE SYSTEMATICALLY**
>
> Never eyeball val_bpb to decide keep vs discard. Always run autojudge.
> It accounts for noise floor, Pareto efficiency, and trend context that
> manual comparison misses. One command replaces guesswork.

## When This Skill Activates

**Triggers** — evaluating experiments when:
- A training run finishes and run.log is written
- Deciding whether to keep or discard an experiment
- Checking if val_bpb actually improved versus noise
- Understanding experiment trends across multiple runs
- Interpreting autojudge output or verdicts

**Does NOT trigger** for:
- Manual one-off val_bpb comparisons outside autoresearch
- Projects that are not autoresearch / GPT pretraining
- Reviewing historical results (autojudge evaluates the latest experiment only)
- Designing new experiments (use autoresearch-steer instead)

---

## Workflow

### 1. Run autojudge after each experiment

```bash
autojudge --results results.tsv --run-log run.log
```

Always include `--run-log` for efficiency metrics and richer analysis.

### 2. Interpret the verdict

| Verdict | Meaning | Action |
|---------|---------|--------|
| STRONG_KEEP | Improvement well above noise floor (3x+) | Keep. Commit and advance. |
| KEEP | Improvement likely real (1.5-3x noise) | Keep. Commit and advance. |
| MARGINAL | Improvement within noise (0.5-1.5x) | Keep cautiously. Watch for diminishing returns. |
| RETEST | Improvement indistinguishable from noise | Run the same experiment again to verify. |
| DISCARD | Regression detected | Discard. Revert train.py changes. |
| CRASH | OOM or runtime error | Discard. Reduce model size or fix error. |

### 3. Use exit codes for scripting

```bash
autojudge --results results.tsv --run-log run.log && echo "KEEP" || echo "DISCARD/ERROR"
```

Exit codes:
- `0` = success (STRONG_KEEP, KEEP, MARGINAL, RETEST)
- `1` = error (file not found, parse error)
- `2` = discard/crash verdict

### 4. JSON format for programmatic use

```bash
autojudge --results results.tsv --format json | jq '.verdict'
```

Key JSON fields: `verdict`, `confidence`, `delta_pct`, `noise_floor`, `on_pareto_frontier`, `suggestion`

### 5. Read the suggestion

Autojudge provides contextual suggestions based on:
- Discard/crash streaks (warns when stuck)
- Pareto frontier position
- Diminishing improvement rates
- Memory pressure

### 6. CLI reference

```
autojudge --help
autojudge --results results.tsv                  # human-readable output
autojudge --results results.tsv --format json     # JSON output
autojudge --results results.tsv --run-log run.log # include training metrics
autojudge --results results.tsv --quiet           # one-line output
autojudge --results results.tsv --no-color        # plain text (no ANSI)
```

---

## Integration

This skill works with sibling autoresearch skills:

| Skill | Relationship |
|-------|-------------|
| **autoresearch-steer** | After evaluating, use steer to decide what to try next |
| **autoresearch-clash** | Clash uses autojudge internally for multi-agent competitions |

Typical flow: **run experiment** -> **evaluate** (this skill) -> **steer** (next experiment).

---

## Anti-Patterns

| Anti-Pattern | Correct Approach |
|-------------|-----------------|
| Eyeballing val_bpb to decide keep/discard | Always run `autojudge` — it accounts for noise floor |
| Ignoring RETEST verdict, keeping anyway | Run the experiment again — the improvement may be noise |
| Skipping `--run-log` flag | Include run.log for efficiency metrics and better analysis |
| Treating MARGINAL same as STRONG_KEEP | MARGINAL means watch for diminishing returns, not "celebrate" |
| Cherry-picking metrics instead of using autojudge | Let autojudge weigh all signals holistically |
| Running autojudge on stale results.tsv | Ensure results.tsv reflects the latest experiment before evaluating |

## Scope Limiter

This skill concerns **autoresearch experiment evaluation** only.

It does **NOT** apply when:
- The user is doing manual one-off val_bpb comparisons outside autoresearch
- Working on a project that is not autoresearch / GPT pretraining
- Reviewing historical results (autojudge evaluates the latest experiment only)
- Choosing what experiment to run next (use autoresearch-steer)
- Setting up multi-agent competitions (use autoresearch-clash)

---

## Extended References

See `references/` for detailed examples and expanded guidance:
- `references/verdicts.md` — Extended verdict examples with real val_bpb data and edge cases
- `references/violations.md` — Anti-pattern examples expanded with failure scenarios
