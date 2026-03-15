---
name: autoresearch-steer
description: >-
  This skill should be used when the user asks to "choose next experiment",
  "get suggestions", "what should I try", "I'm stuck", or needs to decide
  what experiment to run next. Provides guided experiment selection via autosteer.
user-invocable: false
allowed-tools: Bash, Read
---

# Autoresearch Steer

Guided experiment selection via autosteer. Every experiment should be intentional, not a random walk.

---

## Iron Law

> **GUIDED EXPLORATION**
>
> Don't random-walk through experiment space. Use autosteer to direct research
> based on what has and hasn't worked. Every experiment choice must be data-driven.
> Running experiments without consulting autosteer is a violation.

## When This Skill Activates

**Triggers** -- situations where autosteer should be consulted:
- Choosing what experiment to try next
- After 3+ consecutive discards (you're stuck)
- At the start of a research session (fresh ideas needed)
- When switching between explore and exploit phases
- When top suggestions from a previous run feel stale or irrelevant

**Does NOT trigger** for:
- The user already has a specific experiment in mind
- The project is not autoresearch / GPT pretraining
- The user is doing exploratory prototyping outside the experiment loop
- Debugging a crashed or broken training run (fix first, steer after)

---

## Workflow

### 1. Get Suggestions

Always run with an up-to-date `results.tsv`:

```bash
autosteer --results results.tsv
```

Default: 5 suggestions, `auto` strategy.

### 2. Choose a Strategy Mode

| Mode | When to Use | Command |
|------|-------------|---------|
| `auto` | Default. Balances explore/exploit based on experiment count. | `autosteer --strategy auto` |
| `explore` | Early research, or stuck in a rut. Favors untried categories. | `autosteer --strategy explore` |
| `exploit` | You have proven wins. Double down on what works. | `autosteer --strategy exploit` |

See `references/strategies.md` for detailed strategy breakdowns, decision trees, and example output.

### 3. Interpret Suggestion Badges

- **[EXPLORE]** -- This category hasn't produced a keep yet. Higher variance, higher potential.
- **[EXPLOIT]** -- This category has proven wins. Lower risk, incremental gains.

### 4. Use Risk Levels to Calibrate

- **low** -- Safe to try, unlikely to crash or waste time
- **medium** -- May need parameter tuning, possible OOM
- **high** -- Bold change, could crash but big upside

### 5. When Stuck (3+ Consecutive Discards)

```bash
# Switch to explore mode for fresh directions
autosteer --results results.tsv --strategy explore --num-suggestions 10
```

Look for categories you haven't tried yet. Autosteer deprioritizes directions that have already been attempted (keyword deduplication).

### 6. Quiet Mode for Quick Reference

```bash
autosteer --results results.tsv --quiet
# Output: numbered titles only
# 1. [EXPLORE] Adjust depth/width ratio
# 2. [EXPLOIT] Tune matrix learning rate
```

### 7. Full Command Reference

```
autosteer --help
autosteer --results results.tsv                          # default: 5 suggestions, auto strategy
autosteer --results results.tsv --strategy explore       # favor new directions
autosteer --results results.tsv --strategy exploit       # favor proven winners
autosteer --results results.tsv --num-suggestions 10     # more suggestions
autosteer --results results.tsv --format json            # JSON output
autosteer --results results.tsv --quiet                  # minimal output
autosteer --results results.tsv --no-color               # plain text
```

---

## Anti-Patterns

| Anti-Pattern | Correct Approach |
|-------------|-----------------|
| Random-walking through experiments without autosteer | Use autosteer for every experiment choice |
| Always using `--strategy auto` | Switch to `explore` when stuck, `exploit` when winning |
| Ignoring risk levels on suggestions | High-risk suggestions need more careful implementation |
| Running without updated results.tsv | Always commit results before running autosteer |
| Cherry-picking low-ranked suggestions | Respect the ranking order unless you have a specific reason |
| Not requesting more suggestions when top picks are irrelevant | Use `--num-suggestions 10` to widen the search |

## Scope Limiter

This skill concerns **experiment selection and strategy** only.

Does NOT apply when:
- The user already has a specific experiment in mind and wants to proceed
- The project is not autoresearch or GPT pretraining
- The user is doing exploratory prototyping outside the experiment loop
- The task is evaluating results (use `autoresearch-evaluate` instead)
- The task is managing multi-agent competitions (use `autoresearch-arena` instead)

---

## Integration

This skill works with sibling autoresearch skills:

| Skill | Relationship |
|-------|-------------|
| `autoresearch-evaluate` | After running an experiment suggested by autosteer, use autojudge to evaluate the result. The verdict feeds back into results.tsv for the next steer cycle. |
| `autoresearch-arena` | In multi-agent competitions, arena assigns strategies to agents. Each agent still uses autosteer within its assigned strategy to pick individual experiments. |

The feedback loop: **steer** (pick experiment) -> **run** (train) -> **evaluate** (judge result) -> **steer** (pick next).

---

## Extended References

- `references/strategies.md` -- Deep dive on auto, explore, and exploit modes with decision trees and example output
- `references/violations.md` -- Expanded anti-pattern examples with scenarios and consequences
