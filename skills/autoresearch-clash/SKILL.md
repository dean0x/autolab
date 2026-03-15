---
name: autoresearch-clash
description: >-
  This skill should be used when the user asks to "set up clash", "run competing
  agents", "pollinate results", "check leaderboard", or needs to coordinate
  multi-agent research competitions. Provides parallel strategy exploration via autoclash.
user-invocable: false
allowed-tools: Bash, Read
---

# Autoresearch Clash

Run multi-agent research competitions with autoclash. Parallel agents with diverse strategies explore the search space faster than a single agent.

---

## Iron Law

> **COMPETE TO DISCOVER**
>
> Parallel agents with different strategies explore the search space faster
> than a single agent. Use clash to diversify approaches and cross-pollinate wins.
> Homogeneous strategies waste compute. Diversity is the engine of discovery.

## When This Skill Activates

**Triggers** — coordinating multi-agent research when:
- Setting up multiple agents to compete on the same autoresearch problem
- Comparing different research strategies (architecture-first vs hyperparams-first)
- Cross-pollinating successful experiments between agents
- Monitoring which agent or strategy is winning via leaderboard

**Does NOT trigger** for:
- Single-agent research sessions (no competition needed)
- The user does not have multiple compute environments available
- Fewer than 5 experiments expected (insufficient data for comparison)
- Evaluating individual experiments (use autoresearch-evaluate)
- Choosing what experiment to run next (use autoresearch-steer)

---

## Workflow

### 1. Initialize the clash

```bash
autoclash init --agents 3 --tag mar15
```

This creates:
- One git branch per agent (`clash/mar15-agent-1`, `clash/mar15-agent-2`, etc.)
- Each branch gets a `program.md` with a different strategy
- An `clash.json` config file tracking the competition

Agents are assigned strategies round-robin from the built-in pool. With 3 agents, you get 3 different strategies competing. See `references/strategies.md` for the full catalog.

### 2. Start each agent

Check out each branch and launch your AI agent. Each agent works independently, using autojudge to evaluate and autosteer to choose its next moves.

```bash
git checkout clash/mar15-agent-1
# Start agent (claude, codex, gemini, etc.)
```

### 3. Monitor progress

```bash
autoclash status          # quick overview with leader
autoclash leaderboard     # ranked table with keep rates
autoclash leaderboard --detailed  # full trajectories + strategy effectiveness
```

Check the leaderboard regularly to know when to pollinate. A clear leader with 3+ keeps ahead of the pack is a good pollination signal.

### 4. Cross-pollinate winning ideas

```bash
autoclash pollinate
```

This reads the leader's results, finds their most impactful experiments, and writes `clash-hints.md` to the repo root. All agents can read this file regardless of their branch -- it contains winning experiments, code diffs, and incorporation suggestions.

Timing matters: too early spreads noise, too late means agents have diverged beyond benefit. See `references/violations.md` for pollination timing guidance.

### 5. Export for analysis

```bash
autoclash export --format json -o clash-results.json
autoclash export --format tsv -o clash-results.tsv
```

### 6. Built-in strategies

| Strategy | Approach |
|----------|----------|
| Architecture First | Explore model structure before tuning |
| Hyperparams First | Sweep learning rates and schedules first |
| Optimizer First | Tune Muon/Adam parameters first |
| Regularization First | Explore weight decay, dropout, z-loss |
| Efficiency First | Maximize compute efficiency to run more experiments |
| Radical | Bold, unconventional changes |

See `references/strategies.md` for detailed strategy descriptions, tradeoffs, and custom assignment.

### 7. CLI reference

```
autoclash --help
autoclash init --agents N --tag TAG [--base-branch BRANCH]
autoclash status [--quiet]
autoclash leaderboard [--detailed]
autoclash pollinate
autoclash export --format json|tsv [-o FILE]
```

---

## Integration

This skill works with sibling autoresearch skills:

| Skill | Relationship |
|-------|-------------|
| **autoresearch-evaluate** | Each agent uses autojudge to evaluate its own experiments |
| **autoresearch-steer** | Each agent uses autosteer to choose experiments per its assigned strategy |

Typical flow: **init clash** -> agents independently **run** -> **evaluate** -> **steer** -> **monitor leaderboard** -> **pollinate** -> repeat.

---

## Anti-Patterns

| Anti-Pattern | Correct Approach |
|-------------|-----------------|
| All agents using the same strategy | Use diverse strategies for better coverage |
| Never running `pollinate` | Cross-pollinate regularly to spread winning ideas |
| Launching clash with < 3 agents | 3+ agents needed for meaningful strategy diversity |
| Not monitoring via `status`/`leaderboard` | Check leaderboard to know when to pollinate |
| Pollinating too early (before clear winners) | Wait for a leader with 3+ keeps before pollinating |
| Pollinating too late (agents diverged) | Pollinate before strategies fork beyond reconciliation |
| Ignoring strategy effectiveness in leaderboard | Use `--detailed` to see which strategies actually work |

## Scope Limiter

This skill concerns **multi-agent research competitions** only.

It does **NOT** apply when:
- Running single-agent research (no competition needed)
- Fewer than 5 experiments expected (insufficient data for comparison)
- Evaluating individual experiments (use autoresearch-evaluate)
- Choosing next experiments without competition context (use autoresearch-steer)

---

## Extended References

See `references/` for detailed examples and expanded guidance:
- `references/strategies.md` -- Full strategy catalog with tradeoffs, diversity principles, and custom assignment
- `references/violations.md` -- Anti-pattern examples expanded with failure scenarios and timing guidance
