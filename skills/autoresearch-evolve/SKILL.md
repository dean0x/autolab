---
name: autoresearch-evolve
description: >-
  This skill should be used when the user asks to "set up evolve", "run competing
  agents", "pollinate results", "check leaderboard", or needs to coordinate
  multi-agent research competitions. Provides parallel strategy exploration via autoevolve.
user-invocable: false
allowed-tools: Bash, Read
---

# Autoresearch Evolve

Run multi-agent research competitions with autoevolve. Parallel agents with diverse strategies explore the search space faster than a single agent.

---

## Iron Law

> **COMPETE TO DISCOVER**
>
> Parallel agents with different strategies explore the search space faster
> than a single agent. Use evolve to diversify approaches and cross-pollinate wins.
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

### 1. Initialize the evolve

```bash
autoevolve init --agents 3 --tag mar15
```

This creates:
- One git **worktree** per agent in a sibling directory (e.g. `../myrepo-evolve-mar15-agent-1/`)
- Each worktree has its own branch (`evolve/mar15-agent-1`, etc.) with a `program.md` for its strategy
- An `evolve.json` config file tracking the competition (in the main repo)
- An untracked `results.tsv` header in each worktree (not committed — matches autoresearch convention)

Agents are assigned strategies round-robin from the built-in pool. With 3 agents, you get 3 different strategies competing. See `references/strategies.md` for the full catalog.

### 2. Start each agent

Navigate to each agent's worktree directory and launch your AI agent. Each agent works independently, using autojudge to evaluate and autosteer to choose its next moves.

```bash
cd ../myrepo-evolve-mar15-agent-1
# Start agent (claude, codex, gemini, etc.)
```

### 3. Monitor progress

```bash
autoevolve status          # quick overview with leader
autoevolve leaderboard     # ranked table with keep rates
autoevolve leaderboard --detailed  # full trajectories + strategy effectiveness
```

Check the leaderboard regularly to know when to pollinate. A clear leader with 3+ keeps ahead of the pack is a good pollination signal.

### 4. Cross-pollinate winning ideas

```bash
autoevolve pollinate
```

This reads the leader's results, finds their most impactful experiments, and writes `evolve-hints.md` to each agent's worktree directory. All agents can read their local copy — it contains winning experiments, code diffs, and incorporation suggestions.

Timing matters: too early spreads noise, too late means agents have diverged beyond benefit. See `references/violations.md` for pollination timing guidance.

### 5. Export for analysis

```bash
autoevolve export --format json -o evolve-results.json
autoevolve export --format tsv -o evolve-results.tsv
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
autoevolve --help
autoevolve init --agents N --tag TAG [--base-branch BRANCH] [--worktree-dir DIR]
autoevolve status [--quiet]
autoevolve leaderboard [--detailed]
autoevolve pollinate
autoevolve export --format json|tsv [-o FILE]
autoevolve cleanup [--export-first] [--yes]
```

---

## Integration

This skill works with sibling autoresearch skills:

| Skill | Relationship |
|-------|-------------|
| **autoresearch-evaluate** | Each agent uses autojudge to evaluate its own experiments |
| **autoresearch-steer** | Each agent uses autosteer to choose experiments per its assigned strategy |

Typical flow: **init evolve** -> agents independently **run** -> **evaluate** -> **steer** -> **monitor leaderboard** -> **pollinate** -> repeat.

---

## Anti-Patterns

| Anti-Pattern | Correct Approach |
|-------------|-----------------|
| All agents using the same strategy | Use diverse strategies for better coverage |
| Never running `pollinate` | Cross-pollinate regularly to spread winning ideas |
| Launching evolve with < 3 agents | 3+ agents needed for meaningful strategy diversity |
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
