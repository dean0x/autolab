# Autoresearch Clash — Violation Examples

Expanded anti-pattern examples showing what goes wrong when clash is misused.

---

## Homogeneous Strategies

### The Mistake

Launching 3 agents all using the default strategy or manually assigning the same strategy to each.

### What Goes Wrong

- All agents explore the same narrow band of the search space.
- Compute is wasted on redundant experiments.
- Cross-pollination produces no new information (everyone already tried the same things).
- You lose the primary advantage of clash: diverse parallel exploration.

### Correct Approach

Let the round-robin default assign different strategies, or manually assign diverse ones:

```bash
autoclash init --agents 3 --tag experiment-run
# Automatically assigns: Architecture First, Hyperparams First, Optimizer First
```

---

## Not Pollinating

### The Mistake

Running the clash for 20+ experiments per agent without ever running `autoclash pollinate`.

### What Goes Wrong

- Agents reinvent each other's discoveries independently. Agent 2 spends 5 experiments finding something Agent 1 found in experiment #3.
- Winning ideas stay siloed in one branch.
- The collective knowledge of the clash is fragmented.

### Correct Approach

Pollinate regularly — typically after every 5-10 experiments per agent, or whenever the leaderboard shows a clear leader:

```bash
autoclash leaderboard
# If a leader has emerged:
autoclash pollinate
```

---

## Pollinating Too Early

### The Mistake

Running `autoclash pollinate` after only 2-3 experiments per agent.

### What Goes Wrong

- With too few experiments, there's no clear signal about which strategy is winning.
- You spread noise instead of signal — early results may not be representative.
- Agents lose their strategic independence before they've had time to explore.

### Correct Approach

Wait for at least 5 experiments per agent and a clear leader (2+ point gap on the leaderboard) before pollinating.

---

## Pollinating Too Late

### The Mistake

Waiting until agents have each run 30+ experiments before the first pollination.

### What Goes Wrong

- Agents have diverged significantly — their codebases are now very different.
- Incorporating winning ideas from a distant branch becomes harder (merge conflicts, incompatible changes).
- You missed the window where pollination is cheap and effective.

### Correct Approach

Pollinate every 5-10 experiments per agent. Frequent small pollinations are better than rare large ones.

---

## Insufficient Agents

### The Mistake

Launching an clash with only 1-2 agents.

### What Goes Wrong

- 1 agent: no competition, no diversity. Just use autosteer directly.
- 2 agents: binary comparison, minimal strategy diversity. The "clash" adds overhead without meaningful benefit.

### Correct Approach

Minimum 3 agents for meaningful strategy diversity. This gives you 3 different approaches competing and enough data points for the leaderboard to be informative.

---

## Never Monitoring

### The Mistake

Launching the clash and not checking `status` or `leaderboard` until the end.

### What Goes Wrong

- You miss the optimal pollination window.
- One agent may be stuck (all discards) while others advance — you could have intervened.
- Strategy effectiveness data goes unused — you can't adapt the competition.

### Correct Approach

Check the leaderboard periodically:

```bash
autoclash leaderboard --detailed
```

Key signals to watch:
- Keep rate per agent (below 30% = agent may be stuck)
- Leader gap (2+ points = time to pollinate)
- Strategy effectiveness (which categories produce the most keeps)

---

## Ignoring Strategy Effectiveness

### The Mistake

Looking only at the overall leaderboard score without examining which strategies produce the best results.

### What Goes Wrong

- You miss insights about the problem space. If Architecture First is winning, that tells you the bottleneck is structural.
- You can't make informed decisions about custom strategy allocation for the next clash run.
- The `--detailed` flag exists for this purpose — not using it is leaving data on the table.

### Correct Approach

Use `autoclash leaderboard --detailed` to see strategy-level metrics. Use this data to inform:
- Which strategies to duplicate if adding more agents
- Whether to run a follow-up clash with adjusted allocations
- What direction to pursue in single-agent mode after the clash ends
