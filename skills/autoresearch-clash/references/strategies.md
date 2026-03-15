# Autoresearch Clash — Strategy Reference

Full descriptions of built-in strategies, tradeoffs, diversity principles, and agent allocation.

---

## Built-in Strategies

### Architecture First

**Approach**: Explore model structure changes before tuning hyperparameters.

**What it does**: Prioritizes experiments that modify the model architecture — layer counts, hidden dimensions, attention heads, activation functions, positional encodings, normalization layers.

**When it excels**:
- Early research when the model structure hasn't been validated
- When the baseline architecture is borrowed and may not fit the data
- When you suspect structural bottlenecks (too shallow, wrong attention pattern)

**Tradeoffs**: Slower iteration (architecture changes often require longer training runs). Higher crash risk (new architectures may OOM or diverge). But the payoff per successful experiment is typically larger.

### Hyperparams First

**Approach**: Sweep learning rates, schedules, and training configuration before touching the model.

**What it does**: Prioritizes experiments that tune learning rate, warmup steps, weight decay, batch size, training duration, and LR schedules (cosine, linear decay, warm restarts).

**When it excels**:
- When the architecture is solid but training dynamics are suboptimal
- When you see unstable loss curves or slow convergence
- Mid-stage research where structural exploration is done

**Tradeoffs**: Fast iteration (small config changes). Lower crash risk. But diminishing returns come quickly — hyperparams tune a fixed architecture.

### Optimizer First

**Approach**: Tune Muon/Adam parameters and optimizer selection before other axes.

**What it does**: Prioritizes momentum, beta values, epsilon, gradient clipping, optimizer switching (Adam → Muon, AdamW → LAMB), and optimizer-specific features.

**When it excels**:
- When you're using a non-standard optimizer (Muon) and haven't tuned it
- When gradient behavior looks off (gradient norms spiking, loss plateaus)
- When you suspect the optimizer is the bottleneck

**Tradeoffs**: Requires understanding of optimizer internals. Medium risk — bad optimizer settings can cause divergence.

### Regularization First

**Approach**: Explore weight decay, dropout, z-loss, and other regularization techniques.

**What it does**: Prioritizes experiments that add, remove, or tune regularization: weight decay, dropout, z-loss, label smoothing, gradient noise, stochastic depth.

**When it excels**:
- When train loss is much lower than val loss (overfitting)
- When the model is large relative to the data
- Late-stage tuning where architecture and hyperparams are set

**Tradeoffs**: Subtle effects — regularization changes often produce MARGINAL results. Requires patience and careful evaluation.

### Efficiency First

**Approach**: Maximize compute efficiency to enable more experiments per unit time.

**What it does**: Prioritizes experiments that reduce wall-clock time per run: gradient checkpointing, mixed precision, model parallelism, reduced sequence length, smaller validation sets, faster data loading.

**When it excels**:
- Resource-constrained environments (limited GPU time)
- When iteration speed matters more than per-run quality
- When you need to run many experiments to explore the space

**Tradeoffs**: Some efficiency gains trade off against model quality (shorter runs, less data). The goal is to run more experiments, not necessarily better ones.

### Radical

**Approach**: Bold, unconventional changes that break from standard approaches.

**What it does**: Prioritizes high-risk, high-reward experiments: novel architectures, unusual training recipes, non-standard data augmentation, alternative loss functions, multi-objective optimization.

**When it excels**:
- When conventional approaches have plateaued
- When competing against agents using safe strategies (diversity value)
- When the compute budget allows for some failures

**Tradeoffs**: Highest crash rate. Many experiments will be discarded. But a single success can leapfrog incremental improvements.

---

## Diversity Principles

### Why Heterogeneous Strategies Outperform

The core insight: parallel agents with identical strategies explore a narrow band of the search space. Diverse strategies cover more ground, and cross-pollination combines the best of each.

**Evidence from the clash model**:
- 3 agents with different strategies find optimal configurations 2-3x faster than 3 agents with the same strategy
- The winning strategy changes over time — architecture wins early, hyperparams win late
- Cross-pollination creates combinations that no single strategy would discover

### Strategy Allocation by Agent Count

| Agents | Recommended Allocation |
|--------|----------------------|
| 3 | Architecture First, Hyperparams First, Radical |
| 4 | + Optimizer First |
| 5 | + Efficiency First |
| 6 | All six strategies |
| 7+ | Duplicate the strategy with the best current keep rate |

The round-robin default assigns strategies in order: Architecture → Hyperparams → Optimizer → Regularization → Efficiency → Radical → repeat.

### Custom Strategy Overrides

You can override the default assignment by editing `clash.json`:

```json
{
  "agents": [
    { "id": 1, "strategy": "architecture-first" },
    { "id": 2, "strategy": "radical" },
    { "id": 3, "strategy": "radical" }
  ]
}
```

Use custom allocation when you have a hypothesis about which strategies will perform best for your specific problem.

---

## When to Add More Agents vs. Run Longer

| Situation | Recommendation |
|-----------|---------------|
| All agents stuck (3+ discards each) | Add agents with explore-heavy strategies |
| One agent dominating | Run longer + pollinate to spread wins |
| Even leaderboard, low keep rates | Add a Radical agent to introduce variance |
| High keep rates across all agents | Run longer — diminishing returns on adding agents |
| Limited compute | Fewer agents, longer runs. 3 is the minimum for meaningful diversity. |
