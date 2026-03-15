# Autoresearch Steer — Strategy Reference

Deep dive on the three strategy modes, decision trees for switching, and output interpretation.

---

## Strategy Modes

### Auto (Default)

Balances explore and exploit based on experiment count and success rate.

**How it works**:
- Early experiments (< 10): biases toward explore (70/30 explore/exploit)
- Mid experiments (10-25): balanced (50/50)
- Late experiments (25+): biases toward exploit (30/70)
- After 3+ consecutive discards: temporarily overrides to explore regardless of count

**When auto makes the wrong choice**:
- You have a clear winner but auto is still exploring (early experiment count). Override with `--strategy exploit`.
- You're stuck in a rut but auto is exploiting (late experiment count). Override with `--strategy explore`.
- You're in a domain where early broad exploration matters more than the default assumes.

**Example output**:
```
[1] [EXPLOIT] Tune learning rate warmup schedule          risk: low
    Rationale: Learning rate experiments have 3 keeps in 4 attempts.
[2] [EXPLORE] Try rotary position embeddings               risk: medium
    Rationale: Positional encoding category untested. High potential.
[3] [EXPLOIT] Increase batch size from 32 to 48           risk: low
    Rationale: Batch size increase worked well in experiment #12.
```

### Explore

Favors untried categories and novel directions. Use when stuck or early in research.

**How it works**:
- Prioritizes categories with zero keeps
- Deprioritizes categories already attempted (keyword deduplication)
- Assigns higher scores to suggestions that differ most from recent experiments
- Risk distribution: more high-risk suggestions than auto or exploit

**When to use explore**:
- After 3+ consecutive discards
- At the start of a new research session
- When the current optimization axis feels exhausted
- When you want to cast a wide net before committing to a direction

**Example output**:
```
[1] [EXPLORE] Replace GELU with SwiGLU activation         risk: medium
    Rationale: Activation function category untested.
[2] [EXPLORE] Add gradient checkpointing                   risk: high
    Rationale: Memory optimization category untested. Enables larger models.
[3] [EXPLORE] Try cosine annealing with warm restarts     risk: medium
    Rationale: LR schedule category has 0 keeps in 2 attempts — new approach.
```

### Exploit

Doubles down on categories with proven wins. Use when you have clear winners.

**How it works**:
- Prioritizes categories with highest keep rates
- Suggests variations on successful experiments
- Deprioritizes categories with low keep rates
- Risk distribution: mostly low-risk, incremental suggestions

**When to use exploit**:
- You have multiple STRONG_KEEP or KEEP results in a category
- You want to squeeze out remaining gains before switching directions
- Late-stage optimization where bold moves are too risky

**Example output**:
```
[1] [EXPLOIT] Reduce weight decay from 0.1 to 0.05       risk: low
    Rationale: Weight decay reduction produced STRONG_KEEP in experiment #8.
[2] [EXPLOIT] Increase model depth from 6 to 8 layers    risk: low
    Rationale: Architecture scaling has 2 keeps in 2 attempts.
[3] [EXPLOIT] Fine-tune Muon momentum from 0.95 to 0.9   risk: low
    Rationale: Optimizer tuning produced KEEP in experiment #15.
```

---

## Decision Tree for Strategy Switching

```
Start with: --strategy auto (default)

Is auto suggesting stale/irrelevant ideas?
├── Yes → Check your results.tsv is up to date
│   └── Still stale? → Switch to --strategy explore
└── No → Continue with auto

Are you hitting 3+ consecutive discards?
├── Yes → Switch to --strategy explore --num-suggestions 10
└── No → Continue current strategy

Do you have 2+ STRONG_KEEPs in one category?
├── Yes → Switch to --strategy exploit
└── No → Continue current strategy

Has exploit stopped producing keeps?
├── Yes → Switch to --strategy explore (axis exhausted)
└── No → Continue exploiting
```

---

## Output Interpretation

### Suggestion Badges

- **[EXPLORE]**: The category hasn't produced a keep yet. Higher variance — the suggestion might crash or produce a breakthrough.
- **[EXPLOIT]**: The category has proven wins. Lower variance — expect incremental gains.

### Risk Levels

| Risk | Meaning | Guidance |
|------|---------|----------|
| low | Safe to try as-is | Implement directly |
| medium | May need parameter tuning | Start conservative, adjust if OOM |
| high | Bold change, possible crash | Consider reducing scope or adding safeguards |

### Rationale Field

Each suggestion includes a rationale explaining why it was ranked where it is. Key signals:
- "X keeps in Y attempts" — success rate in this category
- "category untested" — no experiments in this area yet
- "similar to experiment #N" — building on a specific prior success
- "deprioritized: keyword overlap with #N" — avoiding repetition

---

## Keyword Deduplication

Autosteer tracks keywords from previous experiment descriptions to avoid suggesting the same thing twice. For example:
- If experiment #5 was "increase learning rate to 3e-4" and it was discarded, autosteer deprioritizes suggestions containing "learning rate" + "increase"
- Exact repeats are fully suppressed
- Partial overlaps are deprioritized but not eliminated (a different approach to learning rate may still appear)

---

## Experiment Count and Auto-Mode Behavior

Auto mode's explore/exploit ratio shifts as experiments accumulate. At 1-5 experiments: 80% explore. At 6-15: balanced. At 16+: exploit-leaning. After 3+ consecutive discards: temporarily overrides to 90% explore regardless of count. See Auto Mode section above for the full breakdown.
