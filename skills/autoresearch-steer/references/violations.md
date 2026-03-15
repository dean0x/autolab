# Autoresearch Steer -- Violation Examples

Expanded anti-pattern examples showing what goes wrong when autosteer is bypassed or misused.

---

## Random-Walking Without Autosteer

Picking experiments based on intuition or guesswork without consulting autosteer.

**What goes wrong:** You repeat experiments already tried, miss high-potential categories autosteer would have surfaced, and lose the data-driven feedback loop. Random walks waste compute on redundant directions.

**Fix:** Always consult autosteer before choosing. You can override its suggestion, but make it deliberate.

```bash
autosteer --results results.tsv
```

---

## Never Switching Strategy

Leaving `--strategy auto` on permanently, even when clearly stuck or clearly winning.

**What goes wrong:** Auto is a heuristic. When stuck (3+ discards), it may still suggest exploit ideas from exhausted categories. When you have clear winners, it may waste compute on low-probability explore suggestions.

**Fix:** Monitor your keep rate and switch:
- 3+ consecutive discards: `--strategy explore`
- 2+ STRONG_KEEPs in a category: `--strategy exploit`
- Exploit stops producing keeps: back to `--strategy explore`

---

## Ignoring Risk Levels

Implementing a high-risk suggestion without adjusting parameters.

**What goes wrong:** High-risk suggestions are more likely to OOM or crash. Without parameter tuning, you waste a full training run on a crash. The suggestion is a direction, not a recipe.

**Fix:** Treat risk levels as implementation guidance:
- **low**: implement as suggested
- **medium**: start conservative (e.g., increase by 1.5x instead of 2x)
- **high**: add safeguards (gradient checkpointing, reduced batch size, shorter run first)

---

## Running Without Updated results.tsv

Running autosteer before committing the latest experiment results to results.tsv.

**What goes wrong:** Suggestions are based on stale data. Keyword deduplication misses your most recent experiment, causing duplicate suggestions. Strategy weighting uses outdated keep rates.

**Fix:** Always update results.tsv before steering:

```bash
# After autojudge: update results.tsv, commit, then steer
autosteer --results results.tsv
```

---

## Cherry-Picking Low-Ranked Suggestions

Skipping the top-ranked suggestion and picking #4 or #5 because it "sounds more interesting."

**What goes wrong:** Ranking reflects a scoring heuristic accounting for success rate, novelty, and risk. Lower-ranked suggestions have lower expected value. Systematically ignoring top suggestions defeats the purpose of autosteer.

**Fix:** Start with the top-ranked suggestion. Skip only if it conflicts with a known constraint (e.g., architecture incompatibility) -- and document why.

---

## Not Requesting More Suggestions

Finding nothing relevant in the default 5 suggestions and giving up on autosteer entirely.

**What goes wrong:** The default 5 is a convenience limit. The pool may have 15-20 candidates, and suggestions #6-10 may be exactly what you need.

**Fix:** Widen the search before abandoning the tool:

```bash
autosteer --results results.tsv --num-suggestions 10
```

If still nothing fits, switch strategy modes before picking an experiment manually.

---

## Quick Detection Signals

| Signal | Likely Violation |
|--------|-----------------|
| No `autosteer` in recent command history | Random-walking |
| `--strategy auto` used 5+ times consecutively | Never switching strategy |
| OOM crash on a high-risk suggestion | Ignoring risk levels |
| results.tsv last modified before latest experiment | Stale results |
| Consistently picking suggestion #3 or lower | Cherry-picking |
| "None of these work" with only 5 requested | Not requesting more |
