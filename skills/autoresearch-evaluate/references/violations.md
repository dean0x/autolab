# Autoresearch Evaluate — Violation Examples

Expanded anti-pattern examples showing what goes wrong when autojudge is bypassed or misused.

---

## Manual val_bpb Comparison

### The Mistake

Comparing val_bpb numbers by eye: "3.91 is less than 3.93, so the experiment improved."

### What Goes Wrong

- You miss the noise floor. A delta of 0.02 means nothing if the noise floor is 0.03.
- You miss Pareto context. The experiment may improve val_bpb but regress on efficiency (tokens/sec).
- You miss trend context. The improvement may be part of a diminishing returns pattern.
- You don't account for incomplete runs. A lower val_bpb from a crashed run (partial convergence) is unreliable.

### Correct Approach

```bash
autojudge --results results.tsv --run-log run.log
```

Let autojudge weigh noise floor, Pareto frontier, and trend context together.

---

## Ignoring RETEST Verdict

### The Mistake

Autojudge returns RETEST, but you keep the experiment anyway because "the number went down a little."

### What Goes Wrong

- The improvement is indistinguishable from noise. Keeping it adds false signal to results.tsv.
- Future autosteer suggestions will be based on an improvement that may not be real.
- If the experiment was actually noise, you wasted a keep slot and now build on a false foundation.

### Correct Approach

Re-run the exact same experiment. If the second run also shows improvement, autojudge will likely upgrade to MARGINAL or KEEP. If it reverts, you avoided a false positive.

---

## Skipping --run-log

### The Mistake

Running `autojudge --results results.tsv` without `--run-log run.log`.

### What Goes Wrong

- Autojudge cannot detect OOM warnings, memory pressure, or training instability.
- The CRASH verdict requires run.log to identify failure point and cause.
- Efficiency metrics (tokens/sec, step time) are missing from the analysis.
- Suggestions are less specific — autojudge can't recommend "reduce batch_size" without memory data.

### Correct Approach

Always include the run log:

```bash
autojudge --results results.tsv --run-log run.log
```

---

## Treating MARGINAL as STRONG_KEEP

### The Mistake

Celebrating a MARGINAL verdict the same way as STRONG_KEEP. "It said keep, so we're winning!"

### What Goes Wrong

- MARGINAL means the improvement is within the noise band (0.5-1.5x noise floor).
- Stacking multiple MARGINAL keeps without changing direction leads to plateau.
- You may be exploiting a direction with diminishing returns instead of exploring new axes.

### Correct Approach

Keep the experiment, but treat it as a signal to consider changing direction. If you see 3+ consecutive MARGINALs, run `autosteer --strategy explore` for fresh ideas.

---

## Cherry-Picking Metrics

### The Mistake

Ignoring the autojudge verdict and focusing on a single metric: "val_bpb went down, so I'll keep it even though autojudge said DISCARD."

### What Goes Wrong

- Autojudge accounts for multiple signals: delta, noise floor, Pareto efficiency, trend.
- Cherry-picking one metric ignores the others. A val_bpb improvement with a 3x regression in tokens/sec is often a net negative.
- Overriding the verdict defeats the purpose of systematic evaluation.

### Correct Approach

Trust the holistic verdict. If you disagree, check the JSON output to understand why the verdict was issued, then adjust your experiment rather than overriding the tool.

---

## Running on Stale results.tsv

### The Mistake

Running autojudge before updating results.tsv with the latest experiment.

### What Goes Wrong

- Autojudge evaluates the last row of results.tsv. If that row is from a previous experiment, the verdict is for the wrong experiment.
- The noise floor estimate is based on stale data.
- Downstream tools (autosteer) will also be working from stale context.

### Correct Approach

Always update results.tsv with the latest experiment before running autojudge. The standard loop is: run training -> update results.tsv -> run autojudge.
