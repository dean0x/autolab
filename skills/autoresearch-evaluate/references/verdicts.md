# Autoresearch Evaluate — Verdict Reference

Detailed breakdown of each autojudge verdict with real val_bpb examples, edge cases, and JSON output.

---

## Verdict Breakdown

### STRONG_KEEP

Improvement well above the noise floor (3x+). Clear win.

**Example**: val_bpb 4.02 -> 3.95, noise floor 0.02. Delta 0.07 (3.5x noise).

```json
{ "verdict": "STRONG_KEEP", "confidence": 0.97, "delta_pct": -1.74, "noise_floor": 0.02, "on_pareto_frontier": true, "suggestion": "Clear improvement. Commit and explore further in this direction." }
```

### KEEP

Improvement likely real (1.5-3x noise floor). Worth keeping but less definitive.

**Example**: val_bpb 3.95 -> 3.91, noise floor 0.02. Delta 0.04 (2x noise).

```json
{ "verdict": "KEEP", "confidence": 0.82, "delta_pct": -1.01, "noise_floor": 0.02, "on_pareto_frontier": true, "suggestion": "Improvement looks real. Commit and continue." }
```

### MARGINAL

Improvement within the noise band (0.5-1.5x noise floor). Signal is weak.

**Example**: val_bpb 3.91 -> 3.89, noise floor 0.02. Delta 0.02 (1x noise).

```json
{ "verdict": "MARGINAL", "confidence": 0.55, "delta_pct": -0.51, "noise_floor": 0.02, "on_pareto_frontier": false, "suggestion": "Improvement is within noise. Keep if consistent with trend, but watch for plateau." }
```

### RETEST

Improvement indistinguishable from noise (below 0.5x noise floor). Cannot tell if real.

**Example**: val_bpb 3.91 -> 3.905, noise floor 0.01. Delta 0.005 (0.5x noise).

```json
{ "verdict": "RETEST", "confidence": 0.31, "delta_pct": -0.13, "noise_floor": 0.01, "on_pareto_frontier": false, "suggestion": "Cannot distinguish from noise. Re-run same experiment to verify." }
```

**When RETEST is more likely than MARGINAL**: RETEST triggers when the delta is below half the noise floor. On well-converged models with tight noise floors (e.g., 0.005), even tiny deltas like 0.002 land in RETEST. This is common in late-stage optimization.

### DISCARD

Regression detected. The experiment made things worse.

**Example**: val_bpb 3.91 -> 3.94. Delta +0.03 (wrong direction).

```json
{ "verdict": "DISCARD", "confidence": 0.94, "delta_pct": 0.77, "noise_floor": 0.02, "on_pareto_frontier": false, "suggestion": "Regression detected. Revert train.py changes." }
```

### CRASH

Training run failed (OOM or runtime error). No valid val_bpb produced.

**Example**: run.log contains `CUDA out of memory` at step 850 of 1000.

```json
{ "verdict": "CRASH", "confidence": 1.0, "delta_pct": null, "noise_floor": null, "on_pareto_frontier": false, "suggestion": "OOM at step 850. Reduce batch_size or model dimensions." }
```

---

## Edge Cases

### Near-Noise Improvement

**Scenario**: val_bpb 3.91 -> 3.90, noise floor 0.01.
- Delta 0.01 is exactly 1x the noise floor — lands in MARGINAL, not KEEP
- KEEP requires 1.5x+ the noise floor
- Common mistake: treating this as a clear win when it is right at the noise boundary
- If you see this pattern repeatedly, the noise floor may be too wide for the gains you are chasing — consider longer training runs to reduce noise

### OOM Crash Mid-Training (Partial run.log)

**Scenario**: Training crashed at step 600 of 1000. run.log exists but is incomplete.
- autojudge detects the incomplete run from the missing final validation step
- Verdict: CRASH, even if early val_bpb numbers looked promising
- Partial val_bpb is not trustworthy — the model did not converge
- Suggestion includes the failure point and memory context from run.log
- If run.log is missing entirely, autojudge still issues CRASH but with less specific guidance

### Baseline Scenario (First Experiment)

**Scenario**: First experiment — no previous val_bpb to compare against.
- autojudge recognizes a single-row results.tsv as a baseline
- Verdict: STRONG_KEEP (baseline always kept — establishes reference point)
- `delta_pct` and `noise_floor` both `null` (no prior data)

```json
{ "verdict": "STRONG_KEEP", "confidence": 1.0, "delta_pct": null, "noise_floor": null, "on_pareto_frontier": true, "suggestion": "Baseline established. This is your reference point." }
```

### Oscillating Results (Keep-Discard-Keep Pattern)

**Scenario**: val_bpb bouncing: 3.92, 3.88, 3.93, 3.87, 3.94.
- autojudge detects the oscillation pattern in trend analysis
- Even if the latest delta looks like improvement, suggestion flags instability
- Typical suggestion: "Oscillating results. Consider reducing learning rate."
- Verdict reflects the latest delta; suggestion adds strategic context
- Root cause is often learning rate too high or batch size too small — the model overshoots on each experiment
- Three or more oscillations in a row is a strong signal to stabilize before continuing

### Diminishing Returns

**Scenario**: Last five deltas: 0.08, 0.05, 0.03, 0.02, 0.01.
- Each improvement smaller than the last
- autojudge flags diminishing returns in the suggestion field
- Verdict may still be KEEP or MARGINAL, but strategic signal is "change direction"
- Typical suggestion: "Diminishing returns detected. Consider a different optimization axis."
- This is where autoresearch-steer becomes critical — steer can suggest orthogonal axes to explore
