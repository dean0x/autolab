<!-- This is a user-facing template to append to your autoresearch program.md.
     It is NOT a Claude Code skill. -->

# Program Addon — Autoresearch Companion Tools

Append this to your autoresearch `program.md` to integrate the companion tools into the experiment loop.

---

## Experiment Evaluation

After every experiment, evaluate results with `autojudge` instead of manually comparing val_bpb:

```bash
autojudge --results results.tsv --run-log run.log
```

Use the verdict to decide your next action:
- **STRONG_KEEP / KEEP**: Commit and advance. The improvement is real.
- **MARGINAL**: Keep, but note that gains are small. Consider a different direction.
- **RETEST**: The change is within noise. Run it again to verify.
- **DISCARD / CRASH**: Revert and try something else.

For scripting, use exit codes: `autojudge --results results.tsv && git commit -m "keep" || git reset --hard HEAD~1`

## Getting Unstuck

If you hit 3 or more consecutive discards, use `autosteer` to find a new direction:

```bash
autosteer --results results.tsv --strategy explore
```

Pick the top-ranked suggestion and implement it. Autosteer accounts for what you've already tried and won't repeat failed directions.

## Experiment Loop (Modified)

1. Read the top suggestion from `autosteer --results results.tsv --quiet`
2. Implement the change in `train.py` and commit
3. Run training: `uv run train.py > run.log 2>&1`
4. Evaluate: `autojudge --results results.tsv --run-log run.log --format json`
5. Parse the JSON verdict to decide keep/discard
6. Update `results.tsv` and commit
7. If 3+ consecutive discards, run `autosteer --strategy explore` for new ideas
8. Repeat
