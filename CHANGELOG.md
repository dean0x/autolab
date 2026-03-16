# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-03-16

### Added
- Git worktrees for agent isolation — each agent gets its own directory, enabling concurrent agents
- `cleanup` command to remove worktrees, branches, and evolve.json when an evolve is complete
- `--worktree-dir` option on `init` for custom worktree placement
- Worktree paths shown in `status` and `leaderboard` output

### Fixed
- results.tsv tracking conflict with autoresearch's untracked model — results.tsv is no longer committed
- `evolve.json` discovery from inside agent worktrees
- Pollinate now writes `evolve-hints.md` to each agent's worktree directory

### Changed
- `init` creates git worktrees instead of checking out branches sequentially
- `program.md` now instructs agents to leave results.tsv untracked (matching autoresearch convention)
- results.tsv is read from the filesystem (worktree path) instead of `git show`

## [1.0.0] - 2026-03-16

### Added
- **autojudge**: Statistical experiment evaluation with noise-aware verdicts
- **autosteer**: Data-driven research direction suggestions with explore/exploit strategies
- **autoevolve**: Multi-agent competition orchestrator with 6 built-in strategies
- Claude Code skills for AI agent integration
- CI/CD pipelines (lint, test, publish)
- 285 tests across all three tools
