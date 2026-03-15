# Contributing to autolab

Thanks for your interest in contributing! This project provides companion tools for [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/autolab.git
cd autolab

# Install all tools in development mode
pip install -e ./auto-judge -e ./auto-steer -e ./auto-clash

# Install dev dependencies
pip install pytest ruff mypy

# Run tests
pytest

# Run linting
ruff check .
```

## Project Structure

```
autolab/
├── auto-judge/          # autojudge CLI — experiment evaluation
├── auto-steer/          # autosteer CLI — research direction suggestions
├── auto-clash/          # autoclash CLI — multi-agent competitions
├── skills/              # Claude Code skill definitions
├── templates/           # User-facing integration templates
└── test-data/           # Sample data for testing
```

Each tool is an independent Python package with its own `pyproject.toml`.

## Making Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b my-feature`
3. Make your changes
4. Run tests: `pytest`
5. Run linting: `ruff check .`
6. Commit with a clear message
7. Open a pull request

## Guidelines

- **No fake solutions** — all code must work against real autoresearch data
- **Result types** — use `Ok`/`Err` for error handling, never bare exceptions in business logic
- **Test behaviors** — tests should validate what the tool does, not how it does it
- **Keep tools independent** — each tool should work standalone with only `click` as a dependency
- **Backward compatible** — don't break existing CLI flags or output formats

## Adding a New Tool

1. Create a directory: `auto-<name>/`
2. Add `pyproject.toml` with `hatchling` build backend
3. Add the tool's README
4. Add a Claude Code skill in `skills/autoresearch-<name>/`
5. Update the root README

## Reporting Issues

Please include:
- Which tool (`autojudge`, `autosteer`, or `autoclash`)
- Python version (`python --version`)
- A sample `results.tsv` if relevant (anonymize if needed)
- The full command you ran
- The full error output
