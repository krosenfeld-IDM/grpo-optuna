# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/grpo_optuna/`. Use `cli.py` for argument parsing and entrypoint flow, `pipeline.py` for training and Optuna orchestration, `data.py` for dataset loading, and `rewards.py` / `text_utils.py` for reward logic and answer parsing. `main.py` is a thin local launcher. Tests live in `tests/`, currently centered on `tests/test_objective.py`. Treat `outputs/` as generated experiment artifacts, not source.

## Build, Test, and Development Commands
Create the environment with `uv venv .venv` and install dependencies with `uv sync`.

Run tests with:
```bash
uv run -m pytest
```

Run the package CLI with:
```bash
uv run grpo-optuna --help
```

Run the documented CPU sanity check with:
```bash
uv run python main.py --model-name hf-internal-testing/tiny-random-gpt2 --output-dir outputs/tiny --run-name tiny --fast-dev-run --report-to none --trials 1 --no-initial
```

## Coding Style & Naming Conventions
Target Python 3.11+ and follow the existing style: 4-space indentation, type hints on public functions, and small focused modules. Use `snake_case` for functions, variables, and module names; use `PascalCase` for classes like `PipelineConfig` and `OptunaObjective`; use `UPPER_SNAKE_CASE` for constants such as default parameter maps. No formatter or linter is configured in this repo, so keep changes consistent with surrounding code.

## Testing Guidelines
Use `pytest`; `pyproject.toml` already sets `testpaths = ["tests"]` and quiet output. Add tests under `tests/test_*.py`. Prefer stubbed unit tests like `test_optuna_objective_with_stubs` for pipeline logic so changes remain fast and hardware-independent. Cover new CLI normalization, reward, and evaluation branches when behavior changes.

## Commit & Pull Request Guidelines
Recent history uses short, imperative subjects, sometimes with prefixes like `chore:`. Keep commit titles concise and specific, for example `cli: normalize generation batch size`. PRs should describe the behavior change, list test coverage, and note any training-impacting defaults or dataset changes. Do not include large checkpoints or generated files from `outputs/` in review unless the PR is explicitly about artifacts.
