# GRPO Optuna

Experimental GRPO fine-tuning experiments driven by Optuna hyperparameter search.

## Setup

```bash
uv venv .venv
uv sync
```

## Tests

```bash
uv run -m pytest
```

## CPU sanity check

Run a tiny-model sweep to verify the pipeline without GPUs:

```bash
uv run python main.py \
  --model-name hf-internal-testing/tiny-random-gpt2 \
  --output-dir outputs/tiny \
  --run-name tiny \
  --fast-dev-run \
  --report-to none \
  --trials 1 \
  --no-initial
```
