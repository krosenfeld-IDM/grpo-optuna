# GRPO Optuna

Experimental GRPO fine-tuning experiments driven by Optuna hyperparameter search.

## Setup

```bash
uv venv .venv
uv sync
```

## Local `.env`

You can set local Weights & Biases defaults in a `.env` file at the repo root. `main.py` loads it on startup.

```env
WANDB_PROJECT=my-project
WANDB_ENTITY=my-team
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


## GSM8K Trial

```bash
uv run python main.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir outputs/Qwen-1.5B-GRPO \
  --run-name Qwen-1.5B-GRPO-gsm8k \
  --report-to wandb \
  --trials 10 \
  --storage sqlite:///db.sqlite3 \
  --no-initial
```
