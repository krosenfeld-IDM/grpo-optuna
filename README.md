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
  --trials 25 \
  --per-device-batch 2 \
  --grad-accumulation 2 \
  --storage sqlite:///db.sqlite3 \
  --train-limit 500 \
  --eval-limit 100  \
  --no-initial

Best hyperparameters: {'learning_rate': 1.2555241675979389e-05, 'weight_decay': 0.12962237672897545, 'warmup_ratio': 0.00721201863424088, 'max_grad_norm': 0.6645119843122771, 'xmlcount_weight': 1.1377750171079324, 'soft_format_weight': 1.050673428577614, 'strict_format_weight': 0.7948705296795794, 'int_weight': 1.402971557794846, 'correctness_weight': 1.027168090547948, 'correct_reward': 1.5346880348331404, 'incorrect_reward': -0.6265495394347819, 'int_reward': 0.9198839350962448, 'non_int_reward': -0.05295717260113257, 'strict_match_reward': 0.9840453754466703, 'strict_no_match_reward': -0.3408981349400929, 'soft_match_reward': 0.6391770469303765, 'soft_no_match_reward': -0.315904275182302, 'xml_count_reward': 0.1950116442817907}
```
