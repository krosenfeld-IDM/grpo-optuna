from __future__ import annotations

import argparse
from typing import Any

import optuna

from .data import get_gsm8k_questions
from .pipeline import OptunaObjective, PipelineConfig, detect_device


DEFAULT_INITIAL_PARAMS: dict[str, float] = {
    "learning_rate": 5e-6,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "max_grad_norm": 0.1,
    "xmlcount_weight": 1.0,
    "soft_format_weight": 1.0,
    "strict_format_weight": 1.0,
    "int_weight": 1.0,
    "correctness_weight": 2.0,
    "correct_reward": 2.0,
    "incorrect_reward": 0.0,
    "int_reward": 0.5,
    "non_int_reward": 0.0,
    "strict_match_reward": 0.5,
    "strict_no_match_reward": 0.0,
    "soft_match_reward": 0.5,
    "soft_no_match_reward": 0.0,
    "xml_count_reward": 0.125,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRPO fine-tuning with Optuna")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="outputs/Qwen-1.5B-GRPO")
    parser.add_argument("--run-name", default="Qwen-1.5B-GRPO-gsm8k")
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--use-peft", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--num-generations", type=int, default=16)
    parser.add_argument("--generation-batch-size", type=int, default=None)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--per-device-batch", type=int, default=1)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--log-on-each-node", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true", help="Enable fast CPU-friendly settings")
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default=None)
    parser.add_argument("--load-if-exists", action="store_true")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--no-initial", action="store_true", help="Do not enqueue the default initial trial")
    return parser.parse_args()


def apply_fast_profile(args: argparse.Namespace) -> None:
    if not args.fast_dev_run:
        return

    args.train_limit = args.train_limit or 8
    args.eval_limit = args.eval_limit or 8
    args.num_generations = min(args.num_generations, 2)
    if args.generation_batch_size is None:
        args.generation_batch_size = args.num_generations
    args.grad_accumulation = min(args.grad_accumulation, 1)
    args.max_completion_length = min(args.max_completion_length, 128)


def normalize_generation_batch_size(args: argparse.Namespace) -> None:
    if args.generation_batch_size is None:
        return

    if args.generation_batch_size < args.num_generations:
        args.generation_batch_size = args.num_generations
    elif args.generation_batch_size % args.num_generations != 0:
        multiplier = (args.generation_batch_size // args.num_generations) + 1
        args.generation_batch_size = multiplier * args.num_generations


def build_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        run_name=args.run_name,
        device=args.device,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        use_peft=args.use_peft,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accumulation,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        log_on_each_node=args.log_on_each_node,
        report_to=args.report_to,
        shuffle_seed=args.shuffle_seed,
        train_dataset_limit=args.train_limit,
        eval_dataset_limit=args.eval_limit,
    )


def load_datasets(config: PipelineConfig) -> tuple[Any, Any]:
    train_dataset = get_gsm8k_questions(
        "train",
        limit=config.train_dataset_limit,
        shuffle_seed=config.shuffle_seed,
    )
    eval_dataset = get_gsm8k_questions(
        "test",
        limit=config.eval_dataset_limit,
        shuffle_seed=config.shuffle_seed,
    )
    return train_dataset, eval_dataset


def main() -> None:
    args = parse_args()
    apply_fast_profile(args)
    normalize_generation_batch_size(args)

    config = build_pipeline_config(args)
    train_dataset, eval_dataset = load_datasets(config)

    objective = OptunaObjective(config=config, train_dataset=train_dataset, validation_dataset=eval_dataset)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.load_if_exists,
    )

    if not args.no_initial:
        study.enqueue_trial(DEFAULT_INITIAL_PARAMS)

    study.optimize(objective, n_trials=args.trials)

    device = detect_device(config.device)
    print(f"Study completed on device: {device}")
    print("Best hyperparameters:", study.best_trial.params)
