from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import optuna

try:
    import torch
except ImportError:  # pragma: no cover - handled during runtime usage
    torch = None  # type: ignore[assignment]

from .rewards import (
    correctness_reward_func,
    int_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    xmlcount_reward_func,
)
from .text_utils import extract_xml_answer


class TrainerProtocol(Protocol):
    model: Any

    def train(self) -> None: ...


RewardFunction = Callable[..., list[float]]


@dataclass(slots=True)
class PipelineConfig:
    model_name: str
    output_dir: str
    run_name: str
    device: str | None = None
    device_map: str | None = None
    attn_implementation: str | None = None
    use_peft: bool = False
    lora_r: int = 16
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: Sequence[str] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    )
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_generations: int = 16
    generation_batch_size: int | None = None
    max_prompt_length: int = 256
    max_completion_length: int = 256
    num_train_epochs: int = 1
    save_steps: int = 100
    log_on_each_node: bool = False
    report_to: str | Sequence[str] | None = "wandb"
    shuffle_seed: int | None = 0
    train_dataset_limit: int | None = None
    eval_dataset_limit: int | None = None


def detect_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch and hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(device: str, prefer_bf16: bool = True) -> torch.dtype:
    if torch is None:
        raise RuntimeError("PyTorch is required to resolve tensor dtypes.")
    if device.startswith("cuda") and prefer_bf16 and torch.cuda.is_available():
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def build_training_args(config: PipelineConfig, params: Mapping[str, float]):
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=config.output_dir,
        run_name=config.run_name,
        learning_rate=params["learning_rate"],
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=params["weight_decay"],
        warmup_ratio=params["warmup_ratio"],
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=params.get("use_bf16", False),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        generation_batch_size=config.generation_batch_size
        if config.generation_batch_size is not None
        else config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        num_train_epochs=config.num_train_epochs,
        save_steps=config.save_steps,
        max_grad_norm=params["max_grad_norm"],
        report_to=config.report_to,
        log_on_each_node=config.log_on_each_node,
    )


def default_trainer_builder(
    model: Any,
    tokenizer: Any,
    reward_funcs: Sequence[RewardFunction],
    training_args: Any,
    train_dataset: Any,
    peft_config: Any | None,
) -> TrainerProtocol:
    from trl import GRPOTrainer

    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=list(reward_funcs),
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )


def create_model_and_tokenizer(config: PipelineConfig) -> tuple[Any, Any, str]:
    if torch is None:
        raise RuntimeError("PyTorch must be installed to load models.")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = detect_device(config.device)
    dtype = resolve_dtype(device)

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation
    if config.device_map:
        model_kwargs["device_map"] = config.device_map

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    if config.device_map is None:
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}{{ message['content'] }}\n"
            "{% elif message['role'] == 'user' %}User: {{ message['content'] }}\n"
            "{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n"
            "{% else %}{{ message['role'] }}: {{ message['content'] }}\n{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}Assistant:{% endif %}"
        )

    return model, tokenizer, device


def build_peft_config(config: PipelineConfig) -> Any | None:
    if not config.use_peft:
        return None

    from peft import LoraConfig

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.target_modules),
        task_type="CAUSAL_LM",
        lora_dropout=config.lora_dropout,
    )


def build_reward_functions(params: Mapping[str, float]) -> list[RewardFunction]:
    def scale(func: Callable[..., list[float]], weight_key: str, **kwargs: Any) -> RewardFunction:
        weight = params[weight_key]

        def wrapper(*args: Any, **inner_kwargs: Any) -> list[float]:
            values = func(*args, **kwargs, **inner_kwargs)
            return [weight * x for x in values]

        return wrapper

    return [
        scale(xmlcount_reward_func, "xmlcount_weight", xml_count_reward=params["xml_count_reward"]),
        scale(
            soft_format_reward_func,
            "soft_format_weight",
            soft_match_reward=params["soft_match_reward"],
            soft_no_match_reward=params["soft_no_match_reward"],
        ),
        scale(
            strict_format_reward_func,
            "strict_format_weight",
            strict_match_reward=params["strict_match_reward"],
            strict_no_match_reward=params["strict_no_match_reward"],
        ),
        scale(
            int_reward_func,
            "int_weight",
            int_reward=params["int_reward"],
            non_int_reward=params["non_int_reward"],
        ),
        scale(
            correctness_reward_func,
            "correctness_weight",
            correct_reward=params["correct_reward"],
            incorrect_reward=params["incorrect_reward"],
        ),
    ]


def evaluate_model(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    *,
    device: str | "torch.device" | None,
    max_new_tokens: int,
) -> float:
    if torch is None:
        raise RuntimeError("PyTorch must be installed to evaluate models.")
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration as exc:  # pragma: no cover - unexpected parameterless model
            raise RuntimeError("Could not determine model device for evaluation.") from exc

    for sample in dataset:
        prompt = sample["prompt"]
        true_answer = sample["answer"]

        if hasattr(tokenizer, "apply_chat_template"):
            inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt")
        else:
            text = "".join(message["content"] for message in prompt)
            inputs = tokenizer(text, return_tensors="pt")

        if torch.is_tensor(inputs):
            inputs = {"input_ids": inputs}
        elif hasattr(inputs, "items"):
            inputs = dict(inputs)
        else:  # pragma: no cover - unexpected tokenizer output
            raise TypeError("Tokenizer returned unsupported input format for generation.")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id),
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_answer = extract_xml_answer(generated_text)

        if predicted_answer == true_answer:
            correct_predictions += 1
        total_predictions += 1

    return correct_predictions / max(total_predictions, 1)


def sample_hyperparameters(trial: optuna.Trial) -> dict[str, float]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.2),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.1, 1.0),
        "xmlcount_weight": trial.suggest_float("xmlcount_weight", 0.5, 1.5),
        "soft_format_weight": trial.suggest_float("soft_format_weight", 0.5, 1.5),
        "strict_format_weight": trial.suggest_float("strict_format_weight", 0.5, 1.5),
        "int_weight": trial.suggest_float("int_weight", 0.5, 1.5),
        "correctness_weight": trial.suggest_float("correctness_weight", 1.0, 3.0),
        "correct_reward": trial.suggest_float("correct_reward", 1.0, 3.0),
        "incorrect_reward": trial.suggest_float("incorrect_reward", -1.0, 0.0),
        "int_reward": trial.suggest_float("int_reward", 0.1, 1.0),
        "non_int_reward": trial.suggest_float("non_int_reward", -0.5, 0.0),
        "strict_match_reward": trial.suggest_float("strict_match_reward", 0.1, 1.0),
        "strict_no_match_reward": trial.suggest_float("strict_no_match_reward", -0.5, 0.0),
        "soft_match_reward": trial.suggest_float("soft_match_reward", 0.1, 1.0),
        "soft_no_match_reward": trial.suggest_float("soft_no_match_reward", -0.5, 0.0),
        "xml_count_reward": trial.suggest_float("xml_count_reward", 0.05, 0.2),
    }


@dataclass
class OptunaObjective:
    config: PipelineConfig
    train_dataset: Any
    validation_dataset: Any
    trainer_builder: Callable[[Any, Any, Sequence[RewardFunction], Any, Any, Any | None], TrainerProtocol] = default_trainer_builder
    model_loader: Callable[[PipelineConfig], tuple[Any, Any, str]] = create_model_and_tokenizer
    training_args_builder: Callable[[PipelineConfig, Mapping[str, float]], Any] = build_training_args
    reward_factory: Callable[[Mapping[str, float]], Sequence[RewardFunction]] = build_reward_functions
    evaluator: Callable[[Any, Any, Any, str | "torch.device" | None, int], float] = (
        lambda model, tokenizer, dataset, device, max_new_tokens: evaluate_model(
            model,
            tokenizer,
            dataset,
            device=device,
            max_new_tokens=max_new_tokens,
        )
    )
    peft_builder: Callable[[PipelineConfig], Any | None] = build_peft_config

    def __call__(self, trial: optuna.Trial) -> float:
        params = sample_hyperparameters(trial)

        device = detect_device(self.config.device)
        if torch and device.startswith("cuda") and torch.cuda.is_available():
            bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        else:
            bf16_supported = False
        params["use_bf16"] = bf16_supported

        training_args = self.training_args_builder(self.config, params)

        model, tokenizer, device = self.model_loader(self.config)
        reward_funcs = self.reward_factory(params)
        peft_config = self.peft_builder(self.config)

        trainer = self.trainer_builder(
            model,
            tokenizer,
            reward_funcs,
            training_args,
            self.train_dataset,
            peft_config,
        )

        trainer.train()

        return self.evaluator(
            trainer.model,
            tokenizer,
            self.validation_dataset,
            None,
            self.config.max_completion_length,
        )
