from __future__ import annotations

import optuna
import pytest

from grpo_optuna.pipeline import OptunaObjective, PipelineConfig


class DummyModel:
    def __init__(self) -> None:
        self.config = type("Config", (), {})()

    def eval(self) -> None:
        pass


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, prompt, return_tensors: str = "pt"):
        return {"input_ids": [[0]]}

    def decode(self, tokens, skip_special_tokens: bool = True) -> str:
        return "<answer>42</answer>"


class DummyTrainer:
    def __init__(self, model, tokenizer, *_args, **_kwargs) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_called = False

    def train(self) -> None:
        self.train_called = True


def test_optuna_objective_with_stubs():
    config = PipelineConfig(
        model_name="stub-model",
        output_dir="./outputs",
        run_name="test-run",
        num_generations=1,
        max_completion_length=16,
    )

    train_dataset = [{"prompt": [], "answer": "42"}]
    eval_dataset = [{"prompt": [], "answer": "42"}]

    trainer_store: dict[str, DummyTrainer] = {}

    def trainer_builder(model, tokenizer, reward_funcs, training_args, dataset, peft_config):
        assert reward_funcs
        trainer = DummyTrainer(model, tokenizer)
        trainer_store["trainer"] = trainer
        return trainer

    objective = OptunaObjective(
        config=config,
        train_dataset=train_dataset,
        validation_dataset=eval_dataset,
        trainer_builder=trainer_builder,
        model_loader=lambda _config: (DummyModel(), DummyTokenizer(), "cpu"),
        training_args_builder=lambda _config, params: {"params": params},
        reward_factory=lambda params: [lambda *args, **kwargs: [1.0]],
        evaluator=lambda model, tokenizer, dataset, device, max_new_tokens: 0.5,
        peft_builder=lambda _config: None,
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    assert pytest.approx(study.best_trial.value, rel=1e-6) == 0.5
    assert trainer_store["trainer"].train_called
