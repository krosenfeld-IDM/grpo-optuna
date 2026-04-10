# Optuna's Role in the GRPO Training Process

Optuna acts as the outer hyperparameter search loop around GRPO in this repository.
GRPO performs the actual policy optimization and model updates, while Optuna decides
which configuration to try, runs a full GRPO training trial with that configuration,
evaluates the resulting model, and keeps track of the best-performing trial.

For each Optuna trial, the pipeline does four things:

1. Samples a set of hyperparameters such as `learning_rate`, `weight_decay`,
   `warmup_ratio`, `max_grad_norm`, and several reward-shaping parameters.
2. Builds a `GRPOConfig` from those values for the training-side settings.
3. Builds the GRPO reward functions using the sampled reward weights and reward
   magnitudes, then runs a GRPO training job.
4. Evaluates the trained model on the validation set and returns that validation
   accuracy to Optuna as the trial score.

In practice, Optuna is used here to tune two categories of settings:

- Training hyperparameters like learning rate, weight decay, warmup ratio, and
  gradient clipping.
- Reward-shaping hyperparameters like the weights and values for XML formatting,
  soft and strict format checks, integer-answer rewards, and correctness rewards.

This means Optuna is not implementing GRPO itself. Its job is to search for the
best GRPO configuration by comparing the results of multiple training runs.

The CLI can also enqueue a default initial parameter set before the search begins,
which provides a baseline trial for Optuna to compare against sampled trials.
