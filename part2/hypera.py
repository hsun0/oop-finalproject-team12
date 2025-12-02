import optuna
import optuna
import numpy as np

from frozen_lake import run


def objective(trial: optuna.Trial) -> float:
	"""Optuna objective to maximize success rate of FrozenLake agent.

	Runs 10 independent train+test cycles sequentially and returns mean success rate.
	"""

	# Suggest hyperparameters (you can adjust ranges as needed)
	min_exploration_rate = trial.suggest_float("min_exploration_rate", 1.5e-5, 1.5e-3, log=True)
	epsilon_decay_rate = trial.suggest_float("epsilon_decay_rate", 6.9443e-06, 6.9443e-04, log=True)
	discount_factor_g = 0.975
	start_learning_rate_a = trial.suggest_float("start_learning_rate_a", 0.1, 0.9, log=True)
	min_learning_rate_a = trial.suggest_float("min_learning_rate_a", 1e-3, 0.1, log=True)
	learning_decay_rate = trial.suggest_float("learning_decay_rate", 1e-5, 1e-3, log=True)

	episodes_train = 15000
	episodes_test = 750

	results = []
	for _ in range(10):
		# Train
		run(
			episodes_train,
			is_training=True,
			render=False,
			epsilon_decay_rate=epsilon_decay_rate,
			min_exploration_rate=min_exploration_rate,
			discount_factor_g=discount_factor_g,
			start_learning_rate_a=start_learning_rate_a,
            min_learning_rate_a=min_learning_rate_a,
            learning_decay_rate=learning_decay_rate,
		)

		# Test
		success_rate = run(
			episodes_test,
			is_training=False,
			render=False,
			epsilon_decay_rate=epsilon_decay_rate,
			min_exploration_rate=min_exploration_rate,
			discount_factor_g=discount_factor_g,
			start_learning_rate_a=start_learning_rate_a,
            min_learning_rate_a=min_learning_rate_a,
            learning_decay_rate=learning_decay_rate,
		)
		results.append(success_rate)

	mean_success_rate = float(np.mean(results))
	return mean_success_rate


def main():
	study = optuna.create_study(direction="maximize")
	study.optimize(objective, n_trials=100)

	print("Best value (success rate):", study.best_value)
	print("Best params:")
	for k, v in study.best_params.items():
		print(f"  {k}: {v}")


if __name__ == "__main__":
	main()

