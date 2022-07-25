import numpy as np

from abc import ABC, abstractmethod
from functools import partial
from typing import Union, Sequence

from scipy.optimize import minimize


class Agent(ABC):
    """
    Agent class is supposed to be in charge of
    choosing actions and updating hyperparameters.
    """

    def __init__(self):
        self.estimated_values = None

    @abstractmethod
    def choose_action(self) -> int:
        """choose_action choose an action."""
        pass

    @abstractmethod
    def learn(self, chosen_action: int, reward: float) -> None:
        """learn is supposed to update hyperparameters in a model."""
        pass

    @abstractmethod
    def fit(self, observed_actions: Sequence[int], observed_rewards: Sequence[int]):
        """estimate free parameters with maximum likelihood estimation."""
        pass


class EpsilonGreedy(Agent):
    """
    EpsilonGreedy implements a agent that follows the epsilon greedy method.
    """

    def __init__(
        self, epsilon: float, initial_values: Sequence[Union[int, float]]
    ) -> None:
        self.epsilon = epsilon
        self.estimated_values = np.array(initial_values, dtype=float)
        self.n_chosen_actions = np.zeros(len(initial_values))

    def choose_action(self) -> int:
        # If there are tie breakers, choose one among them at random.
        max_value = self.estimated_values.max()
        max_index = self.estimated_values[self.choose_action == max_value]
        max_index = np.random.choice(max_index)

        choice_prob = np.tile(
            self.epsilon / (self.estimated_values.shape[0] - 1),
            self.estimated_values.shape[0],
        )
        choice_prob[max_index] = 1 - self.epsilon

        chosen_action = np.random.choice(len(choice_prob), size=1, p=choice_prob)

        # Increment the number of chosen actions.
        self.n_chosen_actions[chosen_action] += 1

        return chosen_action[0]

    def learn(self, chosen_action: int, reward: float) -> None:
        n_chosen_actions = self.n_chosen_actions[chosen_action]
        self.estimated_values[chosen_action] = (
            self.estimated_values[chosen_action] * (n_chosen_actions - 1) + reward
        )
        self.estimated_values /= n_chosen_actions

    def fit(self, observed_actions, observed_rewards) -> Sequence[float]:
        pass

    def calculate_ll(self, observed_actions, observed_rewards) -> Sequence[float]:
        pass


class QSoftmax(Agent):
    """
    QSoftmax implements Q learning model
    and chooses an action using softmax function.
    """

    def __init__(self, learning_rate: float, inverse_temperature: float, initial_values: Sequence[Union[int, float]]) -> None:
        super().__init__()
        self.estimated_values = np.array(initial_values, dtype=float)

        if (0 < learning_rate) and (learning_rate < 1.0):
            self.learning_rate = learning_rate
        else:
            raise ValueError(
                f"learning_rate should be (0, 1), \
                             but the given learning_rate is {learning_rate}"
            )

        self.inverse_temperature = inverse_temperature

        self.estimated_learning_rate = None
        self.estimated_beta = None

    def choose_action(self) -> int:
        action_probs = self._softmax(self.estimated_values, self.inverse_temperature)
        chosen_action = np.random.choice(len(action_probs), size=1, p=action_probs)
        return chosen_action[0]

    def learn(self, chosen_action: int, reward: float) -> None:
        self.estimated_values[chosen_action] = self.estimated_values[
            chosen_action
        ] + self.learning_rate * (reward - self.estimated_values[chosen_action])

    def fit(
            self,
            n_arms: Sequence[int],
            observed_actions: Sequence[int],
            observed_rewards: Sequence[Union[int, float]],
            fixed_free_params: dict,
            init_param,
            initial_q=None,
            fixed_index_q=None,
            const=None,
            options: dict = {"tol": 1e-8, "disp": False, "maxiter": 10000}
    ) -> Sequence[float]:

        params = {}
        params.update(fixed_free_params)
        params["n_arms"] = n_arms
        params["observed_actions"] = observed_actions
        params["observed_rewards"] = observed_rewards
        params["initial_q"] = initial_q
        params["fixed_index_q"] = fixed_index_q

        objective = partial(self.calculate_nll, **params)

        def obj(x):
            return objective(*x)
        res = minimize(
            obj,
            init_param,
            method="COBYLA",
            options=options,
            constraints=const,
        )

        return res

    def calculate_nll(
        self,
        alpha: float,
        beta: float,
        n_arms: int,
        observed_actions: Sequence[int],
        observed_rewards: Union[Sequence[int], Sequence[float]],
        initial_q: Sequence[float] = None,
        fixed_index_q: Sequence[int] = None
    ) -> float:
        n_trials = len(observed_actions)

        Q = np.zeros((n_trials, n_arms), dtype=float)
        if initial_q is not None:
            Q[0] = initial_q

        probs = np.zeros((n_trials, n_arms), dtype=float)
        ll = 0

        for t in range(n_trials):
            probs[t] = self._softmax(Q[t], beta)
            ll += np.log(probs[t, observed_actions[t]] + 1e-8)

            if t < n_trials - 1:
                Q[t + 1] += Q[t]
                if (fixed_index_q is not None) and (observed_actions[t] in fixed_index_q):
                    pass
                else:
                    Q[t + 1, observed_actions[t]] = Q[t, observed_actions[t]] + alpha * (
                        observed_rewards[t] - Q[t, observed_actions[t]]
                    )

        return -ll

    @staticmethod
    def _softmax(x, beta) -> Sequence[float]:
        exponents = beta * x
        max_exp = np.max(exponents)

        # For numerical stability,
        # exponents are subtracted from maximum exponent.
        action_probs = np.exp(exponents - max_exp) / np.sum(np.exp(exponents - max_exp))
        return action_probs