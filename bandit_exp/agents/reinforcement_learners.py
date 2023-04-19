import math

import numpy as np
from bandit_exp.types import array_like
from scipy.special import softmax

from .base import Agent


class EpsilonGreedy(Agent):
    """
    EpsilonGreedy implements a agent that follows the epsilon greedy method.
    """

    def __init__(self, epsilon: float, initial_values: array_like) -> None:
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


class QSoftmax(Agent):
    """
    QSoftmax implements Q learning model
    and chooses an action using softmax function.
    """

    def __init__(
        self,
        learning_rate: float,
        inverse_temperature: float,
        initial_values: array_like,
    ) -> None:
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

    def choose_action(self) -> int:
        action_probs = softmax(self.estimated_values * self.inverse_temperature)
        chosen_action = np.random.choice(len(action_probs), size=1, p=action_probs)
        return chosen_action[0]

    def learn(self, chosen_action: int, reward: float) -> None:
        self.estimated_values[chosen_action] = self.estimated_values[
            chosen_action
        ] + self.learning_rate * (reward - self.estimated_values[chosen_action])


class ActionLearner(Agent):
    """
    This model is adapted from Burke, Christopher J., et al. "Neural mechanisms of observational learning."
    Proceedings of the National Academy of Sciences 107.32 (2010): 14431-14436.
    Assume that a learner only observe others' choice.
    This model only deals with a situation where there are two actions.
    """

    def __init__(self, learning_rate: float, initial_probs: array_like) -> None:
        super().__init__()
        if len(initial_probs) != 2:
            raise ValueError(
                f"The number of actions should be two, so the length of `initial_probs` has to be two. "
                f"But that of the given `initial_probs` is {len(initial_probs)}"
            )

        if not np.isclose(np.sum(initial_probs), 1.0):
            raise ValueError(
                f"The sum of `initial_probs` has to add to 1."
                f"But that of the given is {np.sum(initial_probs)}"
            )

        self.estimated_values = np.array(initial_probs, dtype=float)

        if (0 < learning_rate) and (learning_rate < 1.0):
            self.learning_rate = learning_rate
        else:
            raise ValueError(
                f"learning_rate should be (0, 1), \
                             but the given learning_rate is {learning_rate}"
            )

    def choose_action(self) -> int:
        chosen_action = np.random.choice(
            len(self.estimated_values), size=1, p=self.estimated_values
        )
        return chosen_action[0]

    def learn(self, chosen_action: int, reward=None) -> None:
        observed = int(chosen_action == 0)
        self.estimated_values[0] = self.estimated_values[0] + self.learning_rate * (observed - self.estimated_values[0])
        self.estimated_values[1] = 1 - self.estimated_values[0]