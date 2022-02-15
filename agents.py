import numpy as np

from abc import ABC, abstractclassmethod
from typing import Union, Sequence


class Agent(ABC):
    """
    Agent class is supposed to be in charge of
    choosing actions and updating hyperparameters.
    """
    def __init__(self):
        self.estimated_values = None

    @abstractclassmethod
    def choose_action(self):
        """choose_action choose an action."""
        pass

    @abstractclassmethod
    def learn(self, chosen_action: int, reward: float):
        """learn is supposed to update hyper parameters in a model."""
        pass


class EpsilonGreedy(Agent):
    """
    EpsilonGreedy implements a agent that follows the epsilon greeddy method.
    """
    def __init__(
            self,
            epsilon: float,
            initial_values: Sequence[Union[int, float]]
            ) -> None:
        self.epsilon = epsilon
        self.estimated_values = np.array(initial_values, dtype=float)
        self.n_chosen_actions = np.zeros(len(initial_values))

    def choose_action(self) -> int:
        # If there are tie breakers, choose one among them at random.
        max_value = self.estimated_values.max()
        max_index = self.estimated_values[self.choose_action == max_value]
        max_index = np.random.choice(max_index)

        choice_prob = np.tile(self.epsilon / (self.estimated_values.shape[0] - 1), self.estimated_values.shape[0])
        choice_prob[max_index] = 1 - self.epsilon

        chosen_action = np.random.choice(
                len(choice_prob),
                size=1,
                p=choice_prob
                )

        # Increment the number of chosen actions.
        self.n_chosen_actions[chosen_action] += 1

        return chosen_action[0]

    def learn(self, chosen_action: int, reward: float) -> None:
        n_chosen_actions = self.n_chosen_actions[chosen_action]
        self.estimated_values[chosen_action] = self.estimated_values[chosen_action] * (n_chosen_actions - 1) + reward
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
            initial_values: Sequence[Union[int, float]]
            ) -> None:
        self.estimated_values = np.array(initial_values, dtype=float)

        if (0 < learning_rate) and (learning_rate < 1.0):
            self.learning_rate = learning_rate
        else:
            raise ValueError(f"learning_rate should be (0, 1), \
                             but the given learning_rate is {learning_rate}")

        self.inverse_temperature = inverse_temperature

    def choose_action(self) -> int:
        action_probs = self._softmax()
        chosen_action = np.random.choice(
                len(action_probs),
                size=1,
                p=action_probs
                )
        return chosen_action[0]

    def learn(self, chosen_action: int, reward: float) -> None:
        self.estimated_values[chosen_action] = self.estimated_values[chosen_action] + self.learning_rate * (reward - self.estimated_values[chosen_action])

    def _softmax(self) -> Sequence[float]:
        exponents = self.inverse_temperature * self.estimated_values
        max_exp = np.max(exponents)

        # For numerical stability,
        # exponents are subtracted from maximum exponent.
        action_probs = np.exp(exponents - max_exp) / np.sum(np.exp(exponents - max_exp))
        return action_probs
