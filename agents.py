import numpy as np

from typing import Union, Sequence


class Agent:
    """Agent class is supposed to be in charge of choosing actions."""
    def __init__(self):
        self.estimated_values = None

    def choose_action(self):
        raise NotImplementedError

    def learn(self, chosen_action: int, reward: float):
        raise NotImplementedError


class Q_learner_softmax(Agent):
    """
    Q_learner_softmax implements Q learning model
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
        action_probs = self.softmax()
        chosen_action = np.random.choice(
                len(action_probs),
                size=1,
                p=action_probs
                )
        return chosen_action[0]

    def learn(self, chosen_action: int, reward: float) -> None:
        self.estimated_values[chosen_action] = self.estimated_values[chosen_action] + self.learning_rate * (reward - self.estimated_values[chosen_action])

    def softmax(self) -> Sequence[float]:
        exponents = self.inverse_temperature * self.estimated_values
        max_exp = np.max(exponents)

        # For numerical stability,
        # exponents are subtracted from maximum exponent.
        action_probs = np.exp(exponents - max_exp) / np.sum(np.exp(exponents - max_exp))
        return action_probs
