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

    def __init__(self, learning_rate: float, inverse_temperature: float, initial_values: array_like) -> None:
        super().__init__()
        if len(initial_values) != 2:
            raise ValueError(
                f"The number of actions should be two, so the length of `initial_probs` has to be two. "
                f"But that of the given `initial_probs` is {len(initial_values)}"
            )

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

    def learn(self, chosen_action: int, reward=None) -> None:
        self.estimated_values[chosen_action] = self.estimated_values[chosen_action] + self.learning_rate * (
                    1 - self.estimated_values[chosen_action])

        self.estimated_values[1 - chosen_action] = 1 - self.estimated_values[chosen_action]


class QActionSoftmax(Agent):
    """
    QActionSoftmax implements a hybrid model of Q learning and Action learning models.
    The values of options are based on action history and reward history.
    The model chooses an action using softmax function.

    Parameters
    ----------
    lr_a : float
        The learning rate for Q value
    lr_a : float
        The learning rate for action value
    inverse_temperature : float
        The inverse temperature for the combined value
    weight : float
        The weight of Q value. (1 - w) is the weight of the action value
        The combined value is calculated like this:
    initial_values_q : array_like
        The initial values for Q values
    initial_values_a : array_like
        The initial values for action values
    """

    def __init__(
        self,
        lr_q: float,
        lr_a: float,
        inverse_temperature: float,
        weight: float,
        initial_values_q: array_like,
        initial_values_a: array_like
    ) -> None:
        super().__init__()
        self.q_values = np.array(initial_values_q, dtype=float)
        self.a_values = np.array(initial_values_a, dtype=float)

        if (0 < lr_q) and (lr_q < 1.0):
            self.lr_q = lr_q
        else:
            raise ValueError(
                f"lr_q should be (0, 1), \
                             but the given learning_rate is {lr_q}"
            )

        if (0 < lr_a) and (lr_a < 1.0):
            self.lr_a = lr_a
        else:
            raise ValueError(
                f"lr_a should be (0, 1), \
                             but the given learning_rate is {lr_q}"
            )

        if inverse_temperature >= 0:
            self.inverse_temperature = inverse_temperature
        else:
            raise ValueError(
                f"inverse_temperature should be non-negative, but {inverse_temperature} is given"
            )

        if (0 <= weight) and (weight <= 1.0):
            self.weight = weight
        else:
            raise ValueError(
                f"w should be (0, 1), \
                             but the given weight is {weight}"
            )

    def choose_action(self) -> int:
        values = self.weight * self.q_values + (1 - self.weight) * self.a_values
        action_probs = softmax(values * self.inverse_temperature)
        chosen_action = np.random.choice(len(action_probs), size=1, p=action_probs)
        return chosen_action[0]

    def learn(self, chosen_action: int, reward) -> None:
        # Update Q values
        self.q_values[chosen_action] = self.q_values[chosen_action] + self.lr_q * (
                    1 - self.q_values[chosen_action])

        # Update Action values
        self.a_values[chosen_action] = self.a_values[chosen_action] + self.lr_a * (
                    1 - self.a_values[chosen_action])

        not_chosen_actions = np.delete(np.arange(len(self.a_values)), chosen_action)
        self.a_values[not_chosen_actions] = self.a_values[not_chosen_actions] + self.lr_a * (
                    0 - self.a_values[not_chosen_actions])