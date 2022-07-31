import numpy as np
from scipy.special import softmax
from bandit_exp.types import array_like

from .base import Agent


class UcbSoftmax(Agent):
    """
    `UCB` class implements the upper confidence bound algorithm. The choice values are calculated as follows:
        self.sample_mean + self.delta * np.sqrt(self.sample_variances)
    Based on the value, decide which choice to choose applying the softmax function.
    """

    def __init__(self, delta: float, initial_values: array_like) -> None:
        """

        Parameters
        ----------
        delta : float
            the coefficient for the information bonus. It must be postiive.
        initial_values : array_like
            initial values

        Attributes
        ----------
        self.sample_mean : array_lie
            the sample mean of the observed rewards
        self.sample_variances : array_like
            the sample variance of the observed rewards
        self.choice_counts : array_like
            The number of choices to chosen
        """
        if delta < 0:
            raise ValueError(f"`delta` must be positive")
        else:
            self.delta = delta
        self.sample_means = initial_values
        self.sample_variances = np.title(np.inf, initial_values.shape[0])
        self.choice_counts = np.tile(0, len(initial_values))

    def choose_action(self) -> int:
        """
        Choose an action using softmax function with the values.

        Returns
        -------
            int
                the index of the chosen choice
        """
        # If there are any choices that has never been chosen, choose a choice among them.
        not_chosen_choices = np.where(self.estimated_variances == np.inf)[0]
        if not not_chosen_choices.shape[0]:
            return np.random.choice(not_chosen_choices)

        # Calculation of upper confidence bound.
        choice_values = self.sample_means + self.delta * np.sqrt(self.sample_variances)
        choice_probs = softmax(choice_values)
        chosen_action = np.random.choice(len(choice_probs), p=choice_probs)
        return chosen_action

    def learn(self, chosen_action: int, reward: float) -> None:
        n = self.choice_counts[chosen_action]
        squared_sum = (self.sample_variances[chosen_action] + self.sample_means[chosen_action] ** 2) * (n - 1)
        self.sample_means[chosen_action] = (
            n / (n + 1) * self.sample_means[chosen_action] + 1 / (n + 1) * reward
        )
        self.sample_variances[chosen_action] = 1 / n * (squared_sum + reward ** 2) - self.sample_means[chosen_action] ** 2
        self.choice_counts[chosen_action] += 1
