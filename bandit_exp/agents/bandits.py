import numpy as np
from bandit_exp.types import array_like
from scipy.special import softmax

from .base import Agent


class UcbSoftmax(Agent):
    """
    `UCB` class implements the upper confidence bound algorithm. The choice values are calculated as follows:
        self.sample_mean + self.delta * np.sqrt(1 / self.choice_counts)
    Based on the value, decide which choice to choose applying the softmax function.
    """

    def __init__(self, delta: float, inverse_temperature: float, initial_values: array_like) -> None:
        """
        Parameters
        ----------
        delta : float
            the coefficient for the information bonus. It must be positive.
        inverse_temperature : float
            the inverse temperature that controls the noise
        initial_values : array_like
            initial values

        Attributes
        ----------
        self.sample_mean : array_lie
            the sample mean of the observed rewards
        self.choice_counts : array_like
            The number of choices to chosen
        """
        super().__init__()
        if delta <= 0:
            raise ValueError(f"`delta` must be positive")
        else:
            self.delta = delta

        if inverse_temperature <= 0:
            raise ValueError(f"`inverse_temperature must be positive")
        else:
            self.inverse_temperature = inverse_temperature

        self.sample_means = initial_values
        self.choice_counts = np.zeros(len(initial_values))

    def choose_action(self) -> int:
        """
        Choose an action using softmax function with the values.

        Returns
        -------
            int
                the index of the chosen choice
        """
        # If there are any choices that has never been chosen, choose a choice among them.
        # not_chosen_choices = np.where(self.choice_counts == 0)[0]
        # if not_chosen_choices.shape[0]:
        #    return np.random.choice(not_chosen_choices)

        # Calculation of upper confidence bound.
        choice_values = self.sample_means + self.delta * np.sqrt(1 / (self.choice_counts + 1))
        choice_probs = softmax(choice_values * self.inverse_temperature)
        chosen_action = np.random.choice(len(choice_probs), p=choice_probs)
        return chosen_action

    def learn(self, chosen_action: int, reward: float) -> None:
        n = self.choice_counts[chosen_action]  # How many times the option has been chosen?

        # Update the sample mean using the previous value of the sample mean.
        self.sample_means[chosen_action] = (
            n / (n + 1) * self.sample_means[chosen_action] + 1 / (n + 1) * reward
        )

        # Increment how many times the option has been chosen by 1.
        self.choice_counts[chosen_action] += 1
