from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from scipy.special import softmax

from .errors import AlreadyFittedError
from ..types import array_like


class Model(ABC):
    """
    The computational model class.
    """

    @abstractmethod
    def fit(
            self, num_choices: int, actions: array_like, rewards: array_like, **kwargs: dict
    ) -> None:
        """

        Parameters
        ----------
        num_choices : int
            The number of choices
        actions : array_like
            The observed choices
        rewards : array_like
            The observed rewards
        kwargs : dict, optional
            The optional parameters
        """
        pass


class QSoftmaxModel(Model):
    def __init__(
            self,
            learning_rate: Union[float, None] = None,
            inverse_temperature: Union[float, None] = None,
    ) -> None:
        """
        This class implements a simple reinforcement learning model. The values are estimated via Q-learning and chooses
        an action using softmax function.

        Parameters
        ----------
        learning_rate : float, default=None
            The learning rate
            If `learning_rate` is None, it is treated as a free parameter to be estimated from data.
            Also, `learning_rate` must be between 0 and 1.
        inverse_temperature : float, default=None
            The inverse temperature
            If `inverse_temperature` is None, it is treated as a free parameter to be estimated from data.
            `inverse_temperature` must be positive.

        """
        if (learning_rate is None) or (0 < learning_rate and learning_rate < 1.0):
            self.learning_rate = learning_rate
        else:
            raise ValueError("`learning_rate must be `None` or between 0 and 1.")

        if (inverse_temperature is None) or (inverse_temperature > 0):
            self.inverse_temperature = inverse_temperature
        else:
            raise ValueError("`inverse_temperature` must be `None` or positive.")

        self.is_fitted = False  # A flag to indicate whether the model is fitted or not.

    def fit(
            self, num_choices: int, actions: array_like, rewards: array_like, **kwargs: dict
    ) -> None:
        """
        Fit the model to data using the likelihood estimate method.

        Parameters
        ----------
        num_choices :
        actions :
        rewards :
        kwargs :

        Returns
        -------

        """
        if self.is_fitted:
            raise AlreadyFittedError(
                "The model has been already fitted. Create a new object to fit again."
            )

    def calculate_nll(
            self,
            a: float,
            b: float,
            num_choices: int,
            actions: array_like,
            rewards: array_like,
    ) -> float:
        """

        Parameters
        ----------
        a : float
            learning rate
        b : float
            inverse temperature
        num_choices : int
            The number of choices
        actions : array_like
            Observed actions
        rewards : array_like
            Observed rewards

        Returns
        -------
        float
            The negative log-likelihood
        """
        if len(actions) != len(rewards):
            raise ValueError("The sizes of `actions` and `rewards` must be the same.")
        if max(actions) > num_choices:
            raise ValueError("The range of `actions` exceeds `num_choices`.")

        time_horizon = len(actions)
        q_vals = np.zeros((time_horizon, num_choices))

        # Update the estimated values, which are called Q values.
        # TODO: There must be numpy way
        for t in range(time_horizon - 1):
            q_vals[t + 1, :] = q_vals[t, :]
            prev_q_val = q_vals[t, actions[t]]
            q_vals[t + 1, actions[t]] = prev_q_val + a * (rewards[t] - prev_q_val)

        choice_probs = softmax(b * q_vals, axis=1)
        neg_log_likelihood = - np.log(choice_probs[np.arange(time_horizon), actions]).sum()
        return neg_log_likelihood
