import warnings

import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.special import softmax

from .errors import AlreadyFittedError
from ..types import array_like
from .base import Model


class UcbSoftmaxModel(Model):
    def __init__(self, delta, inverse_temperature):
        """
        This class implements a upper confidence bound learning model. The values are combination of sample means and information bonus.
        It chooses an action using softmax function.

        Parameters
        ----------
        delta : float, default=None
            The coefficient for the information bonus
        inverse_temperature : float, default=None
            The inverse temperature
        """
        self.delta = delta
        self.inverse_temperature = inverse_temperature
        self.is_fitted = False  # A flag to indicate whether the model is fitted or not.

    def fit(self, num_choices: int, actions: array_like, rewards: array_like, **kwargs: dict):
        """
        Fit the model to data using the maximum likelihood estimation.

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
        n_trials = 5
        A = np.eye(2)
        lb = np.array([0, 0])
        ub = [np.inf, np.inf]
        const = LinearConstraint(A, lb, ub)

        def neg_nll(params):
            return self.calculate_nll(params[0], params[1], num_choices, actions, rewards)

        min_nll = np.inf
        opt_x = None
        for _ in range(n_trials):
            init_param = np.random.gamma(2, 2, size=2)
            res = minimize(neg_nll, init_param, method="COBYLA", constraints=const)
            if not res.success:
                warnings.warn(res.message)
            else:
                if min_nll > res.fun:
                    min_nll = res.fun
                    opt_x = res.x
        if opt_x is None:
            warnings.warn("The estimation did not work")
        else:
            self.delta = res.x[0]
            self.inverse_temperature = res.x[1]

    @staticmethod
    def calculate_nll(
            delta,
            inverse_temperature,
            n_choices,
            choices,
            observed_rewards
    ):
        """

        Parameters
        ----------
        delta :
        inverse_temperature :
        n_choices :
        choices :
        observed_rewards :

        Returns
        -------

        """
        time_horizon = len(choices)
        values = np.zeros((time_horizon, n_choices))
        choice_counts = np.zeros((time_horizon, n_choices))
        sample_means = np.zeros((time_horizon, n_choices))
        sample_means[0, :] = np.tile(0.5, n_choices)
        values[0, :] = sample_means[0, :] + delta

        for t in range(time_horizon - 1):
            sample_means[t + 1, :] = sample_means[t, :]
            choice_counts[t + 1, :] = choice_counts[t, :]
            values[t + 1, :] = values[t, :]

            n = choice_counts[t, choices[t]]
            sample_means[t + 1, choices[t]] = n / (n + 1) * sample_means[t, choices[t]] + 1 / (n + 1) * \
                                              observed_rewards[t]
            values[t + 1, choices[t]] = sample_means[t + 1, choices[t]] + delta / np.sqrt(n + 1)

            choice_counts[t + 1, choices[t]] += 1

        choice_probs = softmax(inverse_temperature * values, axis=1)
        neg_log_likelihood = - np.log(choice_probs[np.arange(time_horizon), choices] + 1e-8).sum()
        return neg_log_likelihood
