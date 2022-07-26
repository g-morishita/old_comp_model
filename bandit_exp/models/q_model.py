import warnings
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from scipy.optimize import minimize, LinearConstraint
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
        Fit the model to data using the maximum likelihood estimation (MLE).

        Parameters
        ----------
        num_choices : int
            The number of choices
        actions : array_like
            The observed choices
        rewards : array_like
            The observed rewards
        kwargs : {"maxiter", "tol", "n_trials"}
            The options for `scipy.optimize.minimize`:
            `n_trials` is how many times you run the minimization problems.
            You need to run it multiple times because a solution might be local optimal.
            The default value for n_trials is 5.

        Returns
        -------

        """
        if self.is_fitted:
            raise AlreadyFittedError(
                "The model has been already fitted. Create a new object to fit again."
            )

        # the options for minimization.
        options = {"maxiter": 10000}
        allowed_keywords = set(["maxiter"])
        for k, v in kwargs.items():
            if k in allowed_keywords:
                options[k] = v
        n_trials = kwargs.get("n_trials", 5)

        # Decide the negative log-likelihood function and the constraints according to the free parameters.
        if (self.learning_rate is None) and (self.inverse_temperature is None):
            def neg_ll(args):
                a, b = args
                return self.calculate_nll(a, b, num_choices, actions, rewards)

            A = np.eye(2)
            lb = np.array([0, 0])
            ub = [1, np.inf]
            const = LinearConstraint(A, lb, ub)

            min_nll = np.inf
            opt_x = None
            for _ in range(n_trials):
                init_a = np.random.beta(2, 2)
                init_b = np.random.gamma(2, 0.333)
                init_param = [init_a, init_b]
                # It was too slow when using COBYLA as a method.
                res = minimize(neg_ll, init_param, method="SLSQP", options=options, constraints=const)
                if not res.success:
                    warnings.warn(res.message)
                else:
                    if min_nll > res.fun:
                        min_nll = res.fun
                        opt_x = res.x

            if opt_x is None:
                warnings.warn("The estimation did not work")
            else:
                self.learning_rate = res.x[0]
                self.inverse_temperature = res.x[1]

        # In the case the inverse temperature is a free parameter.
        elif self.learning_rate is not None:
            def neg_ll(b):
                return self.calculate_nll(self.learning_rate, b, num_choices, actions, rewards)

            A = np.eye(1)
            lb = np.array([0])
            ub = [np.inf]
            const = LinearConstraint(A, lb, ub)

            min_nll = np.inf
            opt_x = None
            for _ in range(n_trials):
                init_b = np.random.gamma(2, 0.333)
                init_param = [init_b]
                res = minimize(neg_ll, init_param, method="COBYLA", options=options, constraints=const)
                if not res.success:
                    warnings.warn(res.message)
                else:
                    if min_nll > res.fun:
                        min_nll = res.fun
                        opt_x = res.x

            if opt_x is None:
                warnings.warn("The estimation did not work")
            else:
                self.inverse_temperature = res.x[0]

        # In the case the learning rate is only a free parameter.
        elif self.inverse_temperature is not None:
            def neg_ll(a):
                return self.calculate_nll(a, self.inverse_temperature, num_choices, actions, rewards)

            A = np.eye(1)
            lb = np.array([0])
            ub = [1]
            const = LinearConstraint(A, lb, ub)

            min_nll = np.inf
            opt_x = None
            for _ in range(n_trials):
                init_a = np.random.beta(2, 2, 1)
                init_param = [init_a]
                res = minimize(neg_ll, init_param, method="COBYLA", options=options, constraints=const)
                if not res.success:
                    warnings.warn(res.message)
                else:
                    if min_nll > res.fun:
                        min_nll = res.fun
                        opt_x = res.x

            if opt_x is None:
                warnings.warn("The estimation did not work")
            else:
                self.learning_rate = res.x[0]
        else:
            raise ValueError("There are no free parameters.")

        self.is_fitted = True
        return min_nll, opt_x

    @staticmethod
    def calculate_nll(
            learning_rate: float,
            inverse_temperature: float,
            num_choices: int,
            actions: array_like,
            rewards: array_like,
    ) -> float:
        """

        Parameters
        ----------
        learning_rate : float
            learning rate
        inverse_temperature : float
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
            q_vals[t + 1, actions[t]] = prev_q_val + learning_rate * (rewards[t] - prev_q_val)

        choice_probs = softmax(inverse_temperature * q_vals, axis=1)
        neg_log_likelihood = - np.log(choice_probs[np.arange(time_horizon), actions] + 1e-8).sum()
        return neg_log_likelihood
