from bandit_exp.types import array_like
from abc import ABC, abstractmethod


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
