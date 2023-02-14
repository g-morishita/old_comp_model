from abc import ABC, abstractmethod


class Agent(ABC):
    """
    Agent class is supposed to be in charge of
    choosing actions and updating hyperparameters.
    """

    def __init__(self):
        self.estimated_values = None

    @abstractmethod
    def choose_action(self) -> int:
        """choose_action choose an action."""

    @abstractmethod
    def learn(self, chosen_action: int, reward: float) -> None:
        """learn is supposed to update hyperparameters in a model."""
