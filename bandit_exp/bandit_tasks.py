import numpy as np

from abc import ABC, abstractmethod
from typing import Union, Sequence


class Bandit(ABC):
    """This abstract class Defines a bandit task."""
    @abstractmethod
    def pull_arm(self):
        pass


class NormalMultiArmedBandit(Bandit):
    def __init__(
        self, means: Sequence[Union[int, float]], sds: Sequence[Union[int, float]]
    ) -> None:
        if len(means) != len(sds):
            raise ValueError(
                f"lengths of means and sds must match. \
                    len(means)={len(means)}, len(sds)={len(sds)}"
            )

        self.arms = []
        for mean, sd in zip(means, sds):
            self.arms.append(NormalDistArm(mean, sd))

    def pull_arm(self, chosen_arm: int) -> float:
        if (chosen_arm < 0) or (chosen_arm >= len(self.arms)):
            raise ValueError(
                "chosen_arm must be between 0 and len(self.arms). \
                    {chosen_arm} is given."
            )

        return self.arms[chosen_arm].give_reward()


class BernoulliMultiArmedBandit(Bandit):
    def __init__(
        self,
        means: Sequence[Union[int, float]],
    ) -> None:

        self.arms = []
        for mean in means:
            self.arms.append(BernoulliDistArm(mean))

    def pull_arm(self, chosen_arm: int) -> float:
        if (chosen_arm < 0) or (chosen_arm >= len(self.arms)):
            raise ValueError(
                "chosen_arm must be between 0 and len(self.arms). \
                    {chosen_arm} is given."
            )

        return self.arms[chosen_arm].give_reward()


class Arm(ABC):
    def give_reward(self):
        pass


class NormalDistArm(Arm):
    def __init__(self, mean: Union[int, float], sd: Union[int, float]) -> None:
        self.mean = mean
        self.sd = sd

    def give_reward(self) -> float:
        return np.random.normal(self.mean, self.sd)[0]


class BernoulliDistArm(Arm):
    def __init__(self, mean: Union[int, float]) -> None:
        if (mean < 0) or (mean > 1):
            raise ValueError(
                "mean must be between 0 and 1 \
                    {mean} is given."
            )
        self.mean = mean

    def give_reward(self) -> float:
        return np.random.binomial(n=1, p=self.mean, size=1)[0]
