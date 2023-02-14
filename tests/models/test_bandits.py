from collections import defaultdict

import numpy as np
import pytest

from bandit_exp.agents.reinforcement_learners import QSoftmax
from bandit_exp.bandit_tasks import BernoulliMultiArmedBandit
from bandit_exp.models.bandits import UcbSoftmaxModel

class TestUcbSoftmaxModel:
    def test_calculate_nll(self):
        # Setting
        model = UcbSoftmaxModel(None, None)
        a = 0.3
        b = 2.0
        num_choices = 2
        actions = np.array([1, 1, 0])
        rewards = np.array([1, 0, 1])
        nll = model.calculate_nll(a, b, num_choices, actions, rewards)

        expected_nll = -np.log([0.5, 0.73105858, 0.54382126]).sum()
        assert nll == pytest.approx(expected_nll)

    def test_fit(self):
        model = UcbSoftmaxModel(None, None)
        a = 0.3
        b = 2.0
        num_choices = 2
        actions = np.array([1, 1, 0])
        rewards = np.array([1, 0, 1])
        model.fit(num_choices, actions, rewards)
