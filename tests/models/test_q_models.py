import numpy as np
import pytest

from bandit_exp.models.q_model import QSoftmaxModel


class TestQSoftmaxModel:
    def test_calculate_nll(self):
        # Setting
        q_model = QSoftmaxModel()
        a = 0.3
        b = 2.0
        num_choices = 2
        actions = np.array([1, 1, 1, 0])
        rewards = np.array([1, 0, 0, 1])
        nll = q_model.calculate_nll(a, b, num_choices, actions, rewards)

        expected_nll = -np.log([0.5, 0.64565631, 0.60348325, 0.42702488]).sum()
        assert nll == pytest.approx(expected_nll)
