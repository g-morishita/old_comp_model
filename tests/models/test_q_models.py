from collections import defaultdict

import numpy as np
import pytest

from bandit_exp.agents.agents import QSoftmax
from bandit_exp.bandit_tasks import BernoulliMultiArmedBandit
from bandit_exp.models.q_model import QSoftmaxModel

# Global Variables
learning_rate = 0.1
inverse_temperature = 2.0


def create_dataset():
    means = [0.7, 0.4]
    initial_values = [0, 0]
    time_horizon = 1000
    q_learner = QSoftmax(learning_rate, inverse_temperature, initial_values)
    two_armed_bandit = BernoulliMultiArmedBandit(means)

    history = defaultdict(list)
    for _ in range(time_horizon):
        chosen_action = q_learner.choose_action()
        reward = two_armed_bandit.pull_arm(chosen_action)
        history["reward"].append(reward)
        history["actions"].append(chosen_action)
        history["estimated_vals"].append(np.copy(q_learner.estimated_values))
        q_learner.learn(chosen_action, reward)

    return history


class TestQSoftmaxModel:
    history = create_dataset()

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

    def test_calculate_nll2(self):
        # TODO: In the case where the number of choices is 3.
        pass

    def test_fit(self):
        q_model = QSoftmaxModel(inverse_temperature=inverse_temperature)
        assert q_model.learning_rate is None
        actions = self.history["actions"]
        rewards = self.history["reward"]
        q_model.fit(2, actions, rewards)
        assert q_model.learning_rate is not None

    def test_fit2(self):
        q_model = QSoftmaxModel(learning_rate=learning_rate)
        assert q_model.inverse_temperature is None
        actions = self.history["actions"]
        rewards = self.history["reward"]
        q_model.fit(2, actions, rewards)
        assert q_model.inverse_temperature is not None

    def test_fit3(self):
        q_model = QSoftmaxModel()
        assert q_model.learning_rate is None
        assert q_model.inverse_temperature is None
        actions = self.history["actions"]
        rewards = self.history["reward"]
        q_model.fit(2, actions, rewards)
        assert q_model.learning_rate is not None
        assert q_model.inverse_temperature is not None
