import pytest
import numpy as np
from bandit_exp.agents import bandits

from collections import Counter


class TestQLearnerSoftmax:
    def test_invalid_delta(self):
        with pytest.raises(ValueError):
            bandits.UcbSoftmax(0, 2.0, [0, 0, 0])

    def test_invalid_delta2(self):
        with pytest.raises(ValueError):
            bandits.UcbSoftmax(-1.0, 2.0, [0, 0, 0])

    # check if learn works.
    def test_learn(self):
        agent = bandits.UcbSoftmax(0.1, 2.0, [0, 0, 0])
        agent.learn(0, 1.0)
        assert agent.sample_means[0] == pytest.approx(1.0)
        assert agent.sample_means[1] == pytest.approx(0)
        assert agent.choice_counts[0] == 1
        assert agent.choice_counts[1] == 0

        agent.learn(0, 0)
        assert agent.sample_means[0] == pytest.approx(0.5)
        assert agent.sample_means[1] == pytest.approx(0)
        assert agent.choice_counts[0] == 2
        assert agent.choice_counts[1] == 0

        agent.learn(1, 1.0)
        assert agent.sample_means[0] == pytest.approx(0.5)
        assert agent.sample_means[1] == pytest.approx(1.0)
        assert agent.choice_counts[0] == 2
        assert agent.choice_counts[1] == 1

        agent.learn(1, 1.0)
        assert agent.sample_means[0] == pytest.approx(0.5)
        assert agent.sample_means[1] == pytest.approx(1.0)
        assert agent.choice_counts[0] == 2
        assert agent.choice_counts[1] == 2

    def test_uniform_choose_action(self):
        n_actions = 3
        agent = bandits.UcbSoftmax(0.1, 2.0, [0 for _ in range(n_actions)])

        n_trials = 30000
        action_counter = Counter()
        for _ in range(n_trials):
            chosen_action = agent.choose_action()
            action_counter[chosen_action] += 1

        exp_n_actions = n_trials / n_actions

        chi2_stats = [
            (obs_n_actions - exp_n_actions) ** 2 / exp_n_actions
            for obs_n_actions in action_counter.values()
        ]
        chi2_stats = sum(chi2_stats)

        # Pr(chi2(2) < 4.605) = 0.9
        assert chi2_stats < 4.605

    def test_uniform_choose_action2(self):
        pass

    def test_nonuniform_choose_action2(self):
        pass