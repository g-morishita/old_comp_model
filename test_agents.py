import pytest
import numpy as np
import agents

from collections import Counter


class TestAgents:
    def test_abstract_method_error(self):
        class AgentWithoutLearn(agents.Agent):
            def choose_action(self):
                pass

        with pytest.raises(TypeError):
            AgentWithoutLearn()

    def test_abstract_method_error2(self):
        class AgentWithoutChooseAction(agents.Agent):
            def learn(self):
                pass

        with pytest.raises(TypeError):
            AgentWithoutChooseAction()


class TestQLearnerSoftmax:
    # learning rate is more than 1.
    def test_excess_learning_rate(self):
        with pytest.raises(ValueError):
            agents.Q_learner_softmax(
                    1.1,
                    0.1,
                    [0, 0, 0]
                    )

    # learning rate is negative.
    def test_negative_learning_rate(self):
        with pytest.raises(ValueError):
            agents.Q_learner_softmax(
                    -0.1,
                    0.1,
                    [0, 0, 0]
                    )

    # check if learn works.
    def test_learn(self):
        q_learner = agents.Q_learner_softmax(
                0.1,
                2.0,
                [0, 0, 0]
                )
        q_learner.learn(0, 1.0)
        assert q_learner.estimated_values[0] == pytest.approx(0.1)
        assert q_learner.estimated_values[1] == pytest.approx(0.0)
        assert q_learner.estimated_values[2] == pytest.approx(0.0)

        q_learner.learn(0, 0)
        assert q_learner.estimated_values[0] == pytest.approx(0.09)
        assert q_learner.estimated_values[1] == pytest.approx(0.0)
        assert q_learner.estimated_values[2] == pytest.approx(0.0)

        q_learner.learn(1, 1.0)
        assert q_learner.estimated_values[0] == pytest.approx(0.09)
        assert q_learner.estimated_values[1] == pytest.approx(0.1)
        assert q_learner.estimated_values[2] == pytest.approx(0.0)

        q_learner.learn(1, 1.0)
        assert q_learner.estimated_values[0] == pytest.approx(0.09)
        assert q_learner.estimated_values[1] == pytest.approx(0.19)
        assert q_learner.estimated_values[2] == pytest.approx(0.0)

        q_learner.learn(2, -1.0)
        assert q_learner.estimated_values[0] == pytest.approx(0.09)
        assert q_learner.estimated_values[1] == pytest.approx(0.19)
        assert q_learner.estimated_values[2] == pytest.approx(-0.1)

        q_learner.learn(2, -1.0)
        assert q_learner.estimated_values[0] == pytest.approx(0.09)
        assert q_learner.estimated_values[1] == pytest.approx(0.19)
        assert q_learner.estimated_values[2] == pytest.approx(-0.19)

    def test_uniform_choose_action(self):
        n_actions = 3
        q_learner = agents.Q_learner_softmax(
                0.1,
                2.0,
                [0 for _ in range(n_actions)]
                )

        n_trials = 30000
        action_counter = Counter()
        for _ in range(n_trials):
            chosen_action = q_learner.choose_action()
            action_counter[chosen_action] += 1

        exp_n_actions = n_trials / n_actions

        chi2_stats = [(obs_n_actions - exp_n_actions) ** 2 / exp_n_actions
                      for obs_n_actions in action_counter.values()]
        chi2_stats = sum(chi2_stats)

        # Pr(chi2(2) < 4.605) = 0.9
        assert chi2_stats < 4.605

    def test_uniform_choose_action2(self):
        # inverse_temperature equals 0, which means uniformly random.
        n_actions = 3
        q_learner = agents.Q_learner_softmax(
                0.1,
                0,
                [100, -1, 40]
                )

        n_trials = 30000
        action_counter = Counter()
        for _ in range(n_trials):
            chosen_action = q_learner.choose_action()
            action_counter[chosen_action] += 1

        exp_n_actions = n_trials / n_actions

        chi2_stats = [(obs_n_actions - exp_n_actions) ** 2 / exp_n_actions
                      for obs_n_actions in action_counter.values()]
        chi2_stats = sum(chi2_stats)

        # Pr(chi2(2) < 4.605) = 0.9
        assert chi2_stats < 4.605

    def test_nonuniform_choose_action2(self):
        initial_values = np.array([1, 2, 3])
        q_learner = agents.Q_learner_softmax(
                0.1,
                1.0,
                initial_values
                )

        n_trials = 30000
        action_counter = np.zeros(3)
        for _ in range(n_trials):
            chosen_action = q_learner.choose_action()
            action_counter[chosen_action] += 1

        exp_n_actions = n_trials * np.exp(initial_values) / np.exp(initial_values).sum()
        chi2_stats = ((exp_n_actions - action_counter) ** 2 / exp_n_actions).sum()

        # Pr(chi2(2) < 4.605) = 0.9
        assert chi2_stats < 4.605
