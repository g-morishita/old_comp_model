import pytest
import numpy as np
from scipy.special import softmax
from bandit_exp.agents import reinforcement_learners
from bandit_exp.agents.base import Agent

from collections import Counter


class TestAgents:
    def test_abstract_method_error(self):
        class AgentWithoutLearn(Agent):
            def choose_action(self):
                pass

        with pytest.raises(TypeError):
            AgentWithoutLearn()

    def test_abstract_method_error2(self):
        class AgentWithoutChooseAction(Agent):
            def learn(self):
                pass

        with pytest.raises(TypeError):
            AgentWithoutChooseAction()


class TestQLearnerSoftmax:
    # learning rate is more than 1.
    def test_excess_learning_rate(self):
        with pytest.raises(ValueError):
            reinforcement_learners.QSoftmax(1.1, 0.1, [0, 0, 0])

    # learning rate is negative.
    def test_negative_learning_rate(self):
        with pytest.raises(ValueError):
            reinforcement_learners.QSoftmax(-0.1, 0.1, [0, 0, 0])

    # check if learn works.
    def test_learn(self):
        q_learner = reinforcement_learners.QSoftmax(0.1, 2.0, [0, 0, 0])
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
        q_learner = reinforcement_learners.QSoftmax(0.1, 2.0, [0 for _ in range(n_actions)])

        n_trials = 30000
        action_counter = Counter()
        for _ in range(n_trials):
            chosen_action = q_learner.choose_action()
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
        # inverse_temperature equals 0, which means uniformly random.
        n_actions = 3
        q_learner = reinforcement_learners.QSoftmax(0.1, 0, [100, -1, 40])

        n_trials = 30000
        action_counter = Counter()
        for _ in range(n_trials):
            chosen_action = q_learner.choose_action()
            action_counter[chosen_action] += 1

        exp_n_actions = n_trials / n_actions

        chi2_stats = [
            (obs_n_actions - exp_n_actions) ** 2 / exp_n_actions
            for obs_n_actions in action_counter.values()
        ]
        chi2_stats = sum(chi2_stats)

        # Pr(chi2(2) < 4.605) = 0.9
        assert chi2_stats < 4.605

    def test_nonuniform_choose_action2(self):
        initial_values = np.array([1, 2, 3])
        q_learner = reinforcement_learners.QSoftmax(0.1, 1.0, initial_values)

        n_trials = 30000
        action_counter = np.zeros(3)
        for _ in range(n_trials):
            chosen_action = q_learner.choose_action()
            action_counter[chosen_action] += 1

        exp_n_actions = n_trials * np.exp(initial_values) / np.exp(initial_values).sum()
        chi2_stats = ((exp_n_actions - action_counter) ** 2 / exp_n_actions).sum()

        # Pr(chi2(2) < 4.605) = 0.9
        assert chi2_stats < 4.605


class TestActionLeaner:
    # learning rate is more than 1.
    def test_excess_learning_rate(self):
        with pytest.raises(ValueError):
            reinforcement_learners.ActionLearner(1.1, 5, [0.5, 0.5])

    # learning rate is less than 0.
    def test_excess_learning_rate2(self):
        with pytest.raises(ValueError):
            reinforcement_learners.ActionLearner(-0.1, 5, [0.5, 0.5])

    def test_excess_num_choices(self):
        with pytest.raises(ValueError):
            reinforcement_learners.ActionLearner(0.4, 5,[0.3, 0.5, 0.2])

    def test_invalid_num_choices(self):
        with pytest.raises(ValueError):
            reinforcement_learners.ActionLearner(0.4, 5, [1.0])


    # check if learn works.
    def test_learn(self):
        agent = reinforcement_learners.ActionLearner(0.1, 5, [0.5, 0.5])
        agent.learn(0)
        assert agent.estimated_values[0] == pytest.approx(0.55)
        assert agent.estimated_values[1] == pytest.approx(0.45)

        agent.learn(1)
        assert agent.estimated_values[0] == pytest.approx(0.495)
        assert agent.estimated_values[1] == pytest.approx(0.505)

    def test_choose_action(self):
        initial_values = np.array([0.2, 0.8])
        inv_temp = 3
        agent = reinforcement_learners.ActionLearner(0.1, inv_temp, initial_values)

        n_trials = 30000
        action_counter = np.zeros(2)
        for _ in range(n_trials):
            chosen_action = agent.choose_action()
            action_counter[chosen_action] += 1
        expected_nums = softmax(initial_values * inv_temp) * n_trials
        chi2_stats = ((expected_nums - action_counter) ** 2 / expected_nums).sum()

        # Pr(chi2(2) < 4.605) = 0.9
        print(agent.estimated_values)
        assert chi2_stats < 4.605


class TestQActionSoftmax:
    # learning rate is more than 1.
    def test_excess_learning_rate_q(self):
        with pytest.raises(ValueError):
            reinforcement_learners.QActionSoftmax(1.1,
                                                  0.3,
                                                  5,
                                                  0.1,
                                                  [0.5, 0.5],
                                                  [0.5, 0.5]
                                                  )

    # learning rate is less than 0.
    def test_excess_learning_rate_q2(self):
        with pytest.raises(ValueError):
            reinforcement_learners.QActionSoftmax(-0.1,
                                                  0.3,
                                                  5,
                                                  0.1,
                                                  [0.5, 0.5],
                                                  [0.5, 0.5]
                                                  )
    def test_excess_learning_rate_a(self):
        with pytest.raises(ValueError):
            reinforcement_learners.QActionSoftmax(0.3,
                                                  1.1,
                                                  5,
                                                  0.1,
                                                  [0.5, 0.5],
                                                  [0.5, 0.5]
                                                  )

    # learning rate is less than 0.
    def test_excess_learning_rate_a2(self):
        with pytest.raises(ValueError):
            reinforcement_learners.QActionSoftmax(0.3,
                                                  -0.1,
                                                  5,
                                                  0.1,
                                                  [0.5, 0.5],
                                                  [0.5, 0.5]
                                                  )

    # check if learn works.
    def test_learn(self):
        agent = reinforcement_learners.QActionSoftmax(0.2,
                                              0.1,
                                              5,
                                              0.1,
                                              [0.5, 0.5],
                                              [0.5, 0.5]
                                              )
        agent.learn(0, 1)
        assert agent.q_values[0] == pytest.approx(0.60)
        assert agent.q_values[1] == pytest.approx(0.50)
        assert agent.a_values[0] == pytest.approx(0.55)
        assert agent.a_values[1] == pytest.approx(0.45)

        agent.learn(1, 1)
        assert agent.q_values[0] == pytest.approx(0.60)
        assert agent.q_values[1] == pytest.approx(0.60)
        assert agent.a_values[0] == pytest.approx(0.495)
        assert agent.a_values[1] == pytest.approx(0.505)

    def test_choose_action(self):
        initial_values_q = np.array([0.2, 0.8])
        initial_values_a = np.array([0.7, 0.3])
        inv_temp = 3
        weight = 0.2
        agent = reinforcement_learners.QActionSoftmax(0.2,
                                              0.1,
                                              inv_temp,
                                              weight,
                                              initial_values_q,
                                              initial_values_a
                                              )

        n_trials = 30000
        action_counter = np.zeros(2)
        for _ in range(n_trials):
            chosen_action = agent.choose_action()
            action_counter[chosen_action] += 1

        values = weight * initial_values_q + (1 - weight) * initial_values_a
        expected_nums = softmax(values * inv_temp) * n_trials
        chi2_stats = ((expected_nums - action_counter) ** 2 / expected_nums).sum()

        # Pr(chi2(2) < 4.605) = 0.9
        print(agent.estimated_values)
        assert chi2_stats < 4.605