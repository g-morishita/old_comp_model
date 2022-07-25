from environment import SingleEnvironment
from agents import QSoftmax
from bandit_tasks import BernoulliMultiArmedBandit


if __name__ == "__main__":
    num_arms = 2
    means = [0.2, 0.7]

    learning_rate = 0.2
    inverse_temperature = 5.0
    initial_values = [0, 0]

    time_horizon = 300
    hist_path = "./history.json"

    q_learner = QSoftmax(learning_rate, inverse_temperature, initial_values)

    two_armed_bandit = BernoulliMultiArmedBandit(means)

    env = SingleEnvironment(time_horizon, q_learner, two_armed_bandit, hist_path)
    env.run_simulation()
