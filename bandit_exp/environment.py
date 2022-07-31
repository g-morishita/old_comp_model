import json
from collections import defaultdict

from bandit_exp.agents.agents import Agent
from bandit_tasks import Bandit


class SingleEnvironment:
    def __init__(
        self,
        time_horizon: int,
        agent: Agent,
        bandit_task: Bandit,
        path_history: str,
    ) -> None:
        self.time_horizon = time_horizon
        self.agent = agent
        self.bandit_task = bandit_task
        self.history = defaultdict(list)
        self.path_history = path_history

    def run_simulation(self) -> None:
        for _ in range(self.time_horizon):
            self.run_partial_simulation()

    def run_partial_simulation(self) -> None:
        chosen_arm = self.agent.choose_action()
        reward = self.bandit_task.pull_arm(chosen_arm)
        self.history["action"].append(int(chosen_arm))
        self.history["reward"].append(
            int(reward)
        )  # int is required to dump the dict to json.
        self.history["estimated_reward"].append(self.agent.estimated_values.tolist())
        self.agent.learn(chosen_arm, reward)

    def save_history(self):
        with open(self.path_history, "w") as f:
            json.dump(json.dumps(self.history), f)