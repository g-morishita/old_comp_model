import os
import json
from collections import defaultdict

from agents import Agent
from bandit_tasks import Bandit


class SingleEnvironment:
    def __init__(
            self,
            time_horizon: int,
            agent: Agent,
            bandit_task: Bandit,
            path_history: str
            ) -> None:
        self.time_horizon = time_horizon
        self.agent = agent
        self.bandit_task = bandit_task
        self.history = defaultdict(list)
        self.path_history = path_history

    def run_simulation(self) -> None:
        for t in range(self.time_horizon):
            chosen_arm = self.agent.choose_action()
            reward = self.bandit_task.pull_arm(chosen_arm)
            self.history["reward"].append(int(reward)) # int is required to dump the dict to json.
            self.history["estimated_reward"].append(
                    self.agent.estimated_values.tolist())
            self.agent.learn(chosen_arm, reward)
        self.save_history()

    def save_history(self):
        with open(self.path_history, 'w') as f:
            json.dump(json.dumps(self.history), f)
