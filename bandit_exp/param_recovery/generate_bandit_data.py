from ..agents.base import Agent
from ..bandit_tasks import Bandit
import copy


class Generator:
    def __init__(self, agent: Agent, bandit_task: Bandit) -> None:
        """A model to generate "fake" behavioral data using an agent.

        Parameters
        ----------
        agent : Agent to make choices.
        original_agent : Agent that has initial params and is used for reset method.
        bandit_task : Bandit task that has information like the reward probabilities.
        done_simulation : indicator of if the simulation is done or not
        total_trials : the total number of trials
        """
        if not isinstance(agent, Agent):
            raise ValueError(
                f"agent should be Agent class "
                f'from "agents" package. {agent.__class__.__name__} is given.'
            )
        if not isinstance(bandit_task, Bandit):
            raise ValueError(
                f"bandit_task should be Bandit class {bandit_task.__class__.__name__} is given."
            )

        self.agent = agent
        self.original_agent = copy.deepcopy(agent)
        self.bandit_task = bandit_task
        self.done_simulation = False
        self.history = {"choices": [], "rewards": []}
        self.total_trials = 0

    def simulate(self, n_trials: int) -> None:
        """Generate fake behavorial data."""
        if self.done_simulation:
            print(
                "This agent has been already used to generate the data before."
                "Which means the parameters of the agents have changed."
                "If you do not mean to use the learnt agent, use reset method."
            )
        for _ in range(n_trials):
            choice = self.agent.choose_action()
            self.history["choices"].append(choice)
            reward = self.bandit_task.pull_arm(choice)
            self.history["rewards"].append(reward)

            self.agent.learn(choice, reward)

        self.total_trials += n_trials
        self.done_simulation = True

    def reset(self):
        """Reset agent and history."""
        self.agent = copy.deepcopy(self.original_agent)
        self.history = {"choices": [], "rewards": []}
        self.done_simulation = False
        self.total_trials = 0
