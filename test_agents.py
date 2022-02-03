import pytest
import agents


def test_not_implemented_error():
    class AgentWithoutLearn(agents.Agent):
        def choose_action(self):
            pass

    with pytest.raises(NotImplementedError):
        test_agent = AgentWithoutLearn()
        test_agent.learn()

    class AgentWithoutChooseAction(agents.Agent):
        def learn(self):
            pass

    with pytest.raises(NotImplementedError):
        test_agent = AgentWithoutChooseAction()
        test_agent.choose_action()


def test_Q_learner_softmax():
    pass
