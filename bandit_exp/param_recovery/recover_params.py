from ..models.base import Model
from .generate_bandit_data import Generator


class RecoverParams:
    def __init__(self, model, generator):
        if not isinstance(model, Model):
            raise ValueError(f'model should be Model class. '
                             f'{model.__class__.__name__} is given.')

        if not isinstance(generator, Generator):
            raise ValueError(f'generator should be Generator class. '
                             f'{generator.__class__.__name__} is given.')

        if not generator.done_simulation:
            raise Exception("With given Generator object, simulation has not been done.")

        self.model = model
        self.generator = generator

    def fit(self, **kwargs):
        num_choices = len(self.generator.bandit_task.arms)
        choices = self.generator.history["choices"]
        rewards = self.generator.history["rewards"]

        self.model.fit(num_choices, choices, rewards, **kwargs)
