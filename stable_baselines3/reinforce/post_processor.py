import numpy as np


class PostProcessor:
    def __init__(
        self,
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_steps: int = 5,
        beta: float = 1.0,
    ):
        self.gae_lambda = gae_lambda
        self.n_steps = n_steps
        self.gamma = gamma
        self.beta = beta
