import market_model
import numpy as np


class Simulation:
    def __init__(self, steps: int = 200, no_agents: int = 500, sigma: np.double = 0.1, mu: np.double = 0.0,
                 J: np.double = -0.03, alpha: np.double = 0.0, beta: np.double = 0.0, g: np.double = 0.0):
        self.steps = steps
        self.n = no_agents
        self.sig = sigma
        self.mu = mu
        self.J = J
        self.alpha = alpha
        self.beta = beta
        self.agents = np.full(self.n, 1.0, dtype=np.double)
        self.N: int = 0
        self.g = g

    def _lognormal_multiplication(self):
        self.agents = np.multiply(self.agents, np.exp(np.random.normal(self.mu, self.sig, self.n)))

    def run_standard(self):
        for i in range(self.steps):
            self._lognormal_multiplication()
            self.agents = market_model.standard_simulation_step(self.agents, self.J, self.alpha, self.beta)
        self.N += self.steps

    def run_nonlinear(self):
        for i in range(self.steps):
            self._lognormal_multiplication()
            self.agents = market_model.non_linear_transfer_simulation_step(self.agents, self.J, self.alpha, self.beta,
                                                                           self.g)
        self.N += self.steps

    def get_simulation(self) -> (np.array(np.dtype), int):
        return self.agents, self.N
