import random


class RandomAgent:
    def __init__(self, output_dim):
        self.output_dim = output_dim

    def action(self, state):
        action = random.choice(range(self.output_dim))
        q_vals = [0]*self.output_dim
        q_vals[action] = 1
        return action, q_vals
