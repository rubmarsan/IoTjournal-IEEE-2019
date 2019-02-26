from policy_interface import PolicyIF
import random

class RandomPolicy(PolicyIF):
    def __init__(self, drop_prob = 0.5):
        self.drop_prob = drop_prob

    def state_to_action(self, compound_state, x, y, z):
        if y > 0:
            if z == 0:
                return 2 # if random.random() > self.drop_prob else 0
            else:
                return 1 if random.random() > self.drop_prob else 0
        else:
            return 0