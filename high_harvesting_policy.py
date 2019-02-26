from policy_interface import PolicyIF
import random

class HighHarvestingPolicy(PolicyIF):
    def __init__(self, max_energy):
        self.max_energy = max_energy

    def state_to_action(self, compound_state, x, y, z):
        assert 0 <= x <= self.max_energy, "Invalid value for energy: " + str(x)

        if y > 0:
            if z == 0:
                return 2 # if random.random() > self.drop_prob else 0
            else:
                return 1 if random.random() <= x / self.max_energy else 0
        else:
            return 0