from policy_interface import PolicyIF

class MDP_policy(PolicyIF):
    def __init__(self, policy):
        self.policy = policy

    def state_to_action(self, compound_state, x, y, z):
        return self.policy[compound_state]
