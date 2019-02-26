from policy_interface import PolicyIF


class AlwaysStorePolicy(PolicyIF):
    def state_to_action(self, compound_state, x, y, z):
        if y > 0:
            if z == 0:
                return 2
            else:
                return 1
        else:
            return 0