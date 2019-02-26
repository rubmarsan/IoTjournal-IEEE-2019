from policy_interface import PolicyIF


class NeutralStoringPolicy(PolicyIF):
    def __init__(self, mean_pos, delta_q, transmission_consumptions):
        assert delta_q <= 0
        self.mean_pos = mean_pos    # average distance between gateways
        self.delta_q = delta_q
        self.consumptions = transmission_consumptions

    def state_to_action(self, compound_state, x, y, z):
        if y > 0:
            if z == 0:
                return 2  # if random.random() > self.drop_prob else 0
            else:
                # assess most likely scenario
                required_energy = self.consumptions[y] - max(self.mean_pos - z, 0) * self.delta_q
                return x >= required_energy  # By the time I have a chance to transmit, I'll have enough energy
        else:
            return 0
