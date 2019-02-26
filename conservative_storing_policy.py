from policy_interface import PolicyIF


class ConservativeStoringPolicy(PolicyIF):
    def __init__(self, max_pos, delta_q, transmission_consumptions):
        assert delta_q <= 0
        self.max_pos = max_pos
        self.delta_q = delta_q  # average_energy_drop_while_storing = average_harvesting - ram_consumption
        self.consumptions = transmission_consumptions

    def state_to_action(self, compound_state, x, y, z):
        assert 0 <= z <= self.max_pos, "Invalid value for position: " + str(z)

        if y > 0:
            if z == 0:
                return 2  # if random.random() > self.drop_prob else 0
            else:
                required_energy = self.consumptions[y] - (self.max_pos - z) * self.delta_q  # assess worst-case scenario
                return x >= required_energy  # By the time I have a chance to transmit, I'll have enough energy
        else:
            return 0
