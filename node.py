import numpy as np
import simpy
from scipy.stats import rv_continuous

from nano_parameters import packets_priorities, deltaS


# from matplotlib import pyplot as plt
class fixed_movement(rv_continuous):
    def __init__(self, speed):
        super().__init__()
        self.speed = speed

    def _rvs(self, *args):
        return self.speed


class VeinMobility():
    def __init__(self, descriptor_x, descriptor_y, bounds_x, bounds_y):
        self.desc_x = descriptor_x
        self.desc_y = descriptor_y
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y

    def get_next(self, x, y, dT):
        x += (self.desc_x.rvs() * dT)
        y += (self.desc_y.rvs() * dT)

        if self.bounds_x[2]:  # circular
            x = (x % self.bounds_x[1])
        else:
            x = np.clip(x, self.bounds_x[0], self.bounds_x[1])

        if self.bounds_y[2]:  # circular
            y = (y % self.bounds_y[1])
        else:
            y = np.clip(y, self.bounds_y[0], self.bounds_y[1])

        return x, y


class Node():
    def __init__(self, nodeId, x0, y0, mobility, dT, policy, harvesting_rates, gateways, env: simpy.Environment,
                 transmit_consumption, ram_consumption, energy_quantum_storable_max, distance_accountable_max,
                 gen_prob, packet_length, x_max, y_max, z_max, checker_gateway, debug=False):
        self.id = nodeId
        self.x = x0
        self.y = y0
        self.mobility = mobility
        self.dT = dT
        self.policy = policy
        self.harvesting_rates = harvesting_rates
        self.env = env
        self.prev_action = 0

        self.transmit_consumption = transmit_consumption
        self.ram_consumption = ram_consumption
        self.energy_quantum_storable_max = energy_quantum_storable_max
        self.distance_accountable_max = distance_accountable_max
        self.gen_prob = gen_prob
        self.packet_length = packet_length
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.checker = checker_gateway
        self.debug = debug

        self.battery = 0
        self.last_packet = 0
        self.distance_from_gateway = 0
        self.gateways = gateways
        self.in_range_gateway = None

    def run(self):
        while True:
            yield self.env.timeout(self.dT)

            action = self.policy.state_to_action(self.get_state(), self.battery, self.last_packet,
                                                 self.distance_from_gateway)
            energy_recharge = self.harvesting_rates[self.battery]

            # target van a ser 0/1. 0 es no hacer nada, 1 es hacer algo
            # hacer algo significa, cuando estoy bajo el gateway transmitir, y cuando no, storear

            if action == 2 and (self.distance_from_gateway > 0 or self.last_packet == 0):
                action = 0

            if (action == 2) and (self.transmit_consumption[self.last_packet] > self.battery):
                action = 0

            if (action == 1) and (self.ram_consumption > self.battery):
                action = 0

            if (action == 1) and (self.last_packet == 0):
                action = 0

            if action == 1 and self.prev_action != 1:
                pass    # es el primer store -> consumo flash write TODO: comprobar que haya suficiente batería

            if action == 0:
                self.last_packet = 0
            elif action == 2:
                # if not (self.distance_from_gateway == 0 and self.last_packet != 0):
                #     print('problemo')
                assert self.distance_from_gateway == 0 and \
                       self.last_packet != 0 and \
                       self.battery >= self.transmit_consumption[self.last_packet], \
                    "Node {} failed. Distance: {}. Last packet: {}. Battery: {}.".format(self.id,
                                                                                         self.distance_from_gateway,
                                                                                         self.last_packet, self.battery)

                energy_recharge -= self.transmit_consumption[self.last_packet]
                yield self.env.process(self.transmit())
                self.last_packet = 0

            elif action == 1:
                energy_recharge -= self.ram_consumption

            # update packet generation
            if action != 1:
                self.last_packet = self.draw_new_packet()

            # update battery charge
            self.battery = int(np.clip(self.battery + energy_recharge, 0, self.energy_quantum_storable_max))
            assert self.battery > 0, "Battery dropped below zero: " + str(self.battery)

            # update dist from last gateway
            self.x, self.y = self.mobility.get_next(self.x, self.y, self.dT)

            prev_idx = self.checker[int(np.floor(self.x / deltaS))]
            post_idx = self.checker[int(np.ceil(self.x / deltaS))]

            if prev_idx >= 0 or post_idx >= 0:
                prev_gw = self.gateways[prev_idx]
                post_gw = self.gateways[post_idx]
                prev_coverage = prev_gw.in_coverage(self.x, self.y)
                post_coverage = prev_gw.in_coverage(self.x, self.y)
                gw_coverage = prev_gw if prev_coverage else post_gw if post_coverage else None

                if gw_coverage is not None:
                    self.distance_from_gateway = 0
                    self.in_range_gateway = gw_coverage
                else:
                    self.in_range_gateway = None
                    self.distance_from_gateway = np.clip(self.distance_from_gateway + 1, 0,
                                                         self.distance_accountable_max)

            # # self.in_range_gateway = self.check_in_gateway_range()
            # if self.in_range_gateway:
            #     self.distance_from_gateway = 0
            #
            #     # prev = self.checker[int(np.floor(self.x / deltaS))]
            #     # post = self.checker[int(np.ceil(self.x / deltaS))]
            #     #
            #     # assert self.gateways[prev] == self.in_range_gateway or self.gateways[post] == self.in_range_gateway

            else:
                self.in_range_gateway = None
                self.distance_from_gateway = np.clip(self.distance_from_gateway + 1, 0, self.distance_accountable_max)

            self.prev_action = action

    def draw_new_packet(self):
        pkt = np.random.choice(np.arange(3), p=self.gen_prob)
        return pkt

    def get_state(self):
        return int(self.battery * self.y_max * self.z_max + self.last_packet * self.z_max + self.distance_from_gateway)

    def check_in_gateway_range(self):
        for g in self.gateways:
            if g.in_coverage(self.x, self.y):
                # g.in_coverage(self.x, self.y)
                return g

        return False

    def transmit(self):
        # dist = self.in_range_gateway.calc_dist(self.x, self.y)
        prop_delay = 0  # dist / 3e8
        yield self.env.timeout(prop_delay)
        yield self.env.process(
            self.in_range_gateway.receive(self, packets_priorities[self.last_packet],
                                          self.packet_length[self.last_packet]))

    def __repr__(self):
        return "Node id: {} positioned at ({}, {}) with batt: {}, pkt: {}, dist: {}".format(self.id, self.x, self.y,
                                                                                            self.battery,
                                                                                            self.last_packet,
                                                                                            self.distance_from_gateway)
