import numpy as np
from nano_parameters import inter_pulse_delay

# from matplotlib import pyplot as plt



class Gateway():
    def __init__(self, x0, y0, coverage, env, debug=False):
        self.x = x0
        self.y = y0
        self.coverage = coverage
        self.env = env
        self.packets = dict()
        self.reward = 0
        self.debug = debug

    def calc_dist(self, x, y):
        np.sqrt(np.power(self.x - x, 2) + np.power(self.y - y, 2))


    def receive(self, from_node, packet_priority, packet_length):
        # packet length in bits
        tx_delay = inter_pulse_delay * packet_length #

        start = self.env.now
        end = start + tx_delay
        band = 1
        node = from_node.id
        assert node not in self.packets, 'Already transmitting?'
        self.packets[node] = [start, end, band, False]  # append to the queue

        self.check_collision(node)
        yield self.env.timeout(tx_delay)

        if self.packets[node][3]:
            if self.debug:
                print('Colision nodo {}'.format(node))
        else:
            if self.debug:
                print('Transmitted ok. Start: {}, End: {}, Node: {} @ [{}, {}]'.format(start, end, node, from_node.x, from_node.y))
            self.reward += (packet_priority * packet_length)

        del self.packets[node]

    def in_coverage(self, x, y):
        # return self.calc_dist(x, y) <= self.coverage
        return np.sqrt(np.power(self.x - x, 2) + np.power(self.y - y, 2)) <= self.coverage
        # np.abs(self.x - x) <= self.coverage  # assuming plane wave

    def check_collision(self, node):
        assert node in self.packets
        start, end, band, col = self.packets[node]

        for node_alt, transmission_props in self.packets.items():
            start_alt, end_alt, band_alt, col = transmission_props
            if node == node_alt:
                continue

            if band == band_alt:    # si ambos estan en la cola ahora es que han colisionado
                self.packets[node_alt][3] = True
                self.packets[node][3] = True

        return
