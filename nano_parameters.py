import numpy as np
# from matplotlib import pyplot as plt

inter_pulse_delay = 10e-12
dr = 0.9  # discounted ratio
deltaS = 1e-3 # calculado para que un dS caiga siempre dentro de la zona de cobertura
vNodo = 100e-3  # m/s
deltaT = deltaS / vNodo # en segundos
deltaR = (0.15e-12/6) # en Jules
deltaQ = deltaR * deltaT
packet_length = np.array([0, 32, 64])
packets_priorities = np.array([0, 1, 5])
packets_priorities = packets_priorities / packets_priorities.sum()
packets_priorities_total = packets_priorities.shape[0] - 1  # low-priority and high-priority
packet_length_max = packet_length.max()  # bits
energy_per_bit = 0.0001e-12  # J/bit
energy_per_packet_max = packet_length_max * energy_per_bit
packets_sendable_max = 3  # up to 4 packets can be sent with battery completely full
energy_quantum_storable_max = round(energy_per_packet_max / deltaQ * packets_sendable_max)
gateway_distance_max = 0.04
gateway_distance_max_q = round(gateway_distance_max / deltaS)  # in deltaS
gateway_distance_min = 0.02
gateway_distance_min_q = round(gateway_distance_min / deltaS)  # in deltaS
distance_accountable_max = round(gateway_distance_max / deltaS)
num_actions = 3  # drop, store, transmit
gen_prob = np.array([0.97, 0.02, 0.01])  # 0.97 0.02 0.01 |||| [1 - 3/300, 2/300, 1/300] ||||

# PARA RAM
ram_consumption = 8  # in energy quantums per deltaT
flash_read_consumption = 0
flash_write_consumption = 0

# # PARA FLASH
# ram_consumption = 0
# flash_read_consumption = 4
# flash_write_consumption = 40


transmit_consumption = np.round(packet_length * energy_per_bit / deltaQ).astype(np.int) + flash_read_consumption
gateway_coverage = 1e-3 # en metros
gateway_coverage_markov = np.sqrt(np.power(gateway_coverage, 2) - np.power(0.8e-3, 2)) * 2 # asumiendo vena de 0.8mm # en metros el resultado
gateway_coverage_markov = int(round(gateway_coverage_markov/deltaS))
# FLASH -> 40 para store, 4 para read


# X dimension -> battery storage
# Y dimension -> packet generation
# Z dimension -> distance to last gateway

x_max = int(energy_quantum_storable_max + 1)
y_max = int(packets_priorities_total + 1)
z_max = int(distance_accountable_max + 1)
num_states = x_max * y_max * z_max

def getEnergyHarvestingRates():
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore')
        Vg = 0.2
        charge_wires = 6e-15
        Emax = 0.0064e-12 * 3
        C = Emax * 2 / (Vg ** 2)
        area = 500
        f = 1
        delta_Q = charge_wires * area

        energy = np.linspace(0, Emax, energy_quantum_storable_max + 1)
        rate = Vg * delta_Q * f * (1 - np.exp(np.log(1 - np.sqrt(2 * energy / (Vg ** 2 * C))))) * np.exp(np.log(1 - np.sqrt(2 * energy / (Vg ** 2 * C))))

        harvest_quantized = np.round(rate / deltaR)
        harvest_quantized[np.where(harvest_quantized == 0)] = 1  # force to recharge at least 1 quantum of energy

        # from matplotlib import pyplot as plt
        # fig, ax1 = plt.subplots()
        # cloned_x_axis = ax1.twiny()
        # cloned_x_axis.set_xticks([0, 9.6, 19.2])
        # cloned_x_axis.set_xticklabels(['0%', '50%', '100%'])
        # cloned_x_axis.xaxis.set_ticks_position('bottom')
        # cloned_x_axis.xaxis.set_label_position('bottom')
        # cloned_x_axis.spines['bottom'].set_position(('outward', 25))
        # cloned_x_axis.set_xlabel('Energy accumulated in the battery (fJ)', fontsize=13)
        #
        # # ax1.grid(True)
        # ax1.plot(energy * 1e15, rate * deltaT * 1e15, 'k',linewidth=4.0)
        # # ax1.set_xlabel('Energy accumulated in the battery (fJ)', fontsize=13)
        # ax1.set_ylabel('Energy harvested every T=0.01s (fJ)', color='k', fontsize=13)
        # # ax1.set_ylim([0, 1.6e-13])
        # ax1.tick_params('y', colors='k')
        # ax1.tick_params(axis='x', which='major', labelsize=14)
        # ax1.tick_params(axis='y', which='major', labelsize=14)
        #
        # cloned_x_axis.tick_params(axis='x', which='major', labelsize=14)
        # cloned_x_axis.tick_params(axis='y', which='major', labelsize=14)
        #
        # ax2 = ax1.twinx()
        # # ax2.grid(True)
        # # ax2.plot(range(int(energy_quantum_storable_max + 1)), harvest_quantized, 'r*',linewidth=4.0)
        # harvest_quantized[np.where(harvest_quantized == 6)] = 5.87
        # ax2.plot(energy * 1e15, harvest_quantized, 'b*', linewidth=4.0)
        # ax2.set_ylabel('Number of Q units harvested every T=0.01s', color='b', fontsize=13)
        # # ax2.set_ylim([-0.1, 6.15])
        # ax2.set_ylim([0, 6.15])
        # ax2.tick_params('y', colors='b')
        # ax2.tick_params(axis='x', which='major', labelsize=14)
        # ax2.tick_params(axis='y', which='major', labelsize=14)
        #
        #
        #
        # ax2.set_yticks([5.91, 4.97, 4.03, 3.01, 2, 1, 0][::-1])
        # ax2.set_yticklabels([str(v) for v in range(0, 7)])
        # cloned_x_axis.set_xlim(ax1.get_xlim())
        # fig.tight_layout()
        # plt.savefig('./Article/Images/energy_harvesting_rate.png', dpi=300)
        # plt.show()
        # exit(0)
        return harvest_quantized


# getEnergyHarvestingRates()

if __name__ == '__main__':
    getEnergyHarvestingRates()
