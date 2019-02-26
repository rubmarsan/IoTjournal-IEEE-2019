import os
import pickle
import platform
import random
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import repeat

import mdptoolbox
import simpy
from scipy.stats import uniform

from always_store_policy import AlwaysStorePolicy
from conservative_storing_policy import ConservativeStoringPolicy
from gateway import Gateway
from high_harvesting_policy import HighHarvestingPolicy
from mdp_policy import MDP_policy
from neutral_storing_policy import NeutralStoringPolicy
from only_store_high_priority_policy import OnlyStoreHighPriorityPolicy
from random_policy import RandomPolicy
from tree_policy import treePolicy

""" backup
from always_store_policy import AlwaysStorePolicy
from only_store_high_priority_policy import OnlyStoreHighPriorityPolicy
from random_policy import RandomPolicy
from high_harvesting_policy import HighHarvestingPolicy
from conservative_storing_policy import ConservativeStoringPolicy
from neutral_storing_policy import NeutralStoringPolicy
"""

from nano_parameters import *
from node import Node, VeinMobility, fixed_movement
from os import path

if platform.node() == "alioth":
    print('Alioth config')
    WORKERS = 35
    NUM_NODES = 30
    RUNS = max(WORKERS, 200)
    print("{} WORKERS, {} NODOS, {} RUNS".format(WORKERS, NUM_NODES, RUNS))
else:
    print('Laptop config')
    WORKERS = 1  # 8
    NUM_NODES = 10  # 10
    RUNS = max(WORKERS, 1)  # 50)
    print("{} WORKERS, {} NODOS, {} RUNS".format(WORKERS, NUM_NODES, RUNS))


def compute_reward(params, policy_name):
    policy_array, num_nodes, seed = params

    random.seed(seed)
    np.random.seed(seed)
    sim_time = 100  # 10 secs // 1 hour
    num_updates = 2
    chunk = int(sim_time / num_updates)
    progress = 0
    debug = False
    vein_height = 0.8e-3  # in meters
    body_length = 15  # in meters
    constant_movement = fixed_movement(vNodo)
    vein_mobility = VeinMobility(
        constant_movement,  # x axis movement description
        uniform(-vNodo, vNodo * 2),  # y axis movement description
        (0, body_length, True),  # x axis bounds [inf lim, sup lim, circular?]
        (0, vein_height, False)  # y axis bounds [inf lim, sup lim, circular?]
    )

    min_initial_pos_x = 0
    max_initial_pos_x = 5e-3
    min_initial_pos_y = 0
    max_initial_pos_y = vein_height

    if policy_name == "MDP":
        policy = MDP_policy(policy_array)  # 2760.0
    elif policy_name == "AS":
        policy = AlwaysStorePolicy()  # 2146.181818181818
    elif policy_name == "OSHP":
        policy = OnlyStoreHighPriorityPolicy()
    elif policy_name == "RP":
        policy = RandomPolicy(drop_prob=0.5)
    elif policy_name == "HHP":
        policy = HighHarvestingPolicy(x_max)
    elif policy_name == "CSP":
        average_energy_harvesting = int(round(getEnergyHarvestingRates().mean()))
        average_energy_drop_while_storing = average_energy_harvesting - ram_consumption
        policy = ConservativeStoringPolicy(distance_accountable_max, average_energy_drop_while_storing,
                                           transmit_consumption)
    elif policy_name == "NSP":
        average_energy_harvesting = int(round(getEnergyHarvestingRates().mean()))
        average_energy_drop_while_storing = average_energy_harvesting - ram_consumption
        policy = NeutralStoringPolicy(int(round(distance_accountable_max / 2)), average_energy_drop_while_storing,
                                      transmit_consumption)
    elif policy_name == "TP":
        policy = treePolicy(policy_array, 80, x_max, y_max, z_max)
    else:
        raise Exception("Invalid policy name: " + str(policy_name))

    env = simpy.Environment()
    # policy = pickle.load(open('/home/ruben/mdp-ram.p', 'rb'))
    harvesting_rates = getEnergyHarvestingRates()

    gateways = list()
    x = 0
    while x < body_length:
        x += (gateway_distance_max - gateway_distance_min) * np.random.random() + gateway_distance_min
        g = Gateway(x, 0, gateway_coverage, env, debug)
        gateways.append(g)

    max_covered_dist = min(sim_time * vNodo + max_initial_pos_x, body_length)
    checker_gateway = [-1] * int(np.ceil(max_covered_dist / deltaS))

    x_q = 0
    x_i = 0
    while x_q < max_covered_dist:

        in_coverage = None
        for g_i, g in enumerate(gateways):
            if g.in_coverage(x_q, 0):
                in_coverage = g_i
                break
        if in_coverage is not None:
            checker_gateway[x_i] = g_i

        x_i += 1
        x_q += deltaS

    checker_gateway = [-1] + [max(checker_gateway[i - 1], checker_gateway[i], checker_gateway[i + 1]) for i in
                              range(1, len(checker_gateway) - 1)] + [-1]

    nodes = list()

    start = list()
    for n in range(num_nodes):
        x0 = (max_initial_pos_x - min_initial_pos_x) * np.random.random() + min_initial_pos_x
        y0 = (max_initial_pos_y - min_initial_pos_y) * np.random.random() + min_initial_pos_y
        t0 = np.random.random() * 0.1  # from 0 to 0.1 secs
        # t0 = (n + 1) * 1e-6
        start.append(t0)
        n = Node(n, x0, y0, vein_mobility, deltaT, policy, harvesting_rates, gateways, env, transmit_consumption,
                 ram_consumption, energy_quantum_storable_max, distance_accountable_max, gen_prob, packet_length, x_max,
                 y_max, z_max, checker_gateway, debug)
        nodes.append(n)

    for n_starting in np.argsort(start):
        next_starting_time = start[n_starting]
        env.run(until=next_starting_time)
        env.process(nodes[n_starting].run())

    for i in range(num_updates):
        progress += chunk
        env.run(until=progress)
        print("Progress: {} out of {}".format(progress, sim_time))

    tot_r = 0
    for g in gateways:
        tot_r += g.reward

    if debug:
        print('Total reward: {}\nMean reward: {}'.format(tot_r, tot_r / num_nodes))

    return tot_r


# def get_state(battery, packet, distance):
#     return int(battery * y_max * z_max + packet * z_max + distance)

def simulate_MDP():

    policy = pickle.load(open("./Policies 097 002 001 mean/33.0-17.0-infinite.p", "rb"))
    # policy = np.zeros((len(pi.policy), 3))
    # policy[np.arange(len(pi.policy)), np.array(pi.policy)] = 1
    # policy = pi.policy
    policy_unravelled = np.array(policy).reshape((x_max, y_max, z_max))

    uniform_cdf = [uniform.cdf(x, gateway_distance_min_q, max(gateway_distance_max_q - gateway_distance_min_q, 0.001))
                   for x in range(z_max)]
    harvesting_rates = getEnergyHarvestingRates()

    # 8333.3333
    battery = 0
    distance = 0
    packet = 0

    tot_rewards = []
    max_time = int(100 / deltaT)
    for _ in range(100):
        reward = 0

        for it in range(max_time):
            action = policy_unravelled[battery, packet, distance] # pi.policy[get_state(battery, packet, distance)]

            if np.random.random() <= uniform_cdf[distance]:
                distance = 0
            else:
                distance = (distance + 1) % z_max

            if action == 0:
                battery = int(np.clip(battery + harvesting_rates[battery], 0, energy_quantum_storable_max))
            elif action == 1:
                battery = int(np.clip(battery + harvesting_rates[battery] - ram_consumption, 0, energy_quantum_storable_max))
            else:
                battery = int(np.clip(battery + harvesting_rates[battery] - transmit_consumption[packet], 0, energy_quantum_storable_max))

            reward += (packet_length[packet] * packets_priorities[packet]) * (action == 2)

            if action != 1:
                packet = np.random.choice(3, p=gen_prob)

        tot_rewards.append(reward)

    print(np.mean(tot_rewards))
    exit(-1)
    print("ok")


def sweep_values():
    global gateway_distance_max, gateway_distance_min, gateway_distance_max_q, gateway_distance_min_q, distance_accountable_max, z_max, num_states, gen_prob

    rewards = list()

    policy_name = "MDP"  # MDP, AS, OSHP, RP, HHP, CSP, NSP, TP
    print("Running policy", policy_name)
    steps = 1
    # simulate_MDP()

    probs_lambda_0 = np.round(np.linspace(0.99, 0.8, 11), 3)
    gateway_distance_min = 0.05
    gateway_distance_max = 0.05

    gateway_distance_max_q = round(gateway_distance_max / deltaS)  # in deltaS
    gateway_distance_min_q = round(gateway_distance_min / deltaS)  # in deltaS
    distance_accountable_max = round(gateway_distance_max / deltaS)
    z_max = int(distance_accountable_max + 1)
    num_states = x_max * y_max * z_max

    with open("out_{}.txt".format(policy_name), "a+") as file:
        for i, prob_lambda_0 in enumerate(probs_lambda_0):
            # gen_prob = np.array([0.97, 0.02, 0.01])
            gen_prob[0] = prob_lambda_0
            gen_prob[2] = (1 - prob_lambda_0) / 3
            gen_prob[1] = 2 * gen_prob[2]

            assert abs(gen_prob.sum() - 1) < 1e-4

            assert (gateway_distance_max - gateway_distance_min) > -1e-5
            if gateway_distance_max < gateway_distance_min:
                gateway_distance_max = gateway_distance_min
                print('Correcting small distances')

            infinite = True
            if infinite:
                path_file = "{}-infinite.p".format(prob_lambda_0)
            else:
                path_file = "{}.p".format(prob_lambda_0)

            if path.isfile(path_file):
                warnings.warn('Cargando la politica desde el HDD')
                policy = pickle.load(open(path_file, 'rb'))
            else:
                policy = re_compute_policy(path_file, infinite)

            seeds = np.arange(i * RUNS, (i + 1) * RUNS)

            policy_unravelled = np.array(policy).reshape((x_max, y_max, z_max))
            # exit(0)
            continue

            compute_reward_fixed = partial(compute_reward, policy_name=policy_name)

            if WORKERS > 1:
                with ProcessPoolExecutor(max_workers=WORKERS) as executor:
                    p = executor.map(compute_reward_fixed, zip(repeat(policy), repeat(NUM_NODES), seeds))
                    reward_list = np.array(list(p))
            else:
                reward_list = list()
                for j in range(RUNS):
                    reward_list.append(compute_reward_fixed([policy, NUM_NODES, seeds[j]]))

            mean_r = np.mean(reward_list)
            std_r = np.std(reward_list)
            rewards.append((mean_r, std_r))

            print('Reward {} para pos = {} -> mean: {}, std: {}'.format(policy_name, prob_lambda_0, mean_r, std_r))
            file.write('Reward {} para pos = {} -> mean: {}, std: {}\n'.format(policy_name, prob_lambda_0, mean_r, std_r))
            exit()
            file.flush()
            os.fsync(file.fileno())
            # print(reward_list)

    pickle.dump(rewards, open('rewards-{}.p'.format(policy_name), 'wb'))


def re_compute_policy(path_file, infinite=False):
    uniform_cdf = [uniform.cdf(x, gateway_distance_min_q - 1, max(gateway_distance_max_q -
                                                                  gateway_distance_min_q, 0.001)) for x in range(z_max)]

    assert gen_prob.sum() - 1 < 1e-10
    print('Computing policy for Lambda = {}'.format(gen_prob))

    T = np.zeros((num_actions, num_states, num_states))  # A x S x S'
    R = np.zeros((num_states, num_actions))  # S x A

    harvesting_rates = getEnergyHarvestingRates()

    for x in range(x_max):
        for y in range(y_max):
            for z in range(z_max):
                state = x * y_max * z_max + y * z_max + z
                harvest_rate = harvesting_rates[x]

                # x_c = state // (y_max * z_max)
                # y_c = (state - (x_c * y_max * z_max)) // z_max
                # z_c = state - (x_c * y_max * z_max) - (y_c * z_max)
                #
                # assert x_c == x and y_c == y and z_c == z

                # action drop it
                new_energy_value = min(x + harvest_rate, energy_quantum_storable_max)
                for y_alt in range(3):
                    z_plus_one = min(z + 1, distance_accountable_max)
                    z_gateway = 0

                    state_p_no_gateway = int(new_energy_value * y_max * z_max + y_alt * z_max + z_plus_one)
                    state_p_yes_gateway = int(new_energy_value * y_max * z_max + y_alt * z_max + z_gateway)

                    T[0, state, state_p_no_gateway] = (1 - uniform_cdf[z]) * gen_prob[y_alt]
                    T[0, state, state_p_yes_gateway] = (uniform_cdf[z]) * gen_prob[y_alt]

                # action keep it
                if y != 0 and x >= ram_consumption:  # packet stored in buffer, can still keep it
                    # don't move in the y-axis since we are keeping the packet
                    new_energy_value = max(min(x + harvest_rate - ram_consumption, energy_quantum_storable_max), 0)

                    z_plus_one = min(z + 1, distance_accountable_max)
                    z_gateway = 0

                    state_p_no_gateway = int(new_energy_value * y_max * z_max + y * z_max + z_plus_one)
                    state_p_yes_gateway = int(new_energy_value * y_max * z_max + y * z_max + z_gateway)

                    T[1, state, state_p_no_gateway] = (1 - uniform_cdf[z]) * 1
                    T[1, state, state_p_yes_gateway] = (uniform_cdf[z]) * 1
                else:  #
                    # dont have enough energy to keep the packet in the buffer, drop it and maybe, generate another
                    # however, we model the ram_consumption to penalize trying to keep it
                    new_energy_value = max(min(x + harvest_rate - ram_consumption, energy_quantum_storable_max), 0)

                    for y_alt in range(3):
                        z_plus_one = min(z + 1, distance_accountable_max)
                        z_gateway = 0

                        state_p_no_gateway = int(new_energy_value * y_max * z_max + y_alt * z_max + z_plus_one)
                        state_p_yes_gateway = int(new_energy_value * y_max * z_max + y_alt * z_max + z_gateway)

                        T[1, state, state_p_no_gateway] = (1 - uniform_cdf[z]) * gen_prob[y_alt]
                        T[1, state, state_p_yes_gateway] = (uniform_cdf[z]) * gen_prob[y_alt]

                # action transmit it
                # if y != 0 and z == 0 and x >= transmit_consumption[y]:
                if y != 0 and z < gateway_coverage_markov and x >= transmit_consumption[y]:
                    # packet stored in buffer, i am in a gateway coverage, and have enough energy to transmit it:
                    # let's transmit it!
                    new_energy_value = max(min(x + harvest_rate - transmit_consumption[y], energy_quantum_storable_max),
                                           0)
                    for y_alt in range(3):
                        z_plus_one = min(z + 1, distance_accountable_max)
                        z_gateway = 0

                        state_p_no_gateway = int(new_energy_value * y_max * z_max + y_alt * z_max + z_plus_one)
                        state_p_yes_gateway = int(new_energy_value * y_max * z_max + y_alt * z_max + z_gateway)

                        T[2, state, state_p_no_gateway] = (1 - uniform_cdf[z]) * gen_prob[y_alt]
                        T[2, state, state_p_yes_gateway] = (uniform_cdf[z]) * gen_prob[y_alt]
                else:  # trying to transmit but unable to do so
                    # however, model the consumption derived from the transmision to penalize it
                    new_energy_value = max(min(x + harvest_rate - transmit_consumption[y], energy_quantum_storable_max),
                                           0)

                    for y_alt in range(3):
                        z_plus_one = min(z + 1, distance_accountable_max)
                        z_gateway = 0

                        state_p_no_gateway = int(new_energy_value * y_max * z_max + y_alt * z_max + z_plus_one)
                        state_p_yes_gateway = int(new_energy_value * y_max * z_max + y_alt * z_max + z_gateway)

                        T[2, state, state_p_no_gateway] = (1 - uniform_cdf[z]) * gen_prob[y_alt]
                        T[2, state, state_p_yes_gateway] = (uniform_cdf[z]) * gen_prob[y_alt]

    assert np.allclose(T.sum(axis=2), 1, rtol=1e-7)
    assert len(np.where(T < 0)[0]) == 0, 'This should not happen'

    for x in range(x_max):
        for y in range(1, y_max):  # if y == 0 -> R = 0, regardless of the action
            z = 0  # can only transmit if we are within a gateway range

            if x < transmit_consumption[y]:  # can only transmit if enough charge is available
                continue

            state = x * y_max * z_max + y * z_max + z

            reward = packet_length[y] * packets_priorities[y]
            R[state, 2] = reward


    if infinite:
        pi = mdptoolbox.mdp.RelativeValueIteration(T, R, epsilon=0.005, max_iter=5000)
        pi.run()
        pickle.dump(pi.policy, open(path_file, 'wb'))
        print('finished computing infinite Pi')
    else:
        pi = mdptoolbox.mdp.ValueIteration(T, R, dr)
        pi.run()
        pickle.dump(pi.policy, open(path_file, 'wb'))
        print('finished computing discounted Pi')

    return pi.policy


if __name__ == '__main__':
    sweep_values()
