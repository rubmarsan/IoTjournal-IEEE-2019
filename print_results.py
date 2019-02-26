import re

import numpy as np
from matplotlib import pyplot as plt


def run(out_txt):
    means = []
    # stds = []
    deltas = []
    stds_uniform = []
    r = re.compile(r'Reward [A-Z]{1,4} para pos = \(([0-9.]+), ([0-9.]+)\) -> mean: ([0-9.]+), std: ([0-9.]+)')

    for line in out_txt:
        match = r.findall(line)
        if match:
            pos_a, pos_b, mean, std = match[0]
            var_uniform = (1 / 12) * (float(pos_b) - float(pos_a))
            stds_uniform.append(np.sqrt(var_uniform))
            means.append(float(mean))
            deltas.append(0.05 - float(pos_a))
            # stds.append(float(std))

    # return np.array(stds_uniform), np.array(means), np.array(stds)
    return np.array(stds_uniform), np.array(means)/100/30, np.array(deltas)


def run_mean(out_txt):
    means = []
    mean_position = []
    r = re.compile(r'Reward [A-Z]{1,4} para pos = \(([0-9.]+), ([0-9.]+)\) -> mean: ([0-9.]+), std: ([0-9.]+)')

    for line in out_txt:
        match = r.findall(line)
        if match:
            pos_a, pos_b, mean, std = match[0]
            mean_position.append((float(pos_b) + float(pos_a))/2)
            means.append(float(mean))

    return np.array(mean_position), np.array(means) / 100 / 30




def read_from_file(file_path):
    f = open(file_path, "r")
    return f.readlines()



def study_mean(save=True):
    folder = './low_prob_mean/'

    out_MDP = read_from_file(folder + 'out_MDP.txt')
    positions, means = run_mean(out_MDP)
    plt.plot(positions, means, 'bo-', linewidth=3, ms=8)  # , stds_mdp)

    out_AS = read_from_file(folder + 'out_AS.txt')
    positions, means = run_mean(out_AS)
    plt.plot(positions, means, 'rd-', linewidth=3, ms=8)  # , stds_mdp)

    out_OSHP = read_from_file(folder + 'out_OSHP.txt')
    positions, means = run_mean(out_OSHP)
    plt.plot(positions, means, 'k*-', linewidth=3, ms=8)  # , stds_mdp)

    out_HHP = read_from_file(folder + 'out_HHP.txt')
    positions, means = run_mean(out_HHP)
    plt.plot(positions, means, 'y<-', linewidth=3, ms=8)  # , stds_mdp)

    out_CSP = read_from_file(folder + 'out_CSP.txt')
    positions, means = run_mean(out_CSP)
    plt.plot(positions, means, 'cP-', linewidth=3, ms=8)  # , stds_mdp)

    out_NSP = read_from_file(folder + 'out_NSP.txt')
    positions, means = run_mean(out_NSP)
    plt.plot(positions, means, 'ms-', linewidth=3, ms=8)  # , stds_mdp)

    plt.legend(["MDP", "ASP", "OSHP", "HHP", "CSP", "NSP"], ncol=2)
    plt.xlabel("Average distance between nanorouters in cm", fontsize=16)
    plt.ylabel("Prioritized throughput in bit/s", fontsize=16)
    plt.tick_params(labelsize=13)
    plt.tight_layout()
    plt.xticks(positions, ["{0:.0f}".format(v * 100) for v in positions])


    file_name = "./Article/Images/" + folder.split("/")[1] + "-throughput.png"
    if save:
        plt.savefig(file_name, dpi=300)

    plt.show()

def study_delta(save=True):
    folder = './low_prob/'
    out_MDP = read_from_file(folder + 'out_MDP.txt')
    stds_uniform_mdp, means_mdp, stds_mdp = run(out_MDP)
    plt.plot(stds_mdp, means_mdp, 'bo-', linewidth=3, ms=8)  # , stds_mdp)

    out_AS = read_from_file(folder + 'out_AS.txt')
    stds_uniform_as, means_as, stds_as = run(out_AS)
    plt.plot(stds_as, means_as, 'rd-', linewidth=3, ms=8)  # , stds_as)

    out_OSHP = read_from_file(folder + 'out_OSHP.txt')
    stds_uniform_oshp, means_oshp, stds_oshp = run(out_OSHP)
    plt.plot(stds_oshp, means_oshp, 'k*-', linewidth=3, ms=8)  # , stds_as)

    # out_RP = read_from_file(folder + 'out_RP.txt')
    # stds_uniform_rp, means_rp, stds_rp = run(out_RP)
    # plt.plot(stds_rp, means_rp, 'g*-', linewidth=3, ms=8)  # , stds_as)

    out_HHP = read_from_file(folder + 'out_HHP.txt')
    stds_uniform_hhp, means_hhp, stds_hhp = run(out_HHP)
    plt.plot(stds_hhp, means_hhp, 'y<-', linewidth=3, ms=8)  # , stds_as)

    out_CSP = read_from_file(folder + 'out_CSP.txt')
    stds_uniform_csp, means_csp, stds_csp = run(out_CSP)
    plt.plot(stds_csp, means_csp, 'cP-', linewidth=3, ms=8)  # , stds_as)

    out_NSP = read_from_file(folder + 'out_NSP.txt')
    stds_uniform_nsp, means_nsp, stds_nsp = run(out_NSP)
    plt.plot(stds_nsp, means_nsp, 'ms-', linewidth=3, ms=8)  # , stds_as)

    # plt.legend(["MDP", "ASP", "OSHP", "RP", "HHP", "CSP", "NSP"], ncol=2)
    plt.legend(["MDP", "ASP", "OSHP", "HHP", "CSP", "NSP"], ncol=2)
    plt.xlabel("Imprecision in placing the nanorouters in cm (" + r'$\delta$' + ")", fontsize=16)
    plt.ylabel("Prioritized throughput in bit/s", fontsize=16)
    plt.tick_params(labelsize=13)
    plt.tight_layout()
    plt.xticks(np.linspace(0.0018, 0.03, 6, endpoint=True), ["{0:.2f}".format(v*100) for v in np.linspace(0, 0.03, 6, endpoint=True)])

    file_name = "./Article/Images/" + folder.split("/")[1] + "-throughput.png"
    if save:
        plt.savefig(file_name, dpi=300)

    plt.show()


if __name__ == '__main__':
    study_delta(True)
    study_mean(True)