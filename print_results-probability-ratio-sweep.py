import re

import numpy as np
from matplotlib import pyplot as plt


def run_mean(out_txt):
    means = []
    ratios = []
    r = re.compile(r'Reward [A-Z]{1,4} para pos = ([0-9.]+) -> mean: ([0-9.]+), std: ([0-9.]+)')

    for line in out_txt:
        match = r.findall(line)
        if match:
            ratio, mean, std = match[0]
            ratios.append(float(ratio))
            means.append(float(mean))

    return np.array(ratios), np.array(means)/100/30


def read_from_file(file_path):
    f = open(file_path, "r")
    res = f.readlines()
    f.close()
    return res


def study_mean(save=True):
    folder = './Probability-ratio-sweep/'

    out_MDP = read_from_file(folder + 'out_MDP.txt')
    ratio, means = run_mean(out_MDP)
    plt.plot(ratio, means, 'bo-', linewidth=3, ms=8)  # , stds_mdp)

    out_AS = read_from_file(folder + 'out_AS.txt')
    ratio, means = run_mean(out_AS)
    plt.plot(ratio, means, 'rd-', linewidth=3, ms=8)  # , stds_mdp)

    out_OSHP = read_from_file(folder + 'out_OSHP.txt')
    ratio, means = run_mean(out_OSHP)
    plt.plot(ratio, means, 'k*-', linewidth=3, ms=8)  # , stds_mdp)

    out_HHP = read_from_file(folder + 'out_HHP.txt')
    ratio, means = run_mean(out_HHP)
    plt.plot(ratio, means, 'y<-', linewidth=3, ms=8)  # , stds_mdp)

    out_CSP = read_from_file(folder + 'out_CSP.txt')
    ratio, means = run_mean(out_CSP)
    plt.plot(ratio, means, 'cP-', linewidth=3, ms=8)  # , stds_mdp)

    out_NSP = read_from_file(folder + 'out_NSP.txt')
    ratio, means = run_mean(out_NSP)
    plt.plot(ratio, means, 'ms-', linewidth=3, ms=8)  # , stds_mdp)

    plt.legend(["MDP", "ASP", "OSHP", "HHP", "CSP", "NSP"], ncol=2)
    plt.xlabel("Low-to-high-priority ratio (" + r"$\lambda_1 / \lambda_2$" + ")", fontsize=16)
    plt.ylabel("Prioritized throughput in bit/s", fontsize=16)
    plt.tick_params(labelsize=13)
    plt.tight_layout()
    # plt.xticks(ratio, ["{0:.0f}%".format(v * 100) for v in ratio])



    file_name = "./Article/Images/" + folder.split("/")[1] + "-throughput.png"
    if save:
        plt.savefig(file_name, dpi=300)

    plt.show()



if __name__ == '__main__':
    study_mean(True)