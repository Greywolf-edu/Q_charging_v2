import csv
import pandas as pd
import random
from ast import literal_eval
from scipy import mean
from scipy.stats import sem, t

from Optimizer.Q_learning_Kmeans import Q_learningv2
from Simulator.Mobile_Charger.MobileCharger import MobileCharger
from Simulator.Network.Network import Network
from Simulator.Node.Node import Node

experiment_type = input('experiment_type: ')  # ['node', 'target', 'MC', 'prob', 'package', 'cluster']
df = pd.read_csv("data/" + experiment_type + ".csv")
experiment_index = int(input('experiment_index: '))  # [0..6]

# | Experiment_index      Experiment_type|    0    |    1    |    2    |    3     |    4   |
# |--------------------------------------|---------|---------|---------|----------|--------|
# | **node**                             |   700   |   800   | __900__ |   1000   |   1100 |
# | **target**                           |   500   |   550   | __600__ |   650    |   700  |
# | **MC**                               |   2     | __3__   |   4     |   5      |   6    |
# | **prob**                             |   0.5   | __0.6__ |   0.7   |   0.8    |   0.9  |
# | **package**                          | __500__ |   550   |   600   |   650    |   700  |
# | **cluster**                          |   40    |   50    |   60    |   70     | __80__ |

output_file = open("log/q_learning_Kmeans.csv", "w")
result = csv.DictWriter(output_file, fieldnames=["nb_run", "lifetime", "dead_node"])
result.writeheader()

com_ran = df.commRange[experiment_index]
prob = df.freq[experiment_index]
nb_mc = df.nb_mc[experiment_index]
alpha = df.q_alpha[experiment_index]
clusters = df.charge_pos[experiment_index]
package_size = df.package[experiment_index]
q_alpha = df.qt_alpha[experiment_index]
q_gamma = df.qt_gamma[experiment_index]


life_time = []
for nb_run in range(3):
    random.seed(nb_run)

    energy = df.energy[experiment_index]
    energy_max = df.energy[experiment_index]
    node_pos = list(literal_eval(df.node_pos[experiment_index]))

    list_node = []
    for i in range(len(node_pos)):
        location = node_pos[i]
        node = Node(location=location, com_ran=com_ran, energy=energy, energy_max=energy_max, id=i,
                    energy_thresh=0.4 * energy, prob=prob)
        list_node.append(node)
    mc_list = []
    for id in range(nb_mc):
        mc = MobileCharger(id, energy=df.E_mc[experiment_index], capacity=df.E_max[experiment_index],
                           e_move=df.e_move[experiment_index],
                           e_self_charge=df.e_mc[experiment_index], velocity=df.velocity[experiment_index])
        mc_list.append(mc)
    target = [int(item) for item in df.target[experiment_index].split(',')]
    net = Network(list_node=list_node, mc_list=mc_list, target=target, package_size=package_size)
    q_learning = Q_learningv2(nb_action=clusters, alpha=alpha, q_alpha=q_alpha, q_gamma=q_gamma)
    print("experiment {}, index:{}:\n".format(experiment_type, experiment_index))
    print("Network:\n\tNumber of sensors: {}, Number of targets: {}, Probability of package sending: {}, Package size: {}Bytes, Number of MCs: {}".format(len(net.node), len(net.target), prob, package_size, nb_mc))
    print("Optimizer Q_learning\n\tAlpha: {}, Gamma: {}, Theta: {}, Number of clusters: {}".format(q_learning.q_alpha, q_learning.q_gamma, q_learning.alpha, clusters))
    file_name = "log/q_learning_Kmeans_{}_{}_{}.csv".format(experiment_type, experiment_index, nb_run)
    temp = net.simulate(optimizer=q_learning, file_name=file_name)
    life_time.append(temp[0])
    result.writerow({"nb_run": nb_run, "lifetime": temp[0], "dead_node": temp[1]})

confidence = 0.95
h = sem(life_time) * t.ppf((1 + confidence) / 2, len(life_time) - 1)
result.writerow({"nb_run": mean(life_time), "lifetime": h, "dead_node": 0})
