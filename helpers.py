# This file is part of "MPS Yokohama Deep Learning Series Day 09/10/2016"
#
# "MPS Yokohama Deep Learning Series Day 09/10/2016"
# is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# "MPS Yokohama Deep Learning Series Day 09/10/2016"
# is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#
# (c) Junya Kaneko <jyuneko@hotmail.com>


import os
import json
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from nn.error_funcs import se


# Draw W_histories
def draw_W_histories(W_histories, network_name, dataset_name):
    for i, W_history in enumerate(W_histories):
        plt.figure()
        plt.title('W%s history' % i)
        plt.plot(range(len(W_history)), W_history)
        plt.savefig('img_mnist_%s/W%s_history_%s.png' % (network_name, i, dataset_name))


# Draw SE history and its moving average
def draw_mean_se_history(mean_se_history, network_name, dataset_name):
    plt.figure()
    plt.title('Mean SE History')
    plt.plot(range(len(mean_se_history)), mean_se_history, color='green')
    plt.savefig('img_mnist_%s/mean_se_history_%s.png' % (network_name, dataset_name))


# Draw CPR history
def draw_cpr_history(cpr_history, network_name, dataset_name):
    plt.figure()
    plt.title('CPR')
    plt.plot(range(len(cpr_history)), cpr_history)
    plt.savefig('img_mnist_%s/cpr_%s.png' % (network_name, dataset_name))


# Calculate mean SE and correct prediction rate
def calc_mean_se_and_cpr(network, dataset):
    mean_se = 0.0
    n_correct_predictions = 0
    for x, t in dataset:
        y = network.propagate_forward(x)
        mean_se += se(t, y)
        if np.argmax(t) == network.get_class():
            n_correct_predictions += 1
    return mean_se / len(dataset), n_correct_predictions / len(dataset)


# Train network
def training(network, dataset, n_round):
    W_histories = [[] for _ in network.layers]
    mean_se_history = []
    cpr_history = []
    for _ in tqdm(range(n_round)):
        for x, t in dataset:
            network.propagate_backward(x, t)
            network.update(x)

        # Store W, mean SE and CRP histories
        for i in range(len(W_histories)):
            W_histories[i].append(network.layers[i].ave_W)
        mean_se, cpr = calc_mean_se_and_cpr(network, dataset)
        mean_se_history.append(mean_se)
        cpr_history.append(cpr)
    return W_histories, mean_se_history, cpr_history

    # draw_W_histories(W_histories, network.name, dataset.name)
    # draw_mean_se_history(mean_se_history, network.name, dataset.name)
    # draw_cpr_history(cpr_history, network.name, dataset.name)


# Test network
def test(network, dataset):
    return calc_mean_se_and_cpr(network, dataset)


# Save network
def save_network(network, base_dir):
    with open('%s.json' % os.path.join(base_dir, network.name), 'w') as f:
        json.dump(network.to_json(), f)

