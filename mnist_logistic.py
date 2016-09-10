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


from matplotlib import pyplot as plt
from nn.networks import Classifier
from datasets import MnistTrainingDataset, MnistTestDataset, MnistTrainingDataset1000
from helpers import training, test, draw_W_histories, draw_mean_se_history, draw_cpr_history


if __name__ == '__main__':
    # Load MNIST dataset
    training_dataset = MnistTrainingDataset('./mnist', 1, -1)
    test_dataset = MnistTestDataset('./mnist', 1, 0)

    # Create Deep Neural Network for nmist classification
    classifier = Classifier('logistic', training_dataset.img_size, 'se', 0.15)
    classifier.add_layer('logistic', 200)
    classifier.add_layer('logistic', 10)

    # Training network and obtain histories
    W_histories, mean_se_history, cpr_history = training(classifier, training_dataset, 10)

    # Draw training histories
    draw_W_histories(W_histories, classifier.name, training_dataset.name)
    draw_mean_se_history(mean_se_history, classifier.name, training_dataset.name)
    draw_cpr_history(cpr_history, classifier.name, training_dataset.name)

    # Test network and print mean SE and CPR
    print(test(classifier, test_dataset))

    plt.show()
