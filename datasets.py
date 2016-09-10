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


import numpy as np
from mnist import MNIST


class MnistDataset:
    def __init__(self, name, mnist_dir, val_collect, val_wrong):
        self._name = name
        self._mnist_dir = mnist_dir
        self._val_collect = val_collect
        self._val_wrong = val_wrong
        self._mnist = MNIST(self._mnist_dir)
        self._img_tensor, self._label_tensor = self._load_data()

    def __len__(self):
        return self._img_tensor.shape[0]

    @property
    def name(self):
        return self._name

    @property
    def img_size(self):
        return self._img_tensor.shape[1]

    @property
    def n_label_types(self):
        return self._label_tensor.shape[1]

    @property
    def imgs(self):
        return self._img_tensor

    @property
    def labels(self):
        return self._label_tensor

    def __iter__(self):
        for i in range(len(self._img_tensor)):
            yield self._img_tensor[i], self._label_tensor[i]
        raise StopIteration

    def _preprocess_imgs(self, imgs):
        # データの整形
        return np.array(imgs, dtype=float) / 255.0

    def _imgs_to_tensor(self, imgs):
        return np.array([img.reshape(len(img), 1) for img in imgs])

    def _labels_to_tensor(self, labels):
        def label_to_matrix(label):
            t = np.zeros(shape=(10, 1))
            t.fill(self._val_wrong)
            t[label, 0] = self._val_collect
            return t
        return np.array([label_to_matrix(label) for label in labels])

    def _load_mnist_data(self):
        return None, None

    def _load_data(self):
        imgs, labels = self._load_mnist_data()
        return self._imgs_to_tensor(self._preprocess_imgs(imgs)), self._labels_to_tensor(labels)


class MnistTrainingDataset(MnistDataset):
    def __init__(self, mnist_dir, val_collect, val_wrong):
        super(MnistTrainingDataset, self).__init__('training', mnist_dir, val_collect, val_wrong)

    def _load_mnist_data(self):
        return self._mnist.load_training()


class MnistTrainingDataset1000(MnistDataset):
    def __init__(self, mnist_dir, val_collect, val_wrong):
        super(MnistTrainingDataset1000, self).__init__('training1000', mnist_dir, val_collect, val_wrong)

    def _load_mnist_data(self):
        imgs, labels = self._mnist.load_training()
        return imgs[:1000], labels[:1000]


class MnistTrainingDataset5000(MnistDataset):
    def __init__(self, mnist_dir, val_collect, val_wrong):
        super(MnistTrainingDataset5000, self).__init__('training5000', mnist_dir, val_collect, val_wrong)

    def _load_mnist_data(self):
        imgs, labels = self._mnist.load_training()
        return imgs[:5000], labels[:5000]


class MnistTrainingDataset20000(MnistDataset):
    def __init__(self, mnist_dir, val_collect, val_wrong):
        super(MnistTrainingDataset20000, self).__init__('training10000', mnist_dir, val_collect, val_wrong)

    def _load_mnist_data(self):
        imgs, labels = self._mnist.load_training()
        return imgs[:20000], labels[:20000]


class MnistTestDataset(MnistDataset):
    def __init__(self, mnist_dir, val_collect, val_wrong):
        super(MnistTestDataset, self).__init__('test', mnist_dir, val_collect, val_wrong)

    def _load_mnist_data(self):
        return self._mnist.load_testing()


class MnistTestDataset1000(MnistDataset):
    def __init__(self, mnist_dir, val_collect, val_wrong):
        super(MnistTestDataset1000, self).__init__('test1000', mnist_dir, val_collect, val_wrong)

    def _load_mnist_data(self):
        return self._mnist.load_testing()


class MnistTestDataset5000(MnistDataset):
    def __init__(self, mnist_dir, val_collect, val_wrong):
        super(MnistTestDataset5000, self).__init__('test5000', mnist_dir, val_collect, val_wrong)

    def _load_mnist_data(self):
        return self._mnist.load_testing()


class MnistTestDataset20000(MnistDataset):
    def __init__(self, mnist_dir, val_collect, val_wrong):
        super(MnistTestDataset20000, self).__init__('test5000', mnist_dir, val_collect, val_wrong)

    def _load_mnist_data(self):
        return self._mnist.load_testing()
