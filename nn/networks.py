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
from nn.error_funcs import d_se
from nn.layers import RectifierLayer, TanhLayer, LogisticLayer


class Network:
    _LAYER_CLASSES = {
        'logistic': LogisticLayer,
        'tanh': TanhLayer,
        'rectifier': RectifierLayer
    }

    _D_ERROR_FUNCS = {
        'se': d_se,
    }

    def __init__(self, name, n_input, error_func_name, epsilon):
        self._name = name
        self._n_input = n_input
        self._layers = []
        self._d_error_func = self._get_d_error_func(error_func_name)
        self._epsilon = epsilon

    @property
    def name(self):
        return self._name

    @property
    def layers(self):
        return self._layers

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    @property
    def error_func(self):
        for name, func in self._D_ERROR_FUNCS.items():
            if func == self._d_error_func:
                return name
        return None

    @error_func.setter
    def error_func(self, name):
        self._d_error_func = self._get_d_error_func(name)

    def _get_d_error_func(self, name):
        try:
            return self._D_ERROR_FUNCS[name]
        except KeyError:
            print('error_func_name must be %s or %s' %
                  (', '.join(self._D_ERROR_FUNCS.keys()[:-1]), list(self._D_ERROR_FUNCS.keys())[-1]))

    def add_layer(self, type, n_output, **kwargs):
        n_prev_output = self._layers[-1].n_output if self._layers else self._n_input
        try:
            layer = self._LAYER_CLASSES[type](n_output, n_prev_output, **kwargs)
        except KeyError:
            raise Exception('Layer type must be %s or %s.' %
                            (', '.join(self._LAYER_CLASSES.keys()[:-1]), list(self._LAYER_CLASSES.keys())[-1]))
        self._layers.append(layer)

    def propagate_forward(self, input_datum):
        output = input_datum
        for layer in self._layers:
            output = layer.propagate_forward(output)
        return output

    def propagate_backward(self, input_datum, teaching_datum):
        delta = self._d_error_func(teaching_datum, self.propagate_forward(input_datum))

        next_layer = None
        for layer in reversed(self._layers):
            delta = layer.propagate_backward(delta, next_layer.W if next_layer is not None else None)
            next_layer = layer

    def update(self, input_datum):
        prev_layer = None
        for layer in self._layers:
            layer.update(input_datum if prev_layer is None else prev_layer.y, self._epsilon)
            prev_layer = layer

    def to_json(self):
        def get_layer_name(layer):
            for key, cls in self._LAYER_CLASSES.items():
                if isinstance(layer, cls):
                    return key
            return Exception('Unknown layer type')

        return {'meta': {'name': self.name,
                         'n_input': self._n_input,
                         'error_func': self.error_func,
                         'epsilon': self.epsilon},
                'layers': [{'type': get_layer_name(layer), 'W': layer.W.tolist(), 'b': layer.b.tolist()}
                           for layer in self._layers]}

    @classmethod
    def from_json(cls, json_data):
        network = cls(json_data['meta']['name'], json_data['meta']['n_input'],
                      json_data['meta']['error_func'], json_data['meta']['epsilon'])
        for layer in json_data['layers']:
            network.add_layer(layer['type'], len(layer['W']))
            network.layers[-1].W = np.array(layer['W'])
            network.layers[-1].b = np.array(layer['b'])
        return network


class Classifier(Network):
    def get_class(self):
        return np.argmax(self._layers[-1].y)
        # return np.argmin(np.abs(1.0 - self._layers[-1].y))
