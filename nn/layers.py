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
from nn.activation_funcs import logistic, d_logistic, tanh, d_tanh, rectifier, d_rectifier


class BaseLayer:
    def __init__(self, n_output, n_prev_output, f, df):
        self._W = self._init_W(n_output, n_prev_output)
        self._b = self._init_b(n_output)
        self._f = f
        self._df = df
        self._y = None
        self._delta = None

    def _init_W(self, n_output, n_prev_output, *args, **kwargs):
        return np.random.randn(n_output, n_prev_output)

    def _init_b(self, n_output, *args, **kwargs):
        return np.random.randn(n_output, 1)

    @property
    def n_output(self):
        return self._W.shape[0]

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, value):
        self._W = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b = value

    @property
    def ave_W(self):
        return np.average(np.abs(self._W))

    @property
    def y(self):
        return self._y

    @property
    def delta(self):
        return self._delta

    def propagate_forward(self, x):
        self._y = self._f(self._W @ x + self._b)
        return self._y

    def propagate_backward(self, next_delta, next_W):
        if next_W is not None:
            self._delta = next_W.T @ next_delta * self._df(self._y)
        else:
            self._delta = next_delta * self._df(self._y)
        return self._delta

    def update(self, prev_y, epsilon):
        Delta_W = self._delta @ prev_y.T
        self._W -= epsilon * Delta_W
        self._b -= epsilon * self._delta


class LogisticLayer(BaseLayer):
    def __init__(self, n_output, n_prev_output):
        super().__init__(n_output, n_prev_output, logistic, d_logistic)


class TanhLayer(BaseLayer):
    def __init__(self, n_output, n_prev_output, alpha, beta):
        def _tanh(s):
            return tanh(s, alpha, beta)

        def _d_tanh(s):
            return d_tanh(s, alpha, beta)

        super().__init__(n_output, n_prev_output, _tanh, _d_tanh)


class RectifierLayer(BaseLayer):
    def __init__(self, n_output, n_prev_output):
        super().__init__(n_output, n_prev_output, rectifier, d_rectifier)
