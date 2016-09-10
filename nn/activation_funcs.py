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
from copy import deepcopy


def logistic(s):
    return 1/(1 + np.exp(-s))


def d_logistic(y):
    return y * (1 - y)


def tanh(s, alpha, beta):
    return alpha * np.tanh(beta * s)


def d_tanh(s, alpha, beta):
    return alpha * beta * (1 - np.power(np.tanh(beta * s), 2))


def rectifier(s):
    val = deepcopy(s)
    val[val < 0.0] = 0.0
    return val


def d_rectifier(s):
    val = deepcopy(s)
    val[val > 0.0] = 1.0
    val[val <= 0.0] = 0.0
    return val
