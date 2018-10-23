########################################################################################
# 
# Sequential Attend, Infer, Repeat (SQAIR)
# Copyright (C) 2018  Adam R. Kosiorek, Oxford Robotics Institute and
#     Department of Statistics, University of Oxford
#
# email:   adamk@robots.ox.ac.uk
# webpage: http://akosiorek.github.io/
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# 
########################################################################################

"""Initially it was a collection of ops used for SQAIR priors, but now I'm not sure.
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as tfd

from ops import clip_preserve
from index import sample_from_tensor


def _cumprod(tensor, axis=0):
    """A custom version of cumprod to prevent NaN gradients when there are zeros in `tensor`
    as reported here: https://github.com/tensorflow/tensorflow/issues/3862.

    :param tensor: tf.Tensor
    :return: tf.Tensor
    """
    transpose_permutation = None
    n_dim = len(tensor.get_shape())
    if n_dim > 1 and axis != 0:

        if axis < 0:
            axis += n_dim

        transpose_permutation = np.arange(n_dim)
        transpose_permutation[-1], transpose_permutation[0] = 0, axis

    tensor = tf.transpose(tensor, transpose_permutation)

    def prod(acc, x):
        return acc * x

    prob = tf.scan(prod, tensor)
    tensor = tf.transpose(prob, transpose_permutation)
    return tensor


def bernoulli_to_modified_geometric(presence_prob):
    presence_prob = tf.cast(presence_prob, tf.float64)
    inv = 1. - presence_prob
    prob = _cumprod(presence_prob, axis=-1)
    modified_prob = tf.concat([inv[..., :1], inv[..., 1:] * prob[..., :-1], prob[..., -1:]], -1)
    modified_prob /= tf.reduce_sum(modified_prob, -1, keep_dims=True)
    return tf.cast(modified_prob, tf.float32)


class NumStepsDistribution(object):
    """Probability distribution used for the number of steps.

    Transforms Bernoulli probabilities of an event = 1 into p(n) where n is the number of steps
    as described in the AIR paper."""

    def __init__(self, steps_probs):
        """

        :param steps_probs: tensor; Bernoulli success probabilities.
        """
        self._steps_probs = steps_probs
        self._joint = bernoulli_to_modified_geometric(steps_probs)
        self._bernoulli = None

    def sample(self, n=None):
        if self._bernoulli is None:
            self._bernoulli = tfd.Bernoulli(self._steps_probs)

        sample = self._bernoulli.sample(n)
        sample = tf.cumprod(sample, tf.rank(sample) - 1)
        sample = tf.reduce_sum(sample, -1)
        return sample

    def prob(self, samples):
        probs = sample_from_tensor(self._joint, samples)
        return probs

    def log_prob(self, samples):
        prob = self.prob(samples)
        
        prob = clip_preserve(prob, 1e-16, 1.)
        return tf.log(prob)

    @property
    def probs(self):
        return self._joint
