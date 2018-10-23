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

"""Implementation of an state-space model and priors for propagation.
"""
import collections

import numpy as np
import sonnet as snt
import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli, Normal
from tensorflow.python.util import nest


def make_prior(name, *args, **kwargs):
    prior_map = {
        'rnn': PropagatePrior,
        'rw': RandomWalkPropagatePrior,
        'guided': GuidedWalkPropagatePrior
    }

    if name not in prior_map:
        raise ValueError('Invalid prior type: "{}". Choose from {}.'.format(name, prior_map.keys()))

    return prior_map[name](*args, **kwargs)


class PropagatePrior(snt.AbstractModule):
    """Flexible RNN prior for propagation.

    This implementation treats all objects as independet.
    """

    def __init__(self, n_what, cell, prop_logit_bias, where_loc_bias=None):
        """Initialises the module.

        :param n_what:
        :param cell:
        :param prop_logit_bias:
        :param where_loc_bias:
        """
        super(PropagatePrior, self).__init__()
        self._n_what = n_what
        self._cell = cell
        self._prop_logit_bias = prop_logit_bias
        self._where_loc_bias = where_loc_bias

    def _build(self, z_tm1, prior_rnn_hidden_state):
        """Applies the op.

        :param z_tm1:
        :param prior_rnn_hidden_state:
        :return:
        """
        what_tm1, where_tm1, presence_tm1 = z_tm1[:3]

        prior_rnn_inpt = tf.concat((what_tm1, where_tm1), -1)
        rnn = snt.BatchApply(self._cell)

        outputs, prior_rnn_hidden_state = rnn(prior_rnn_inpt, prior_rnn_hidden_state)
        n_outputs = 2 * (4 + self._n_what) + 1
        stats = snt.BatchApply(snt.Linear(n_outputs))(outputs)

        prop_prob_logit, stats = tf.split(stats, [1, n_outputs - 1], -1)
        prop_prob_logit += self._prop_logit_bias
        prop_prob_logit = presence_tm1 * prop_prob_logit + (presence_tm1 - 1.) * 88.

        locs, scales = tf.split(stats, 2, -1)
        prior_where_loc, prior_what_loc = tf.split(locs, [4, self._n_what], -1)
        prior_where_scale, prior_what_scale = tf.split(scales, [4, self._n_what], -1)
        prior_where_scale, prior_what_scale = (tf.nn.softplus(i) + 1e-2 for i in (prior_where_scale, prior_what_scale))

        if self._where_loc_bias is not None:
            bias = np.asarray(self._where_loc_bias).reshape((1, 4))
            prior_where_loc += bias

        prior_stats = (prior_where_loc, prior_where_scale, prior_what_loc, prior_what_scale, prop_prob_logit)
        return prior_stats, prior_rnn_hidden_state

    def initial_state(self, batch_size, trainable=True, initializer=None):
        if initializer is not None and not isinstance(initializer, collections.Sequence):
            state_size = self._cell.state_size
            flat_state_size = nest.flatten(state_size)
            initializer = [initializer] * len(flat_state_size)
            initializer = nest.pack_sequence_as(state_size, initializer)

        init_state = self._cell.initial_state(batch_size, tf.float32,
                                              trainable=trainable,
                                              trainable_initializers=initializer)

        return init_state

    def make_distribs(self, (prior_where_loc, prior_where_scale, prior_what_loc, prior_what_scale, prop_prob_logit)):
        """Converts parameters return by `_build` into probability distributions.
        """
        what_prior = Normal(prior_what_loc, prior_what_scale)
        where_prior = Normal(prior_where_loc, prior_where_scale)
        prop_prior = Bernoulli(logits=tf.squeeze(prop_prob_logit, -1))

        return what_prior, where_prior, prop_prior


class RandomWalkPropagatePrior(PropagatePrior):
    """"Flexible RNN prior for propagation centred on latent variables from the previous time-step.
    """

    def _build(self, z_tm1, prior_rnn_hidden_state):

        prior_stats, prior_rnn_hidden_state \
            = super(RandomWalkPropagatePrior, self)._build(z_tm1, prior_rnn_hidden_state)

        what_tm1, where_tm1, presence_tm1, presence_logit_tm1 = z_tm1
        prior_stats = list(prior_stats)

        alpha = .1
        prior_stats[0] = where_tm1
        prior_stats[2] = what_tm1
        prior_stats[4] = presence_logit_tm1 + alpha * prior_stats[4]
        return tuple(prior_stats), prior_rnn_hidden_state


class GuidedWalkPropagatePrior(PropagatePrior):
    """"Flexible RNN prior, whose statistics are computed relative to statistics at the previous time-step.
    """

    def _build(self, z_tm1, prior_rnn_hidden_state):

        prior_stats, prior_rnn_hidden_state \
            = super(GuidedWalkPropagatePrior, self)._build(z_tm1, prior_rnn_hidden_state)

        what_tm1, where_tm1, presence_tm1, presence_logit_tm1 = z_tm1
        prior_stats = list(prior_stats)

        alpha = .1
        prior_stats[0] = where_tm1 + alpha * prior_stats[0]
        prior_stats[2] = what_tm1 + alpha * prior_stats[2]
        prior_stats[4] = presence_logit_tm1 + alpha * prior_stats[4]
        return tuple(prior_stats), prior_rnn_hidden_state


class SequentialSSM(snt.AbstractModule):
    """State-space model used for propagation."""

    def __init__(self, cell):
        super(SequentialSSM, self).__init__()
        self._cell = cell

    def _build(self, img, z_tm1, temporal_hidden_state):

        initial_state = self._cell.initial_state(img)
        unstacked_z_tm1 = zip(*[tf.unstack(z, axis=-2) for z in z_tm1])
        unstacked_temp_state = tf.unstack(temporal_hidden_state, axis=-2)
        inpt = zip(unstacked_z_tm1, unstacked_temp_state)

        hidden_outputs, hidden_state = tf.nn.static_rnn(self._cell, inpt, initial_state)
        hidden_outputs = self._cell.outputs_by_name(hidden_outputs)

        delta_what, delta_where = hidden_outputs.what_sample, hidden_outputs.where_sample
        del hidden_outputs.what_sample
        del hidden_outputs.where_sample

        num_steps = tf.reduce_sum(tf.squeeze(hidden_outputs.presence, -1), -1)

        return hidden_outputs, num_steps, delta_what, delta_where

    def _temporal_to_step_hidden_state(self, temporal_hidden_state):
        """Linear projection of the temporal hidden state to the step-wise hidden state.
        """

        with tf.variable_scope('temporal_to_step_hidden_state'):
            flat_hidden_state = tf.concat(nest.flatten(temporal_hidden_state), -1)
            state_size = self._cell.state_size[-1]
            flat_state_size = sum([int(s) for s in state_size])
            state = snt.Linear(flat_state_size)(flat_hidden_state)
            state = tf.split(state, state_size, -1)

        if len(state) == 1:
            state = state[0]

        return state