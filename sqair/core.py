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

"""RNN cores for discovery and propagation.
"""

import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from tensorflow.python.util import nest

import orderedattrdict

from modules import SpatialTransformer, AffineDiagNormal, GaussianFromParamVec
from neural import MLP, Nonlinear
import nested


class BaseSQAIRCore(snt.RNNCore):
    """Base class for recurrent SQAIR cores.

    Derived classes should set `_init_presence_value` and `_output_names` accordingly.
    """

    _n_transform_param = 4
    _init_presence_value = None
    _output_names = None
    _what_scale_bias = 0.

    def __init__(self, img_size, crop_size, n_what,
                 transition, input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                 where_loc_bias=None,
                 debug=False):
        """Creates the cell

        :param img_size: int tuple, size of the image
        :param crop_size: int tuple, size of the attention glimpse
        :param n_what: number of latent units describing the "what"
        :param transition: an RNN cell for maintaining the internal hidden state
        :param input_encoder: callable, encodes the original input image before passing it into the transition
        :param glimpse_encoder: callable, encodes the glimpse into latent representation
        :param transform_estimator: callabe, transforms the hidden state into parameters for the spatial transformer
        :param steps_predictor: callable, predicts whether to take a step
        :param debug: boolean, adds checks for NaNs in the inputs to distributions
        """

        super(BaseSQAIRCore, self).__init__()
        self._img_size = img_size
        self._n_pix = np.prod(self._img_size)
        self._crop_size = crop_size
        self._n_what = n_what
        self._cell = transition
        self._n_hidden = int(self._cell.output_size[0])

        self._where_loc_bias = where_loc_bias

        self._debug = debug

        with self._enter_variable_scope():

            self._spatial_transformer = SpatialTransformer(img_size, crop_size)
            self._transform_estimator = transform_estimator()
            self._input_encoder = input_encoder()
            self._glimpse_encoder = glimpse_encoder()
            self._steps_predictor = steps_predictor()

    @property
    def n_what(self):
        return self._n_what

    @property
    def n_where(self):
        return self._n_where

    @property
    def state_size(self):
        return [
            np.prod(self._img_size),  # image
            self._n_what,  # what
            self._n_transform_param,  # where
            1,  # presence
            self._cell.state_size,  # hidden state of the rnn
        ]

    @property
    def output_names(self):
        return self._output_names

    @classmethod
    def outputs_by_name(cls, hidden_outputs, stack=True):
        if stack:
            hidden_outputs = nested.stack(hidden_outputs, axis=1)

        d = orderedattrdict.AttrDict()
        for n, o in zip(cls._output_names, hidden_outputs):
            d[n] = o

        return d

    def initial_state(self, img, hidden_state=None):
        """Initialises the hidden state.

        :param img: Image to perform inference on.
        :param hidden_state: If not None, uses it as the hidden state for the internal RNN.
        """
        batch_size = img.get_shape().as_list()[0]

        if hidden_state is None:
            hidden_state = self._cell.initial_state(batch_size, tf.float32, trainable=True)

        where_code = tf.zeros([1, self._n_transform_param], dtype=tf.float32, name='where_init')
        what_code = tf.zeros([1, self._n_what], dtype=tf.float32, name='what_init')

        where_code, what_code = (tf.tile(i, (batch_size, 1)) for i in (where_code, what_code))

        flat_img = tf.reshape(img, (batch_size, self._n_pix))
        init_presence = tf.ones((batch_size, 1), dtype=tf.float32) * self._init_presence_value
        return [flat_img, what_code, where_code, init_presence, hidden_state]

    def _compute_presence(self, previous_presence, presence_logit, *features):
        presence_distrib = self._steps_predictor(previous_presence, presence_logit, features)
        presence = presence_distrib.sample() * previous_presence
        return presence, presence_distrib.probs, presence_distrib.logits


class DiscoveryCore(BaseSQAIRCore):
    """Recurrent discovery core.

    Discovery core represents a single inference step for discovery. It is run iteratively to discover many objects.
    """
    _output_names = 'what what_loc what_scale where where_loc where_scale presence_prob presence presence_logit'.split()
    _init_presence_value = 1.  # at the beginning we assume all objects were present
    _what_scale_bias = 0.5

    def initial_z(self, batch_size, n_steps):
        what = tf.zeros((1, 1, self._n_what))
        where = tf.zeros((1, 1, 4))
        presence = tf.zeros((1, 1, 1))
        presence_logit = tf.zeros((1, 1, 1))
        z0 = [tf.tile(i, (batch_size, n_steps, 1)) for i in what, where, presence, presence_logit]
        return z0

    def _prepare_rnn_inputs(self, inpt, img, what, where, presence):
        rnn_inpt = [self._input_encoder(img)]
        if inpt is not None:
            rnn_inpt.extend(nest.flatten(inpt))

        rnn_inpt.extend([what, where, presence])

        if len(rnn_inpt) > 1:
            rnn_inpt = tf.concat(rnn_inpt, -1)
        else:
            rnn_inpt = rnn_inpt[0]

        return rnn_inpt

    @property
    def output_size(self):
        return [
            self._n_what,  # what code
            self._n_what,  # what loc
            self._n_what,  # what scale
            self._n_transform_param,  # where code
            self._n_transform_param,  # where loc
            self._n_transform_param,  # where scale
            1,  # presence prob
            1,  # presence
            1  # presence_logit
        ]

    def _build(self, (inpt, is_allowed), (img_flat, what_code, where_code, presence, hidden_state)):
        """Input is unused; it's only to force a maximum number of steps"""
        img = tf.reshape(img_flat, (-1,) + tuple(self._img_size))

        with tf.variable_scope('rnn_inpt'):
            rnn_inpt = self._prepare_rnn_inputs(inpt, img, what_code, where_code, presence)
            hidden_output, hidden_state = self._cell(rnn_inpt, hidden_state)

        with tf.variable_scope('where'):
            where_code, where_loc, where_scale = self._compute_where(hidden_output)

        with tf.variable_scope('what'):
            what_code, what_loc, what_scale = self._compute_what(img, where_code)

        with tf.variable_scope('presence'):
            presence, presence_prob, presence_logit\
                = self._compute_presence(presence, None, hidden_output, what_code)

        output = [what_code, what_loc, what_scale, where_code, where_loc, where_scale,
                  presence_prob, presence, presence_logit]
        new_state = [img_flat, what_code, where_code, presence, hidden_state]

        return output, new_state

    def _compute_what(self, img, where_code):
        what_distrib = self._glimpse_encoder(img, where_code)[0]
        return what_distrib.sample(), what_distrib.loc, what_distrib.scale

    def _compute_where(self, hidden_output):
        loc, scale = self._transform_estimator(hidden_output)
        if self._where_loc_bias is not None:
            loc += np.asarray(self._where_loc_bias).reshape((1, 4))

        scale = tf.nn.softplus(scale) + 1e-2
        where_distrib = tfd.Normal(loc, scale, validate_args=self._debug, allow_nan_stats=not self._debug)
        return where_distrib.sample(), loc, scale


class PropagationCore(BaseSQAIRCore):
    """Recurrent propagation core.

    It is run iteratively to propagate several objects.
    """
    _output_names = 'what what_sample what_loc what_scale where where_sample where_loc where_scale presence_prob' \
                    ' presence presence_logit temporal_state'.split()

    _init_presence_value = 0.  # at the beginning we assume no objects
    _what_scale_bias = -3.

    def __init__(self, img_size, crop_size, n_what,
                 transition, input_encoder, glimpse_encoder, transform_estimator, steps_predictor, temporal_cell,
                 where_update_scale=1.0, debug=False):
        """Initialises the model.

        If argument is not covered here, see BaseSQAIRCore for documentation.

        :param temporal_cell: RNNCore for the temporal rnn.
        :param where_update_scale: Float, rescales the update of the `where` variables.
        """

        super(PropagationCore, self).__init__(img_size, crop_size, n_what, transition, input_encoder,
                                              glimpse_encoder, transform_estimator, steps_predictor,
                                              debug=debug)

        self._temporal_cell = temporal_cell
        with self._enter_variable_scope():
            self._where_update_scale = tf.get_variable('where_update_scale', shape=[], dtype=tf.float32,
                                                       initializer=tf.constant_initializer(where_update_scale),
                                                       trainable=False)
            self._where_distrib = AffineDiagNormal(validate_args=self._debug, allow_nan_stats=not self._debug)

    @property
    def output_size(self):
        return [
            self._n_what,  # what code
            self._n_what,  # what sample
            self._n_what,  # what loc
            self._n_what,  # what scale
            self._n_transform_param,  # where code
            self._n_transform_param,  # where sample
            self._n_transform_param,  # where loc
            self._n_transform_param,  # where scale
            1,  # presence prob
            1,  # presence
            1,  # presence_logit,
            self._temporal_cell.state_size,
        ]

    def _build(self, (z_tm1, temporal_hidden_state), state):
        """Input is unused; it's only to force a maximum number of steps"""
        # same object, previous timestep
        what_tm1, where_tm1, presence_tm1, presence_logit_tm1 = z_tm1
        temporal_state = nest.flatten(temporal_hidden_state)[-1]

        # different object, current timestep
        img_flat, what_km1, where_km1, presence_km1, hidden_state = state

        img = tf.reshape(img_flat, (-1,) + tuple(self._img_size))
        with tf.variable_scope('rnn_inpt'):
            where_bias = MLP(128, n_out=4)(temporal_state) * .1
            what_distrib = self._glimpse_encoder(img, where_tm1 + where_bias, mask_inpt=temporal_state)[0]
            rnn_inpt = what_distrib.loc

            rnn_inpt = [
                rnn_inpt,                                             # img
                what_km1, where_km1, presence_km1,                    # explaining away
                what_tm1, where_tm1, presence_tm1, temporal_state     # previous state
            ]

            rnn_inpt = tf.concat(rnn_inpt, -1)
            hidden_output, hidden_state = self._cell(rnn_inpt, hidden_state)

        with tf.variable_scope('where'):
            where, where_sample, where_loc, where_scale = self._compute_where(where_tm1, hidden_output, temporal_state)

        with tf.variable_scope('what'):
            what, what_sample, what_loc, what_scale, temporal_hidden_state\
                = self._compute_what(img, what_tm1, where, hidden_output, temporal_hidden_state, temporal_state)

        with tf.variable_scope('presence'):
            presence, presence_prob, presence_logit \
                = self._compute_presence(presence_tm1, presence_logit_tm1, hidden_output, temporal_state, what)

        output = [what, what_sample, what_loc, what_scale, where, where_sample, where_loc, where_scale,
                  presence_prob, presence, presence_logit, temporal_hidden_state]
        new_state = [img_flat, what, where, presence, hidden_state]

        return output, new_state

    def _compute_where(self, where_tm1, hidden_output, temporal_state):

        inpt = tf.concat((hidden_output, where_tm1, temporal_state), -1)
        loc, scale = self._transform_estimator(inpt)

        loc = where_tm1 + self._where_update_scale * loc
        scale = tf.nn.softplus(scale - 1.) + 1e-2

        where_distrib = self._where_distrib(loc, scale)
        where_sample = where_distrib.sample()

        where = where_sample
        return where, where_sample, loc, scale

    def _compute_what(self, img, what_tm1, where, hidden_output, temporal_hidden_state, temporal_state):
        what_distrib = self._glimpse_encoder(img, where, mask_inpt=temporal_state)[0]
        loc, scale = what_distrib.loc, what_distrib.scale

        inpt = tf.concat((hidden_output, where, loc, scale), -1)
        temporal_output, temporal_hidden_state = self._temporal_cell(inpt, temporal_hidden_state)

        n_dim = int(what_tm1.shape[-1])
        temporal_distrib = GaussianFromParamVec(n_dim)(temporal_output)

        remember_bias = {'b': tf.constant_initializer(1.)}
        gates = Nonlinear(n_dim * 3, tf.nn.sigmoid, remember_bias)(temporal_output)

        gates *= .9999
        forget_gate, input_gate, temporal_gate = tf.split(gates, 3, -1)

        what_distrib = tfd.Normal(
            loc=forget_gate * what_tm1 + (1. - input_gate) * loc + (1. - temporal_gate) * temporal_distrib.loc,
            scale=(1. - input_gate) * scale + (1. - temporal_gate) * temporal_distrib.scale
        )

        what_sample = what_distrib.sample()
        what = what_sample

        return what, what_sample, what_distrib.loc, what_distrib.scale, temporal_hidden_state
