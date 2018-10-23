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

"""Sequential models.

These models handle unrolling over a time-series of inputs and all necessary book-keeping.
"""
import sonnet as snt
import tensorflow as tf

import tensorflow.contrib.distributions as tfd
from tensorflow.python.util import nest

from attrdict import AttrDict

from core import BaseSQAIRCore
from sqair_modules import SQAIRTimestep, PropagateOnlyTimestep
from modules import SpatialTransformer


class SequentialAIR(snt.AbstractModule):
    """Unrolls SQAIR over a time-series of inputs.
    """
    
    def __init__(self, max_steps, glimpse_size, discover, propagate, time_cell, decoder,
                 sample_from_prior=False, generate_after=-1):
        """Initialises the module.

        :param max_steps:
        :param glimpse_size:
        :param discover:
        :param propagate:
        :param time_cell:
        :param decoder:
        :param sample_from_prior:
        :param generate_after:
        """

        super(SequentialAIR, self).__init__()
        self._max_steps = max_steps
        self._glimpse_size = glimpse_size
        self._decoder = decoder
        self._sample_from_prior = sample_from_prior
        self._generate_after = generate_after

        with self._enter_variable_scope():
            self._sqair = SQAIRTimestep(self._max_steps, discover, propagate, time_cell)

    def _build(self, obs, coords=None, sample_from_prior=False):
        """Unrolls the model in time.

        :param obs: Sequence of images of the shape `[T, B, H, W, C]`.
        :param coords: Should be always None; required for compatilibity reasons.
        :param sample_from_prior: Sampels from the prior instead of the inference networks if True.
        :return: A dictionary of outputs, look at `self._loop_body` for an idea.
        """
        batch_size = int(obs.shape[1])
        loop_vars, ta_names = self._prepare_loop_vars(batch_size, obs, coords)

        res = tf.while_loop(self._loop_cond, self._loop_body, loop_vars, parallel_iterations=batch_size)

        tas = res[len(res)-len(ta_names):]
        outputs = AttrDict({k: ta.stack() for (k, ta) in zip(ta_names, tas)})
        return outputs

    def _prepare_loop_vars(self, batch_size, obs, coords=None):
        """Prepares the initial state.

        :param batch_size: Int, batch size.
        :param obs: Images.
        :param coords: Should be None.
        :return:
        """

        t = tf.constant(0, dtype=tf.int32, name='time')
        z0 = self._sqair.initial_z(batch_size)
        time_state = self._sqair.initial_temporal_state(batch_size, tf.float32, trainable=True)
        prev_ids = -tf.ones((batch_size, self._max_steps, 1))
        last_used_id = -tf.ones((batch_size, 1))
        init_prior_prop_state = self._sqair.initial_prior_state(batch_size)

        loop_vars = [
            t, obs, z0, time_state, prev_ids, last_used_id, init_prior_prop_state
        ]

        n_timesteps = tf.shape(obs)[0]
        img_size = obs.shape[2:].as_list()
        tas = []
        ta_names = []

        def make_ta(name, shape=[], usual_shape=True):
            assert name not in ta_names, 'Name "{}" already exists!'.format(name)

            if usual_shape:
                shape = [batch_size] + shape
            ta = tf.TensorArray(tf.float32, n_timesteps, dynamic_size=False, element_shape=shape)
            tas.append(ta)
            ta_names.append(name)

        # RNN outputs
        make_ta('what', [self._max_steps, self._sqair.n_what])
        make_ta('what_loc', [self._max_steps, self._sqair.n_what])
        make_ta('what_scale', [self._max_steps, self._sqair.n_what])
        make_ta('where', [self._max_steps, 4])
        make_ta('where_loc', [self._max_steps, 4])
        make_ta('where_scale', [self._max_steps, 4])
        make_ta('presence_prob', [self._max_steps])
        make_ta('presence', [self._max_steps])
        make_ta('presence_logit', [self._max_steps])

        # Aux, returned as hidden outputs
        make_ta('obj_id', [self._max_steps])
        make_ta('step_log_prob')

        # Others

        n_channels = img_size[-1]
        canvas_size = list(img_size[:2])
        glimpse_size = list(self._glimpse_size)

        if n_channels != 1:
            canvas_size.append(n_channels)
            glimpse_size.append(n_channels)

        make_ta('canvas', canvas_size)
        make_ta('glimpse', [self._max_steps] + glimpse_size)
        # 
        make_ta('disc_what_log_prob', [self._max_steps])
        make_ta('disc_where_log_prob', [self._max_steps])
        make_ta('disc_what_prior_log_prob', [self._max_steps])
        make_ta('disc_where_prior_log_prob', [self._max_steps])
        make_ta('disc_log_prob')
        make_ta('disc_prior_log_prob')
        make_ta('disc_prob', [self._max_steps + 1])
        # 
        make_ta('prop_what_log_prob', [self._max_steps])
        make_ta('prop_where_log_prob', [self._max_steps])
        make_ta('prop_what_prior_log_prob', [self._max_steps])
        make_ta('prop_where_prior_log_prob', [self._max_steps])
        make_ta('prop_log_prob')
        make_ta('prop_prior_log_prob')
        make_ta('prop_prob', [self._max_steps])
        #
        make_ta('discrete_log_prob')
        #
        make_ta('num_prop_steps_per_sample')
        make_ta('num_disc_steps_per_sample')
        make_ta('num_steps_per_sample')
        #
        make_ta('prop_pres', [self._max_steps])
        make_ta('disc_pres', [self._max_steps])
        #
        make_ta('data_ll_per_sample')
        make_ta('kl_per_sample')
        make_ta('log_q_z_given_x_per_sample')
        make_ta('log_p_z_per_sample')
        make_ta('log_weights_per_timestep')

        return loop_vars + tas, ta_names

    def _loop_body(self, t, img_seq, z_tm1, time_state, prev_ids, last_used_id, prop_prior_hidden_state, *tas):
        """Implements a single time-step as the body of the temporal while loop.

        :param t: Current time-step.
        :param img_seq: Sequence of images.
        :param z_tm1: Latent variables from the previous time-step.
        :param time_state: Hidden state of the temporal RNN.
        :param prev_ids: Object IDs at the previous time-step.
        :param last_used_id: Maximum object ID used so far.
        :param prop_prior_hidden_state: Hidden state of the propagation prior.
        :param tas: TensorArrays prepared in `self._prepare_loop_vars`.
        :return: Results and book-keeping variables.
        """

        # parse inputs
        img = img_seq[t]

        do_generate = False
        if self._generate_after > 0:
            do_generate = tf.greater(t, self._generate_after)

        apdr_outputs = self._sqair(img, z_tm1, time_state, prop_prior_hidden_state, last_used_id, prev_ids,
                                   t, self._sample_from_prior, do_generate)
        hidden_outputs = apdr_outputs.hidden_outputs
        z_t = apdr_outputs.z_t

        # ## Decode
        p_x_given_z, glimpse = self._decoder(*z_t[:-1])
        data_ll, kl, log_weights = self._compute_log_weights(img, p_x_given_z, apdr_outputs)

        prop, disc = apdr_outputs.prop, apdr_outputs.disc
        
        # write outputs
        tas = list(tas)
        outputs = list(hidden_outputs.values()) + \
        [
            apdr_outputs.obj_ids,
            apdr_outputs.presence_log_prob,
            p_x_given_z.mean(),
            glimpse,
            # 
            disc.what_log_prob,
            disc.where_log_prob,
            disc.what_prior_log_prob,
            disc.where_prior_log_prob,
            disc.num_step_log_prob,
            disc.num_step_prior_log_prob,
            disc.num_steps_prob,
            # 
            prop.what_log_prob,
            prop.where_log_prob,
            prop.what_prior_log_prob,
            prop.where_prior_log_prob,
            prop.prop_log_prob,
            prop.prop_prior_log_prob,
            prop.prop_prob,
            #
            prop.prop_log_prob + disc.num_step_log_prob,  # discrete log prob
            #
            prop.num_steps,
            disc.num_steps,
            apdr_outputs.num_steps,
            prop.presence,
            disc.presence,
            #
            data_ll,
            kl,
            apdr_outputs.q_z_given_x,
            apdr_outputs.p_z,
            log_weights,
        ]

        for i, (ta, output) in enumerate(zip(tas, outputs)):
            if int(output.shape[-1]) == 1:
                output = tf.squeeze(output, -1)

            tas[i] = ta.write(t, output)

        # Increment time index
        t += 1
        time_state = apdr_outputs.temporal_hidden_state

        outputs = [
            t, img_seq, z_t, time_state, apdr_outputs.ids, apdr_outputs.highest_used_ids,
            apdr_outputs.prop_prior_state

        ] + tas

        return outputs

    def _compute_log_weights(self, img, p_x_given_z, outputs):
        data_ll_per_pixel = p_x_given_z.log_prob(img)
        data_ll = tf.reduce_sum(data_ll_per_pixel, (1, 2, 3))
        kl = outputs.q_z_given_x - outputs.p_z
        log_weight = data_ll - kl
        return data_ll, kl, log_weight

    def _loop_cond(self, t, img_seq, *args):
        return t < tf.shape(img_seq)[0]
