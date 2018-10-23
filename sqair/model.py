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

"""Model class which handles training and logging for SQIR.
"""
import tensorflow as tf

import index
import ops
import targets


class Model(object):
    """Generic class for training and evaluating SQAIR models.
    """
    VI_TARGETS = 'iwae reinforce'.split()
    TARGETS = VI_TARGETS

    time_transition_class = None
    prior_rnn_class = None
    output_std = 1.

    def __init__(self, obs, coords, seqence_model, k_particles, presence=None, is_training=None, debug=False):
        """Creates the model.

        :param obs: tf.Tensor of image sequences with shape `[T, B, H, W, C]` with T - number of time-steps, B - batch-size, H - height,
         W - width, C - number of channels.
        :param coords: tf.Tensor of bounding box sequences, necessary for evaluation. Shape is `[T, B, k, 4]` with k
         the maximum number of objects.
        :param seqence_model: Callable that computes all relevant quantities given `obs`.
        :param k_particles: Int, number of particles for the IWAE bound.
        :param presence: tf.Tensor of presence indicator variables; used for evaluation.
        :param debug: Boolean, print or logs additional info if True.
        """

        self.obs = obs
        self.coords = coords
        self.sequence = seqence_model
        self.k_particles = k_particles

        self.gt_presence = presence
        self.debug = debug

        shape = self.obs.shape.as_list()
        self.n_timesteps = self.n_timesteps = shape[0] if shape[0] is not None else tf.shape(self.obs)[0]
        self.batch_size = shape[1]

        self.img_size = shape[2:]
        self.tiled_batch_size = self.batch_size * self.k_particles
        self.tiled_obs = index.tile_input_for_iwae(obs, self.k_particles, with_time=True)

        if self.coords is not None:
            self.tiled_coords = index.tile_input_for_iwae(coords, self.k_particles, with_time=True)

        self._is_training = is_training

        self._build()

    def _build(self):

        inpts = [self.tiled_obs]
        if self.coords is not None:
            inpts.append(self.tiled_coords)

        self.outputs = self.sequence(*inpts)
        self.__dict__.update(self.outputs)

        log_weights = tf.reduce_sum(self.outputs.log_weights_per_timestep, 0)
        self.log_weights = tf.reshape(log_weights, (self.batch_size, self.k_particles))

        self.elbo_vae = tf.reduce_mean(self.log_weights)
        self.elbo_iwae_per_example = targets.iwae(self.log_weights)
        self.elbo_iwae = tf.reduce_mean(self.elbo_iwae_per_example)

        self.normalised_elbo_vae = self.elbo_vae / tf.to_float(self.n_timesteps)
        self.normalised_elbo_iwae = self.elbo_iwae / tf.to_float(self.n_timesteps)
        tf.summary.scalar('normalised_vae', self.normalised_elbo_vae)
        tf.summary.scalar('normalised_iwae', self.normalised_elbo_iwae)

        self.importance_weights = tf.stop_gradient(tf.nn.softmax(self.log_weights, -1))
        self.ess = ops.ess(self.importance_weights, average=True)
        self.iw_distrib = tf.distributions.Categorical(probs=self.importance_weights)
        self.iw_resampling_idx = self.iw_distrib.sample()


        # Logging
        self._log_resampled(self.data_ll_per_sample, 'data_ll')
        self._log_resampled(self.log_p_z_per_sample, 'log_p_z')
        self._log_resampled(self.log_q_z_given_x_per_sample, 'log_q_z_given_x')
        self._log_resampled(self.kl_per_sample, 'kl')

        # Mean squared error between inpt and mean of output distribution
        inpt_obs = self.tiled_obs
        if inpt_obs.shape[-1] == 1:
            inpt_obs = tf.squeeze(inpt_obs, -1)

        axes = [0] + list(range(inpt_obs.shape.ndims)[2:])
        self.mse_per_sample = tf.reduce_mean((inpt_obs - self.canvas) ** 2, axes)
        self._log_resampled(self.mse_per_sample, 'mse')
        self.raw_mse = tf.reduce_mean(self.mse_per_sample)
        tf.summary.scalar('raw_mse', self.raw_mse)

        if hasattr(self, 'num_steps_per_sample'):
            self._log_resampled(self.num_steps_per_sample, 'num_steps')

        if self.gt_presence is not None:
            self.gt_num_steps = tf.reduce_sum(self.gt_presence, -1)

            num_steps_per_sample = tf.reshape(self.num_steps_per_sample, (-1, self.batch_size, self.k_particles))
            gt_num_steps = tf.expand_dims(self.gt_num_steps, -1)

            self.num_step_accuracy_per_example = tf.to_float(tf.equal(gt_num_steps, num_steps_per_sample))
            self.raw_num_step_accuracy = tf.reduce_mean(self.num_step_accuracy_per_example)
            self.num_step_accuracy = self._imp_weighted_mean(self.num_step_accuracy_per_example)
            tf.summary.scalar('num_step_acc', self.num_step_accuracy)

        # For rendering
        resampled_names = 'obj_id canvas glimpse presence_prob presence presence_logit where'.split()
        for name in resampled_names:
            try:
                setattr(self, 'resampled_' + name, self.resample(getattr(self, name), axis=1))
            except AttributeError:
                pass
        try:
            self._log_resampled(self.num_disc_steps_per_sample, 'num_disc_steps')
            self._log_resampled(self.num_prop_steps_per_sample, 'num_prop_steps')
        except AttributeError:
            pass

    def make_target(self, opt, n_train_itr=None, l2_reg=0.):

        if hasattr(self, 'discrete_log_prob'):
            log_probs = tf.reduce_sum(self.discrete_log_prob, 0)
            target = targets.vimco(self.log_weights, log_probs, self.elbo_iwae_per_example)
        else:
            target = -self.elbo_iwae

        target /= tf.to_float(self.n_timesteps)

        target += targets.l2_reg(l2_reg)
        gvs = opt.compute_gradients(target)

        assert len(gvs) == len(tf.trainable_variables())
        for g, v in gvs:
            # print v.name, v.shape.as_list(), g is None
            assert g is not None, 'Gradient for variable {} is None'.format(v)

        return target, gvs

    def resample(self, *args, **kwargs):
        axis = -1
        if 'axis' in kwargs:
            axis = kwargs['axis']
            del kwargs['axis']

        res = list(args)

        if self.k_particles > 1:
            for i, arg in enumerate(res):
                res[i] = self._resample(arg, axis)

        if len(res) == 1:
            res = res[0]

        return res

    def _resample(self, arg, axis=-1):
        iw_sample_idx = self.iw_resampling_idx + tf.range(self.batch_size) * self.k_particles
        shape = arg.shape.as_list()
        shape[axis] = self.batch_size
        resampled = index.gather_axis(arg, iw_sample_idx, axis)
        resampled.set_shape(shape)
        return resampled

    def _log_resampled(self, tensor, name):
        resampled = self._resample(tensor)
        setattr(self, 'resampled_' + name, resampled)
        value = self._imp_weighted_mean(tensor)
        setattr(self, name, value)
        tf.summary.scalar(name, value)

    def _imp_weighted_mean(self, tensor):
        tensor = tf.reshape(tensor, (-1, self.batch_size, self.k_particles))
        tensor = tf.reduce_mean(tensor, 0)
        return tf.reduce_mean(self.importance_weights * tensor * self.k_particles)

    def img_summaries(self):
        recs = tf.cast(tf.round(tf.clip_by_value(self.resampled_canvas, 0., 1.) * 255), tf.uint8)
        rec = tf.summary.image('reconstructions', recs[0])
        inpt = tf.summary.image('inputs', self.obs[0])

        return tf.summary.merge([rec, inpt])


