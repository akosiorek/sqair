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

"""Optimisation targets and control variates.
"""
import tensorflow as tf

import math


def l2_reg(weight):
    if weight == 0.:
        return 0.

    return weight * sum(map(tf.nn.l2_loss, tf.trainable_variables()))


def iwae(log_weights):
    """Importance-weighted ELBO.
    """

    k_particles = log_weights.shape.as_list()[-1]
    return tf.reduce_logsumexp(log_weights, -1) - math.log(float(k_particles))


def vimco_control_variate(target_per_particle):
    """Computes VIMCO control variates for the given targets

    :param target_per_particle: tf.Tensor of shape [..., k_particles]
    :return: control variate of the shape same as `per_sample_target`
    """

    k_particles = int(target_per_particle.shape[-1])
    summed_per_particle_target = tf.reduce_sum(target_per_particle, -1, keep_dims=True)
    all_but_one_average = (summed_per_particle_target - target_per_particle) / (k_particles - 1.)

    diag = tf.matrix_diag(all_but_one_average - target_per_particle)
    baseline = target_per_particle[..., tf.newaxis] + diag
    return tf.reduce_logsumexp(baseline, -2) - math.log(float(k_particles))


def vimco(log_weights, log_probs, elbo_iwae=None):
    """Computes VIMCO target.
    """

    control_variate = vimco_control_variate(log_weights)
    learning_signal = tf.stop_gradient(log_weights - control_variate)
    log_probs = tf.reshape(log_probs, tf.shape(log_weights))
    reinforce_target = learning_signal * log_probs

    if elbo_iwae is None:
        elbo_iwae = iwae(log_weights)

    proxy_loss = -tf.expand_dims(elbo_iwae, -1) - reinforce_target
    return tf.reduce_mean(proxy_loss)


def reinforce(log_weights, log_probs, elbo_iwae=None):
    """Computes REINFORCE target.
    """

    learning_signal = tf.stop_gradient(log_weights)
    log_probs = tf.reshape(log_probs, tf.shape(log_weights))
    reinforce_target = learning_signal * log_probs

    if elbo_iwae is None:
        elbo_iwae = iwae(log_weights)

    proxy_loss = -tf.expand_dims(elbo_iwae, -1) - reinforce_target
    return tf.reduce_mean(proxy_loss)