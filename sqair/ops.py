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

"""Various ops that did not fit in other files.
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from sqair import nested


def clip_preserve(expr, min, max):
    """Clips the immediate gradient but preserves the chain rule.

    :param expr: tf.Tensor, expr to be clipped
    :param min: float
    :param max: float
    :return: tf.Tensor, clipped expr
    """
    clipped = tf.clip_by_value(expr, min, max)
    return tf.stop_gradient(clipped - expr) + expr


def maybe_getattr(obj, name):
    attr = None
    if name is not None:
        attr = getattr(obj, name, None)
    return attr


def ess(weights, average=False):
    """Effective sample size.
    """
    res = tf.pow(tf.reduce_sum(weights, -1), 2) / tf.reduce_sum(tf.pow(weights, 2), -1)
    if average:
        res = tf.reduce_mean(res)

    return res


def split(x, *args, **kwargs):
    if isinstance(x, tf.Tensor):
        return tf.split(x, *args, **kwargs)
    return np.split(x, *args, **kwargs)


def concat(x, axis):
    if isinstance(x[0], tf.Tensor):
        return tf.concat(x, axis)

    return np.concatenate(x, axis)


def maybe_concat(tensors, axis=-1):
    tensors = nest.flatten(tensors)
    if len(tensors) > 1:
        tensors = tf.concat(tensors, axis)
    else:
        tensors = tensors[0]

    return tensors


def broadcast_against(tensor, against_expr):
    """Adds trailing dimensions to mask to enable broadcasting against data

    :param tensor: tensor to be broadcasted
    :param against_expr: tensor will be broadcasted against it
    :return: mask expr with tf.rank(mask) == tf.rank(data)
    """

    def cond(data, tensor):
        return tf.less(tf.rank(tensor), tf.rank(data))

    def body(data, tensor):
        return data, tf.expand_dims(tensor, -1)

    shape_invariants = [against_expr.get_shape(), tf.TensorShape(None)]
    _, tensor = tf.while_loop(cond, body, [against_expr, tensor], shape_invariants)
    return tensor


def delay_training_for(expr, num_train_iters):

    if num_train_iters == 0:
        return expr

    global_step = tf.train.get_or_create_global_step()
    is_trainable = tf.to_float(tf.greater(global_step, num_train_iters))

    def delay(expr):
        return is_trainable * expr + (1. - is_trainable) * tf.stop_gradient(expr)

    return nested.map_nested(delay, expr)


def delayed_trainable_variable(num_training_iters, *args, **kwargs):

    var = tf.get_variable(*args, trainable=True, **kwargs)
    return delay_training_for(var, num_training_iters)