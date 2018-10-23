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

"""Functions used for tensor indexing.
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from ops import broadcast_against


def sample_from_1d_tensor(arr, idx):
    """Takes samples from `arr` indicated by `idx`

    :param arr:
    :param idx:
    :return:
    """
    arr = tf.convert_to_tensor(arr)
    assert len(arr.get_shape()) == 1, "shape is {}".format(arr.get_shape())

    idx = tf.to_int32(idx)
    arr = tf.gather(tf.squeeze(arr), idx)
    return arr


def sample_from_tensor(tensor, idx):
    """Takes sample from `tensor` indicated by `idx`, works for minibatches

    :param tensor:
    :param idx:
    :return:
    """
    tensor = tf.convert_to_tensor(tensor)

    assert tensor.shape.ndims == (idx.shape.ndims + 1) \
           or ((tensor.shape.ndims == idx.shape.ndims) and (idx.shape[-1] == 1)), \
        'Shapes: tensor={} vs idx={}'.format(tensor.shape.ndims, idx.shape.ndims)

    batch_shape = tf.shape(tensor)[:-1]
    trailing_dim = int(tensor.shape[-1])
    n_elements = tf.reduce_prod(batch_shape)
    shift = tf.range(n_elements) * trailing_dim

    tensor_flat = tf.reshape(tensor, (-1,))
    idx_flat = tf.reshape(tf.to_int32(idx), (-1,)) + shift
    samples_flat = sample_from_1d_tensor(tensor_flat, idx_flat)
    samples = tf.reshape(samples_flat, batch_shape)

    return samples


def gather_axis(tensor, idx, axis=-1):
    """Gathers indices `idx` from `tensor` along axis `axis`

    The shape of the returned tensor is as follows:
    >>> shape = tensor.shape
    >>> shape[axis] = len(idx)
    >>> return shape

    :param tensor: n-D tf.Tensor
    :param idx: 1-D tf.Tensor
    :param axis: int
    :return: tf.Tensor
    """

    axis = tf.convert_to_tensor(axis)
    neg_axis = tf.less(axis, 0)
    axis = tf.cond(neg_axis, lambda: tf.shape(tf.shape(tensor))[0] + axis, lambda: axis)
    shape = tf.shape(tensor)
    pre, post = shape[:axis + 1], shape[axis + 1:]
    shape = tf.concat((pre[:-1], tf.shape(idx)[:1], post), -1)

    n = tf.reduce_prod(pre[:-1])
    idx = tf.tile(idx[tf.newaxis], (n, 1))
    idx += tf.range(n)[:, tf.newaxis] * pre[-1]
    linear_idx = tf.reshape(idx, [-1])

    flat = tf.reshape(tensor, tf.concat(([n * pre[-1]], post), -1))
    flat = tf.gather(flat, linear_idx)
    tensor = tf.reshape(flat, shape)
    return tensor


def tile_input_for_iwae(tensor, iw_samples, with_time=False):
    """Tiles tensor `tensor` in such a way that tiled samples are contiguous in memory;
    i.e. it tiles along the axis after the batch axis and reshapes to have the same rank as
    the original tensor

    :param tensor: tf.Tensor to be tiled
    :param iw_samples: int, number of importance-weighted samples
    :param with_time: boolean, if true than an additional axis at the beginning is assumed
    :return:
    """

    shape = tensor.shape.as_list()
    if with_time:
        shape[0] = tf.shape(tensor)[0]
    shape[with_time] *= iw_samples

    tiles = [1, iw_samples] + [1] * (tensor.shape.ndims - (1 + with_time))
    if with_time:
        tiles = [1] + tiles

    tensor = tf.expand_dims(tensor, 1 + with_time)
    tensor = tf.tile(tensor, tiles)
    tensor = tf.reshape(tensor, shape)
    return tensor


def select_present(x, presence, batch_size=None, name='select_present'):
    """Rearranges entries in `x` according to the binary `presence` vector.

    Rearranges entires in `x` according to the binary `presence` tensor. The resulting tensor is the same as the input,
    with the difference that those entries for whose corresponding `presence` entries are True are moved to the
    beginning of that axis (smaller values of index) and those for which `presence` evaluates to False are moved to the
    end. The respective ordering is preserved for items with the same value of `presence`.

    :param x: A feature tensor of shape `[B, K, d]`.
    :param presence: A binary tensor of shape `[B, K]`.
    :param batch_size: Integer, represents `B`.
    :param name: String, name of the operation
    :return: Rearranged feature vector `x`.
    """
    with tf.variable_scope(name):
        presence = 1 - tf.to_int32(presence)  # invert mask

        if batch_size is None:
            try:
                batch_size = int(x.shape[0])
            except TypeError:
                raise ValueError('Batch size cannot be determined. Please provide it as an argument.')

        num_partitions = 2 * batch_size
        r = tf.range(0, num_partitions, 2)
        r.set_shape(tf.TensorShape(batch_size))
        r = broadcast_against(r, presence)
        presence += r

        selected = tf.dynamic_partition(x, presence, num_partitions)
        selected = tf.concat(selected, 0)
        selected = tf.reshape(selected, tf.shape(x))

    return selected


def select_present_nested(tensors, presence, batch_size=None, name='select_present_nested'):
    """Like `select_present`, but handles nested tensors.

     It concatenates the tensors along the last dimension, calls `select_present` only once
     and splits the tensors again. It's faster and the graph is less complicated that
     way.

    :param tensors:
    :param presence:
    :param batch_size:
    :param name:
    :return:
    """
    orig_inpt = tensors
    with tf.variable_scope(name):
        tensors = nest.flatten(tensors)
        lens = [0] + [int(t.shape[-1]) for t in tensors]
        lens = np.cumsum(lens)

        merged = tf.concat(tensors, -1)
        merged = select_present(merged, presence, batch_size)
        tensors = []

        for i in xrange(len(lens) - 1):
            st, ed = lens[i], lens[i + 1]
            tensors.append(merged[..., st:ed])

    return nest.pack_sequence_as(structure=orig_inpt, flat_sequence=tensors)


def compute_object_ids(last_used_id, prev_ids, propagated_pres, discovery_pres):
    """Computes IDs of propagated and discovered objects.

    :param last_used_id: Scalar tensor; the maximum value of ID used so far.
    :param prev_ids: Tensor with IDs of objects present at the previous time-step.
    :param propagated_pres: Tensor of presence variables for propagated objects.
    :param discovery_pres: Tensor of presence variables for newely discovered objects.
    :return: Maximum id used so far, new ids of all present objects
    """
    last_used_id, prev_ids, propagated_pres, discovery_pres = [tf.convert_to_tensor(i) for i in (
    last_used_id, prev_ids, propagated_pres, discovery_pres)]
    prop_ids = prev_ids * propagated_pres - (1 - propagated_pres)

    # each object gets a new id
    id_increments = tf.cumsum(discovery_pres, 1)
    # find the new id by incrementing the last used ids
    disc_ids = id_increments + last_used_id[:, tf.newaxis]

    # last used ids needs to be incremented by the maximum value
    last_used_id += id_increments[:, -1]

    disc_ids = disc_ids * discovery_pres - (1 - discovery_pres)
    new_ids = tf.concat([prop_ids, disc_ids], 1)
    return last_used_id, new_ids


def dynamic_truncate(tensor, t_max):
    """Truncates a tensor v with dynamic shape (T, ...) and
    static shape (None, ...) to (:t_max, ...) and preserves its
    static shape (useful if v is input to dynamic_rnn).

    :param tensor: tensor with shape [T, ...], where T can be unknown.
    :param t_max: scalar int32 tensor
    :return: truncated tensor
    """

    static_shape = tensor.shape.as_list()
    if static_shape[0] is not None and isinstance(t_max, tf.Tensor):
        static_shape[0] = None

    tensor = tensor[:t_max]
    tensor.set_shape(static_shape)

    return tensor
