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

"""Functions to dela with nested tensors.

Partially adapted from https://github.com/tensorflow/models/blob/master/research/fivo.
"""
import tensorflow as tf
from tensorflow.python.util import nest


def map_nested(map_fn, nested):
    """Executes map_fn on every element in a (potentially) nested structure.
    Args:
      map_fn: A callable to execute on each element in 'nested'.
      nested: A potentially nested combination of sequence objects. Sequence
        objects include tuples, lists, namedtuples, and all subclasses of
        collections.Sequence except strings. See nest.is_sequence for details.
        For example [1, ('hello', 4.3)] is a nested structure containing elements
        1, 'hello', and 4.3.
    Returns:
      out_structure: A potentially nested combination of sequence objects with the
        same structure as the 'nested' input argument. out_structure
        contains the result of applying map_fn to each element in 'nested'. For
        example map_nested(lambda x: x+1, [1, (3, 4.3)]) returns [2, (4, 5.3)].
    """
    out = map(map_fn, nest.flatten(nested))
    return nest.pack_sequence_as(nested, out)


def tile_tensors(tensors, multiples):
    """Tiles a set of Tensors.

    Args:
      tensors: A potentially nested tuple or list of Tensors with rank
        greater than or equal to the length of 'multiples'. The Tensors do not
        need to have the same rank, but their rank must not be dynamic.
      multiples: A python list of ints indicating how to tile each Tensor
        in 'tensors'. Similar to the 'multiples' argument to tf.tile.
    Returns:
      tiled_tensors: A potentially nested tuple or list of Tensors with the same
        structure as the 'tensors' input argument. Contains the result of
        applying tf.tile to each Tensor in 'tensors'. When the rank of a Tensor
        in 'tensors' is greater than the length of multiples, multiples is padded
        at the end with 1s. For example when tiling a 4-dimensional Tensor with
        multiples [3, 4], multiples would be padded to [3, 4, 1, 1] before tiling.
    """

    def tile_fn(x):
        return tf.tile(x, list(multiples) + [1] * (x.shape.ndims - len(multiples)))

    return map_nested(tile_fn, tensors)


def tile_along_newaxis(tensors, n_tiles, axis):
    assert axis >= 0, 'Supports tensors with different number of dimensions but works only with positive axes'
    tiles = [1] * axis + [n_tiles]
    tensors = map_nested(lambda x: tf.expand_dims(x, axis), tensors)
    return tile_tensors(tensors, tiles)


def combine(tensors, func, *args, **kwargs):
    tensor_with_structure = tensors[0]
    tensors = [nest.flatten(tensor) for tensor in tensors]
    tensors = [func(row, *args, **kwargs) for row in zip(*tensors)]
    return nest.pack_sequence_as(tensor_with_structure, tensors)


def concat(tensors, axis):
    return combine(tensors, tf.concat, axis)


def stack(tensors, axis):
    return combine(tensors, tf.stack, axis)
