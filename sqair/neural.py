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

"""Implementations of neural net modules.
"""
import abc
import tensorflow as tf
from tensorflow.python.util import nest
import sonnet as snt

from functools import partial


class Nonlinear(snt.Linear):
    """Layer implementing an affine non-linear transformation.
    """

    def __init__(self, n_output, transfer=tf.nn.elu, initializers=None):

        super(Nonlinear, self).__init__(output_size=n_output, initializers=initializers)
        self._transfer = transfer

    def _build(self, inpt):
        output = super(Nonlinear, self)._build(inpt)
        if self._transfer is not None:
            output = self._transfer(output)
        return output


class FeedForwardNet(snt.AbstractModule):
    """Abstract class for any feed-forward type neural nets.
    """

    def __init__(self, n_hiddens, hidden_transfer=tf.nn.elu, n_out=None, transfer=None,
                 initializers=None, output_initializers=None, name=None):
        """

        :param n_hiddens: int or an interable of ints, number of hidden units in layers
        :param hidden_transfer: callable or iterable; a transfer function for hidden layers or an interable thereof.
            If it's an iterable its length should be the same as length of `n_hiddens`
        :param n_out: int or None, number of output units
        :param transfer: callable or None, a transfer function for the output
        """

        super(FeedForwardNet, self).__init__(name=name)
        self._n_hiddens = nest.flatten(n_hiddens)
        transfers = nest.flatten(hidden_transfer)
        if len(transfers) > 1:
            assert len(transfers) == len(self._n_hiddens)
        else:
            transfers *= len(self._n_hiddens)
        self._hidden_transfers = nest.flatten(transfers)
        self._n_out = n_out
        self._transfer = transfer
        self._initializers = initializers

        if output_initializers is None:
            output_initializers = initializers
        self._output_initializers = output_initializers

    @property
    def output_size(self):
        if self._n_out is not None:
            return self._n_out
        return self._n_hiddens[-1]

    @abc.abstractmethod
    def _create_layer(self, n_hidden, transfer, init, n):
        """Creates a layer.

        :param n_hidden: int, number of hidden units
        :param transfer: callable, an activation function
        :param init: dictionary of callable initializers.
        :param n: int, layer number; -1 mean output layer (not hidden)
        :return: callable
        """
        raise NotImplementedError

    def _build(self, inpt):
            layers = []
            for n, (n_hidden, hidden_transfer) in enumerate(zip(self._n_hiddens, self._hidden_transfers)):
                layers.append(self._create_layer(n_hidden, hidden_transfer, self._initializers, n=n))

            if self._n_out is not None:
                layers.append(self._create_layer(self._n_out, self._transfer, self._output_initializers, n=-1))

            module = snt.Sequential(layers)
            return module(inpt)


class MLP(FeedForwardNet):
    """Implements a multi-layer perceptron.
    """

    def _create_layer(self, n_hidden, transfer, init, n):
        return Nonlinear(n_hidden, transfer, init)


class ConvNet(FeedForwardNet):
    """Implements a ConvNet.
    """

    def __init__(self, kernel_shape, n_hiddens, hidden_transfer=tf.nn.elu, n_out=None, transfer=None,
                 stride=1, rate=1, initializers=None, output_initializers=None, batch_norm=False,
                 is_training=True, name=None):

        FeedForwardNet.__init__(self, n_hiddens, hidden_transfer, n_out, transfer,
                 initializers, output_initializers, name)

        self._kernel_shape = kernel_shape
        self._stride = stride
        self._rate = rate
        self._batch_norm = batch_norm
        self._is_training = is_training

    def _create_layer(self, n_hidden, transfer, init, n, stride=None, rate=None):

        if stride is None:
            stride = self._choose_layer_param(self._stride, n)

        if rate is None:
            rate = self._choose_layer_param(self._rate, n)

        use_batch_norm = n != -1 and self._batch_norm

        conv = snt.Conv2D(output_channels=n_hidden,
                          kernel_shape=self._kernel_shape,
                          stride=stride,
                          rate=rate,
                          initializers=init,
                          use_bias=not use_batch_norm
                          )

        modules = [conv]

        # Don't apply batch norm to the output layer
        if use_batch_norm:
            # TODO: This is super slow when if_training is True; we disable it for now, which means that moving
            #  averages are not updated and batch_norm uses only local batch statistics at train and test time.
            bbn = snt.BatchNorm(offset=True, scale=True, fused=True, update_ops_collection=None)
            bn = partial(bbn, is_training=False)
            modules.append(bn)

        if transfer is not None:
            modules.append(transfer)

        return snt.Sequential(modules) if len(modules) > 1 else modules[0]

    def _choose_layer_param(self, param, n):
        """Chooses parameter for a layer.

        :param param: list of ints or int, parameters
        :param n: layer number
        :return:
        """

        param = nest.flatten(param)
        if len(param) > 1:
            return param[n]

        return param[0]


class UpConvNet(ConvNet):
    """Implements a Subpixel Convolution Net.
    """

    def _create_layer(self, n_hidden, transfer, init, n):

        stride = self._choose_layer_param(self._stride, n)
        area = stride ** 2
        conv = super(UpConvNet, self)._create_layer(n_hidden * area, transfer, init, n, stride=1)

        if stride == 1:
            return conv

        depth_to_space = partial(tf.depth_to_space, block_size=stride)
        return snt.Sequential([conv, depth_to_space])
