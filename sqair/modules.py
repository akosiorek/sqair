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

"""Implementations of different modules used in SQAIR.
"""
import functools
import numpy as np

import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from tensorflow.contrib.resampler import resampler as tf_resampler

import sonnet as snt

from neural import MLP, ConvNet
import ops


class GaussianFromParamVec(snt.AbstractModule):
    """Diagonal Gaussian parametrised by an arbitrary feature vector.

    Mean and variance of the Gaussian are computed as linear projections of the provided feature vector.
    Standard deviation is parametrised by as `tf.nn.softplus(s + scale_offset) + min_std` where `s` is the linear
    projection of the feature vector.
    """

    def __init__(self, n_params, scale_offset=0., min_std=1e-2, *args, **kwargs):
        """Initialises the module.

        :param n_params: Int, dimensionality of the Gaussian.
        :param scale_offset: `Tensor`, offset applied to std before applying softplus.
        :param min_std: `Tensor`, lower bound on standard deviation.
        :param args: args for tfd.Normal
        :param kwargs: kwargs for tfd.Normal
        """
        super(GaussianFromParamVec, self).__init__()
        self._n_dim = n_params
        self._scale_offset = scale_offset
        self._min_std = min_std
        self._make_normal = lambda loc, scale: tfd.Normal(loc, scale, *args, **kwargs)

    def _build(self, inpt):
        if int(inpt.shape[-1] != 2 * self._n_dim):
            transform = snt.Linear(2 * self._n_dim)
            if inpt.shape.ndims > 2:
                transform = snt.BatchApply(transform)

            inpt = transform(inpt)

        loc, scale = tf.split(inpt, 2, -1)

        min_std = tf.get_variable('min_std', initializer=self._min_std, trainable=False)
        scale = tf.nn.softplus(scale + self._scale_offset) + min_std
        return self._make_normal(loc, scale)


class StochasticTransformParam(snt.AbstractModule):
    """Computes mean and std for `where` distribution.

    Very similar to GaussianFromParamVec class but uses an MLP and returns mean and std instead of a
    distribution object.
    """

    def __init__(self, n_hidden, scale_offset=-2.):
        super(StochasticTransformParam, self).__init__()
        self._n_hidden = n_hidden
        self._scale_offset = scale_offset

    def _build(self, inpt):

        flatten = snt.BatchFlatten()
        mlp = MLP(self._n_hidden, n_out=8)
        seq = snt.Sequential([flatten, mlp])
        params = seq(inpt)

        scale_offset = tf.get_variable('scale_offset', initializer=self._scale_offset)
        return params[..., :4], params[..., 4:] + scale_offset


class Encoder(snt.AbstractModule):
    """MLP that can take feature-map sized inputs.
    """

    def __init__(self, n_hidden):
        super(Encoder, self).__init__()
        self._n_hidden = n_hidden

    def _build(self, inpt):
        flat = snt.BatchFlatten()
        mlp = MLP(self._n_hidden)
        seq = snt.Sequential([flat, mlp])
        return seq(inpt)


class ConvEncoder(snt.AbstractModule):
    """ConvNet with flattened outputs.
    """

    def __init__(self, n_hidden, kernel_shape, strides):
        super(ConvEncoder, self).__init__()
        self._n_hidden = n_hidden
        self._kernel_shape = kernel_shape
        self._strides = strides

    def _build(self, inpt):

        conv = ConvNet(self._kernel_shape, self.n_hidden, stride=self._strides)
        return snt.Sequential([conv, snt.BatchFlatten])


class Decoder(snt.AbstractModule):
    """MLP decoder that reshapes the output into a feature map.
    """

    def __init__(self, n_hidden, output_size, output_scale=.25):
        super(Decoder, self).__init__()
        self._n_hidden = n_hidden
        self._output_size = output_size
        self._output_scale = output_scale

    def _build(self, inpt):
        n = np.prod(self._output_size)

        mlp = MLP(self._n_hidden, n_out=n)
        reshape = snt.BatchReshape(self._output_size)
        seq = snt.Sequential([mlp, reshape])
        return seq(inpt) * tf.get_variable('output_scale', initializer=self._output_scale)


class SpatialTransformer(snt.AbstractModule):
    """Spatial Transformer module.
    """

    def __init__(self, img_size, crop_size, inverse=False):
        """Initialises the module.

        :param img_size: Tuple of ints, size of the input image.
        :param crop_size: Tuple of ints, size of the resampled image.
        :param inverse: Boolean; inverts the given transformation if True and then maps crops into full-sized images.
        """

        super(SpatialTransformer, self).__init__()

        with self._enter_variable_scope():
            constraints = snt.AffineWarpConstraints.no_shear_2d()
            self._warper = snt.AffineGridWarper(img_size[:2], crop_size, constraints)
            if inverse:
                self._warper = self._warper.inverse()

    def _sample_image(self, img, transform_params):
        grid_coords = self._warper(transform_params)
        return tf_resampler(img, grid_coords)

    def _build(self, img, sentinel=None, coords=None, logits=None):
        """Applies the transformation.

        Either coords or logits must be given, but not both. They have to be passed as kwarg. Assume that
        they have the shape of (..., n_params) where n_params = n_scales + n_shifts
        and:
            scale = transform_params[..., :n_scales]
            shift = transform_params[..., n_scales:n_scales+n_shifts]



        :param img: Tensor of images of shape `[B, H, W, C]`.
        :param sentinel: Unused; it is to guard `coords` or `logits` from being passed as positional arguments.
        :param coords: Tensor of coordinates in Spatial Transformer space.
        :param logits: Tensor of logits; they are converted to ST space.
        :return: Tensor of transformed images.
        """

        if sentinel is not None:
            raise ValueError('Either coords or logits must be given by kwargs!')

        if coords is not None and logits is not None:
            raise ValueError('Please give eithe coords or logits, not both!')

        if coords is None and logits is None:
            raise ValueError('Please give coords or logits!')

        if coords is None:
            coords = self.to_coords(logits)

        axis = coords.shape.ndims - 1
        sx, sy, tx, ty = tf.split(coords, 4, axis=axis)
        sx, sy = (ops.clip_preserve(s, 1e-4, s) for s in (sx, sy))

        transform_params = tf.concat([sx, tx, sy, ty], -1)

        if len(img.get_shape()) == 3:
            img = img[..., tf.newaxis]

        if len(transform_params.get_shape()) == 2:
            return self._sample_image(img, transform_params)
        else:
            transform_params = tf.unstack(transform_params, axis=1)
            samples = [self._sample_image(img, tp) for tp in transform_params]
            return tf.stack(samples, axis=1)

    @staticmethod
    def to_coords(logits):
        scale_logit, shift_logit = tf.split(logits, 2, -1)

        scale = tf.nn.sigmoid(scale_logit)
        shift = tf.nn.tanh(shift_logit)
        coords = tf.concat((scale, shift), -1)
        return coords

    @staticmethod
    def to_logits(coords, eps=1e-4):
        coords = tf.convert_to_tensor(coords)
        scale, shift = tf.split(coords, 2, -1)

        # scale = tf.nn.sigmoid(scale)
        scale_logit = tf.clip_by_value(scale, eps, 1. - eps)
        scale_logit = tf.log(scale_logit / (1. - scale_logit))

        # shift = tf.nn.tanh(shift)
        shift_logit = tf.clip_by_value(shift, eps - 1., 1. - eps)
        shift_logit = 0.5 * (tf.log(1. + shift_logit) - tf.log(1. - shift_logit))

        logits = tf.concat((scale_logit, shift_logit), -1)
        return logits

    @staticmethod
    def stn_to_pixel_coord(scale, translation, length):
        size = (length + 1.) * scale
        shift = 0.5 * (length - 1.) * (translation - scale + 1.)
        return shift, size

    @staticmethod
    def stn_to_pixel_coords(stn_coords, img_size):

        if not isinstance(stn_coords, tf.Tensor):
            stn_coords = np.asarray(stn_coords)

        sx, sy, tx, ty = ops.split(stn_coords, 4, axis=-1)
        y, h = SpatialTransformer.stn_to_pixel_coord(sy, ty, img_size[0])
        x, w = SpatialTransformer.stn_to_pixel_coord(sx, tx, img_size[1])

        coords = ops.concat((y, x, h, w), -1)
        return coords

    @staticmethod
    def pixel_to_stn_coords(yxhw, img_size):

        img_size = np.asarray(img_size).astype(np.float32)
        if not isinstance(yxhw, tf.Tensor):
            yxhw = np.asarray(yxhw).astype(np.float32)

        while len(img_size.shape) < len(yxhw.shape):
            img_size = img_size[np.newaxis, ...]

        scale = yxhw[..., 2:] / (img_size + 1)
        shift = 2 * yxhw[..., :2] / (img_size - 1.) + scale - 1.

        sy, sx = ops.split(scale, 2, -1)
        ty, tx = ops.split(shift, 2, -1)
        stn_coords = ops.concat((sx, sy, tx, ty), -1)
        return stn_coords


class AIRGlimpse(snt.AbstractModule):
    """Abstract class for AIR decoders and encoders.
    """

    def __init__(self, img_size, glimpse_size, inverse):
        """Initialises the module.

        :param img_size: Tuple of ints.
        :param crop_size: Tuple of ints.
        :param inverse: Boolean; see SpatialTransformer for details.
        """

        super(AIRGlimpse, self).__init__()
        self._glimpse_size = glimpse_size
        self._transformer = SpatialTransformer(img_size, glimpse_size, inverse=inverse)

    def to_coords(self, where_logits):
        return self._transformer.to_coords(where_logits)

    def to_logits(self, where):
        return self._transformer.to_logits(where)


class AIREncoder(AIRGlimpse):
    """Extracts and stochastically encodes glimpses."""

    def __init__(self, img_size, glimpse_size, n_what, glimpse_encoder, scale_offset=0.,
                 masked_glimpse=False, debug=False):

        super(AIREncoder, self).__init__(img_size, glimpse_size, inverse=False)
        self.n_what = n_what
        self._masked_glimpse = masked_glimpse

        with self._enter_variable_scope():
            self._glimpse_encoder = glimpse_encoder
            self._what_distrib = GaussianFromParamVec(n_what,
                                                      scale_offset=scale_offset,
                                                      validate_args=debug, allow_nan_stats=not debug)

            if self._masked_glimpse:
                self._mask_mlp = MLP(128, n_out=np.prod(glimpse_size), transfer=tf.nn.sigmoid,
                                     output_initializers={'b': tf.constant_initializer(1.)})

    def _build(self, img, where=None, mask_inpt=None):
        """Extracts and encodes glimpses.s

        :param img: img tensor.
        :param where: where logits.
        :param mask_inpt: if not None, this input is used to compute soft-attention mask over the extracted glimpse
            before encoding it.
        :return: what_distrib, glimpses.
        """

        if where is not None:
            coords = self.to_coords(where)

            if coords.shape.ndims == 3:
                coords = tf.unstack(coords, axis=-2)
                glimpse = [self._transformer(img, coords=c) for c in coords]
            else:
                glimpse = [self._transformer(img, coords=coords)]

            glimpse = tf.stack(glimpse, 1)

        else:
            glimpse = img

        if self._masked_glimpse and mask_inpt is not None:
            if mask_inpt.shape.ndims == 2:
                mask_inpt = tf.expand_dims(mask_inpt, 1)

            glimpse_mask = snt.BatchApply(self._mask_mlp)(mask_inpt)
            glimpse_mask = tf.reshape(glimpse_mask, tf.shape(glimpse)[:-1])
            glimpse *= tf.expand_dims(glimpse_mask, -1)

        what_params = snt.BatchApply(self._glimpse_encoder)(glimpse)

        if what_params.shape[1] == 1:
            what_params = tf.squeeze(what_params, 1)

        what_distrib = self._what_distrib(what_params)
        return what_distrib, glimpse


class AIRDecoder(AIRGlimpse):
    """Decoder for AIR and SQAIR.
    """

    def __init__(self, img_size, glimpse_size, glimpse_decoder, batch_dims=2,
                 mean_img=None, output_std=0.3, learn_std=False, bg_std=None,
                 learn_bg_std=False, min_std=0., bg_bigger_than_fg_std=False):
        """Initialises the module.

        :param img_size: Tuple of ints, size of the output image.
        :param glimpse_size: Tuple of ints, size of the reconstructed glimpse.
        :param glimpse_decoder: Callable returning a glimpse given a feature vector, e.g. Decoder object.
        :param batch_dims: Int, number of batch dimensions. AIRDecoder flattens initial `batch_dims` dimensions of
            a batch before processing, similarly to `snt.BatchApply`.
        :param mean_img: `Tensor` or None; if not None, it is added to the recreated image (potentially only at the
            locations of reconstructed glimpses).
        :param output_std: `Tensor` or float, output std of a Gaussian.
        :param learn_std: Boolean; learns output std if True and initialises it with `output_std`.
        :param bg_std: `Tensor` or float. If not None, AIRDecoder uses different output std for pixels belonging to
            the background and to the glimpses.
        :param learn_bg_std: Boolean; learns background std if True.
        :param min_std: Float, lower bound on output std.
        :param bg_bigger_than_fg_std: Boolean, constraints background std to be >= glimpse std if True.
        """

        super(AIRDecoder, self).__init__(img_size, glimpse_size, inverse=True)
        self._batch_dims = batch_dims
        self._mean_img = mean_img

        with self._enter_variable_scope():
            self._glimpse_decoder = glimpse_decoder(glimpse_size)
            if self._mean_img is not None:
                self._mean_img = tf.Variable(self._mean_img, dtype=tf.float32, trainable=True)
                self._mean_img = tf.expand_dims(self._mean_img, 0)

            self._batch = functools.partial(snt.BatchApply, n_dims=self._batch_dims)

            # TODO: verify that the model still works.
            # Handle lower-bounding and learnability of output standard deviations
            if bg_std is None:
                bg_std = output_std

            tuples = 'output background'.split(), (output_std, bg_std), (learn_std, learn_bg_std)
            for name, value, is_learnable in zip(*tuples):

                name += '_std'

                if min_std != 0.:
                    assert (0. < min_std <= value)
                    offset = 2 * value * min_std - min_std ** 2  # offset is now the lower bound on std
                    value -= min_std

                value = np.sqrt(value)
                value = tf.get_variable(name, shape=[], dtype=tf.float32, initializer=tf.constant_initializer(value),
                                            trainable=is_learnable)
                value **= 2.
                if min_std != 0.:
                    value += offset

                setattr(self, '_' + name, value)

            # constrain bg std to be a little bit bigger than foreground std
            if bg_bigger_than_fg_std:
                self._background_std = tf.maximum(self._background_std, self._output_std + 1e-4)

            tf.summary.scalar('output_std', self._output_std)
            tf.summary.scalar('background_std', self._background_std)

    def _decode(self, glimpse, presence=None, where=None):
        inversed = glimpse

        if where is not None:
            coords = self.to_coords(where)
            inversed = self._batch(self._transformer)(glimpse, coords=coords)

        if presence is not None:
            inversed *= presence[..., tf.newaxis, tf.newaxis]

        return tf.reduce_sum(inversed, axis=-4)

    def _build(self, what, where=None, presence=None):

        glimpse = self._batch(self._glimpse_decoder)(what)
        canvas = self._decode(glimpse, presence, where)
        canvas, written_to_mask = self._add_mean_image(canvas, presence, where)

        output_std = written_to_mask * self._output_std + (1. - written_to_mask) * self._background_std
        pdf = tfd.Normal(canvas, output_std)

        return pdf, glimpse

    def _add_mean_image(self, canvas, presence, where):

        ones = tf.ones(where.shape.as_list()[:2] + list(self._glimpse_size))
        non_zero_mask = self._decode(ones, presence, where)
        non_zero_mask = tf.nn.sigmoid(-10. + non_zero_mask * 20.)

        if self._mean_img is not None:
                canvas += self._mean_img * non_zero_mask

        return canvas, non_zero_mask


class StepsPredictor(snt.AbstractModule):
    """Computes the probability logit for discovering or propagating an object."""

    def __init__(self, n_hidden, steps_bias=0., max_rel_logit_change=np.inf, max_logit_change=np.inf, **kwargs):
        """

        :param n_hidden:
        :param steps_bias:
        :param max_rel_logit_change: float; maximum relative logit change since the previous time-step
        :param kwargs:
        """
        super(StepsPredictor, self).__init__()
        self._n_hidden = n_hidden
        self._steps_bias = steps_bias
        self._max_rel_logit_change = max_rel_logit_change
        self._bernoulli = lambda logits: tfd.Bernoulli(logits=logits, dtype=tf.float32, **kwargs)

        with self._enter_variable_scope():

            if max_logit_change != np.inf and max_rel_logit_change != np.inf:
                raise ValueError('Only one of max_logit_change and max_rel_logit_change can be used!')

            if max_rel_logit_change != np.inf:
                max_rel_logit_change = tf.get_variable('max_rel_logit_change',
                                                       shape=[],
                                                       initializer=tf.constant_initializer(max_rel_logit_change),
                                                       trainable=False)
            self._max_rel_logit_change = max_rel_logit_change

            if max_logit_change != np.inf:
                max_logit_change = tf.get_variable('max_logit_change',
                                                       shape=[],
                                                       initializer=tf.constant_initializer(max_logit_change),
                                                       trainable=False)
            self._max_logit_change = max_logit_change

    def _build(self, previous_presence, previois_logit, *features):

        init = {'b': tf.constant_initializer(self._steps_bias)}
        mlp = MLP(self._n_hidden, n_out=1, output_initializers=init)

        features = ops.maybe_concat(features)
        logit = mlp(features)
        logit = previous_presence * logit + (previous_presence - 1.) * 88.

        if previois_logit is not None:
            if self._max_rel_logit_change != np.inf:
                min_logit = (1. - self._max_rel_logit_change) * previois_logit
                max_logit = (1. + self._max_rel_logit_change) * previois_logit
                logit = tf.clip_by_value(logit, min_logit, max_logit)

            elif self._max_logit_change != np.inf:
                logit = previois_logit + self._max_logit_change * tf.nn.tanh(logit)

        return self._bernoulli(logit)


class AffineDiagNormal(snt.AbstractModule):
    """Gaussian with traingular covariance matrix, whose rows are rescaled by the `scale` argument.
    """

    def __init__(self, *args, **kwargs):
        super(AffineDiagNormal, self).__init__()
        self._factory = lambda loc, scale_tril: tfd.MultivariateNormalTriL(loc, scale_tril, *args, **kwargs)

    def _build(self, loc, scale):

        dims = int(loc.shape[-1])
        scale_tril = tfd.fill_triangular(tf.get_variable(shape=[dims * (dims + 1) / 2], dtype=tf.float32,
                                                         name="cholesky_scale"))

        expanded_scale = tf.expand_dims(scale, -1)
        scale_tril = tf.expand_dims(scale_tril, 0)

        batch_scale_tril = scale_tril * expanded_scale + tf.matrix_diag(scale)
        return self._factory(loc, batch_scale_tril)


class RecurrentNormalImpl(snt.AbstractModule):
    """Computational module for the RecurrentNormal.
    """
    _cond_state = None

    def __init__(self, n_dim, n_hidden, conditional=False, output_initializers=None):
        super(RecurrentNormalImpl, self).__init__()
        self._n_dim = n_dim
        self._n_hidden = n_hidden

        with self._enter_variable_scope():
            self._rnn = snt.VanillaRNN(self._n_dim)
            self._readout = snt.Linear(self._n_dim * 2, initializers=output_initializers)
            self._init_state = self._rnn.initial_state(1, trainable=True)
            self._init_sample = tf.get_variable('init_sample', shape=(1, self._n_dim), trainable=True)

            if conditional:
                self._cond_state = snt.Sequential([snt.Linear(self._n_hidden), tf.nn.elu])

    def _build(self, batch_size=1, seq_len=1, override_samples=None, conditioning=None):

        s = self._init_sample, self._init_state
        sample, state = (tf.tile(ss, (batch_size, 1)) for ss in s)

        if conditioning is not None:
            assert self._cond_state is not None, 'This distribution is unconditional. Pass conditional=True in init!'
            state = tf.concat((state, conditioning), -1)
            state = self._cond_state(state)

        outputs = [[] for _ in xrange(4)]
        if override_samples is not None:
            override_samples = tf.unstack(override_samples, axis=-2)
            seq_len = len(override_samples)

        for i in xrange(seq_len):

            if override_samples is None:
                override_sample = None
            else:
                override_sample = override_samples[i]

            results = self._forward(sample, state, override_sample)
            sample = results[0]

            for res, output in zip(results, outputs):
                output.append(res)

        return [tf.stack(o, axis=-2) for o in outputs]

    def _forward(self, sample_m1, hidden_state, sample=None):
        output, state = self._rnn(sample_m1, hidden_state)
        stats = self._readout(output)
        loc, scale = tf.split(stats, 2, -1)
        scale = tf.nn.softplus(scale) + 1e-2
        pdf = tfd.Normal(loc, scale)

        if sample is None:
            sample = pdf.sample()

        return sample, loc, scale, pdf.log_prob(sample)


class RecurrentNormal(object):
    """Autoregressive Normal Distribution.
    """

    def __init__(self, n_dim, n_hidden, conditional=False, output_initializers=None):
        self._impl = RecurrentNormalImpl(n_dim, n_hidden, conditional, output_initializers)

    def log_prob(self, samples, conditioning=None):
        batch_size = samples.shape.as_list()[0]
        _, _, _, logprob = self._impl(batch_size=batch_size, override_samples=samples, conditioning=conditioning)
        return logprob

    def sample(self, sample_size=(1, 1), conditioning=None):
        """

        :param sample_size: tuple of (num samples, seq_length)
        :return:
        """
        sample_size, length = sample_size
        samples, _, _, _ = self._impl(batch_size=sample_size, seq_len=length, conditioning=conditioning)
        return samples


class ConditionedNormalAdaptor(tfd.Normal):

    def log_prob(self, *args, **kwargs):
        if 'conditioning' in kwargs:
            del kwargs['conditioning']

        return super(ConditionedNormalAdaptor, self).log_prob(*args, **kwargs)

    def sample(self, *args, **kwargs):
        if 'conditioning' in kwargs:
            del kwargs['conditioning']

        return super(ConditionedNormalAdaptor, self).sample(*args, **kwargs)