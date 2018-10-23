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

"""Config file for MLP-SQAIR for MNIST.
"""
from functools import partial

import numpy as np
import sonnet as snt
import tensorflow as tf

from sqair.common_model_flags import flags, get_params

from sqair.sqair_modules import Propagate, Discover
from sqair.model import Model
from sqair.core import DiscoveryCore, PropagationCore
from sqair.modules import Encoder, StochasticTransformParam, StepsPredictor, Decoder, AIRDecoder, AIREncoder
from sqair.ops import maybe_getattr
from sqair.seq import SequentialAIR
from sqair.propagate import make_prior, SequentialSSM

flags.DEFINE_string('disc_prior_type', 'cat', 'Prior for the number of discovery inference steps; choose from '
                                              '{geom, cat} for geometric and categorical, respectively.')
flags.DEFINE_float('step_success_prob', 0.75, 'Step success prob for the geometric prior for discovery; in [0, 1.]')

flags.DEFINE_float('disc_step_bias', 1., 'Value added to the logit of discovering a new object; positive values'
                                         ' increase the probability.')
flags.DEFINE_float('prop_step_bias', 5., 'Value added to the logit of propagating an existing object; positive values'
                                         ' increase the probability.')

flags.DEFINE_boolean('sample_from_prior', False, 'Samples from the prior instead of q if True.')
flags.DEFINE_boolean('rec_where_prior', True, 'Uses a recurrent prior for where in discovery.')


def parse_string_flag(flag, dtype=np.float32, sep=',', num_elements=-1):
    try:
        values = [dtype(f.strip()) for f in flag.split(sep)]
    except:
        try:
            values = [np.float32(flag)]
        except:
            raise
        

    if len(values) == 1 and num_elements > 1:
        values *= num_elements

    elif num_elements != -1 and len(values) != num_elements:
        raise ValueError('Incorrect number of elements in flag "{}"'.format(flag))

    return values


def load(img, coords, num, mean_img=None, debug=False):
    F = flags.FLAGS

    if img.shape.ndims == 4:
        img = tf.expand_dims(img, -1)
        if mean_img is not None:
            mean_img = mean_img[..., np.newaxis]

    params = get_params()
    shape = img.shape.as_list()
    batch_size, img_size = shape[1], shape[2:]

    rnn_class = maybe_getattr(snt, F.transition)
    time_rnn_class = maybe_getattr(snt, F.time_transition)

    input_encoder = partial(Encoder, params.n_hiddens)

    def glimpse_encoder():
        return AIREncoder(img_size, params.glimpse_size, F.n_what, Encoder(params.n_hiddens),
                          masked_glimpse=F.masked_glimpse, debug=F.debug)

    transform_estimator = partial(StochasticTransformParam, params.n_hiddens, F.transform_var_bias)
    steps_predictor = partial(StepsPredictor, params.steps_pred_hidden, F.disc_step_bias)

    with tf.variable_scope('discovery'):
        discover_cell = DiscoveryCore(img_size, params.glimpse_size, F.n_what, rnn_class(params.n_hidden),
                                      input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                                      debug=debug
                                      )

        discover = Discover(F.n_steps_per_image, discover_cell,
                            step_success_prob=F.step_success_prob,
                            where_mean=parse_string_flag(F.scale_prior, float, num_elements=2) + [0, 0],
                            disc_prior_type=F.disc_prior_type,
                            rec_where_prior=F.rec_where_prior)

    with tf.variable_scope('propagation'):
        # Prop cell should have a different rnn cell but should share all other estimators
        input_encoder = lambda: discover_cell._input_encoder
        glimpse_encoder = lambda: discover_cell._glimpse_encoder
        transform_estimator = partial(StochasticTransformParam, params.n_hiddens, F.transform_var_bias)
        steps_predictor = partial(StepsPredictor, params.steps_pred_hidden, F.prop_step_bias)

        # Prop cell should have a different rnn cell but should share all other estimators
        propagate_rnn_cell = rnn_class(params.n_hidden)
        temporal_rnn_cell = time_rnn_class(params.n_hidden)
        propagation_cell = PropagationCore(img_size, params.glimpse_size, F.n_what, propagate_rnn_cell,
                                           input_encoder, glimpse_encoder, transform_estimator,
                                           steps_predictor, temporal_rnn_cell,
                                           debug=debug)

        prior_rnn = maybe_getattr(snt, F.prior_transition)(params.n_hidden)
        propagation_prior = make_prior(F.prop_prior_type, F.n_what, prior_rnn, F.prop_prior_step_bias)

        ssm = SequentialSSM(propagation_cell)
        propagate = Propagate(ssm, propagation_prior)

    with tf.variable_scope('decoder'):
        glimpse_decoder = partial(Decoder, params.n_hiddens, output_scale=F.output_scale)
        decoder = AIRDecoder(img_size, params.glimpse_size, glimpse_decoder,
                             batch_dims=2,
                             mean_img=mean_img,
                             output_std=F.output_std,
                             )

    with tf.variable_scope('sequence'):
        time_cell = maybe_getattr(snt, F.time_transition)
        if time_cell is not None:
            time_cell = time_cell(params.n_hidden)

        sequence_apdr = SequentialAIR(F.n_steps_per_image, params.glimpse_size, discover, propagate, time_cell, decoder,
                                      sample_from_prior=F.sample_from_prior)

    with tf.variable_scope('model'):
        model = Model(img, coords, sequence_apdr, F.k_particles, num, debug)

    return model
