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

"""Common flags used by moodel configurations.
"""

from attrdict import AttrDict

from sqair import tf_flags as flags


flags.DEFINE_float('transform_var_bias', -3., 'Bias added to the the variance logit of Gaussian `where` distributions.')

flags.DEFINE_float('output_scale', .25, 'It\'s used to scale the output mean of the glimpse decoder.')
flags.DEFINE_string('scale_prior', '-2', 'A single float or four comma-separated floats representing the mean of the '
                                         'Gaussian prior for scale logit.')
flags.DEFINE_integer('glimpse_size', 20, 'Glimpse size.')

flags.DEFINE_float('prop_prior_step_bias', 10., '')
flags.DEFINE_string('prop_prior_type', 'rnn', 'Choose from {rnn, rw_rnn} for a recurrent prior and a random-walk '
                                              'recurrent prior.')
flags.DEFINE_boolean('masked_glimpse', True, 'Masks glimpses based on what_tm1 in propagation if True')


flags.DEFINE_integer('k_particles', 5, 'Number of particles used for the IWAE bound computation')
flags.DEFINE_integer('n_steps_per_image', 3, 'Number of inference steps per frame.')

flags.DEFINE_string('transition', 'VanillaRNN', 'RNNCore from Sonnet to use in discovery and propagation cores.')
flags.DEFINE_string('time_transition', 'GRU', 'RNNCore used for temporal rnn in propagation core.')
flags.DEFINE_string('prior_transition', 'GRU', 'RNNCore used by the propagation prior.')

flags.DEFINE_float('output_std', .3, 'Standard deviation of Gaussian p(x|z)')

flags.DEFINE_integer('n_units', 8, 'Determines the size of the number; each unit is 32 neurons; 8 means 8x32=256 '
                                   'neurons in every fully-connected layer.')
flags.DEFINE_integer('n_what', 50, 'Dimensionality of `what` variables.')


def get_params():
    F = flags.FLAGS

    params = AttrDict(
        glimpse_size=[F.glimpse_size] * 2,
        n_hidden=32 * F.n_units,
        n_layers=2,
    )

    params.n_hiddens = [params.n_hidden] * params.n_layers,
    params.steps_pred_hidden = [params.n_hidden / 2],

    return params
