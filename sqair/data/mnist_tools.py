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

import numpy as np
import tensorflow as tf
from attrdict import AttrDict

from sqair.data import load_data as _load_data, tensors_from_data as _tensors
from sqair import tf_flags as flags
from sqair.index import dynamic_truncate


flags.DEFINE_integer('seq_len', 0, 'Length of loaded data sequences. If 0, it defaults to the maximum length.')
flags.DEFINE_integer('stage_itr', 0, 'If > 0 it setups a curriculum learning where `seq_len` starts as given and '
                                     'increases by one every `stage_itr` until it gets to the maximum value.')

axes = {'imgs': 1, 'labels': 0, 'nums': 1, 'coords': 1}


def truncate(data_dict, n_timesteps):
    data_dict['imgs'] = data_dict['imgs'][:n_timesteps]
    data_dict['coords'] = data_dict['coords'][:n_timesteps]
    data_dict['nums'] = data_dict['nums'][:n_timesteps]
    return data_dict


def process_data(data, n_timesteps):

    if n_timesteps is not None:
        truncate(data, n_timesteps)

    n_steps = data.nums.shape[-1]
    to_pad = n_steps - data.coords.shape[-2]
    if to_pad > 0:

        shape = list(data.coords.shape)
        shape[-2] = to_pad
        zeros = np.zeros(shape, dtype=data.coords.dtype)
        data.coords = np.concatenate((data.coords, zeros), -2)


def load(batch_size, n_timesteps=None):

    F = flags.FLAGS

    valid_data = _load_data(F.valid_path)
    train_data = _load_data(F.train_path)

    if F.stage_itr == 0 and n_timesteps is None and F.seq_len != 0:
        n_timesteps = F.seq_len

    process_data(valid_data, n_timesteps)
    process_data(train_data, n_timesteps)

    train_tensors = _tensors(train_data, batch_size, axes, shuffle=True)
    valid_tensors = _tensors(valid_data, batch_size, axes, shuffle=False)

    n_timesteps = tf.shape(train_tensors['imgs'])[0]

    if train_data['imgs'].shape[0] != train_data['nums'].shape[0]:
        train_tensors['nums'] = tf.tile(train_tensors['nums'], (n_timesteps, 1, 1))
        valid_tensors['nums'] = tf.tile(valid_tensors['nums'], (n_timesteps, 1, 1))

    if F.seq_len != 0 and F.stage_itr > 0:
        global_step = tf.to_int32(tf.train.get_or_create_global_step())
        stage = global_step // F.stage_itr
        stage_seq_len = tf.minimum(F.seq_len + stage, n_timesteps)
        tf.summary.scalar('seq_len', stage_seq_len)

        for d in (train_tensors, valid_tensors):
            for k, v in d.iteritems():
                d[k] = dynamic_truncate(v, stage_seq_len)

    data_dict = AttrDict(
        train_img=train_tensors['imgs'],
        valid_img=valid_tensors['imgs'],
        train_num=train_tensors['nums'],
        valid_num=valid_tensors['nums'],
        train_coord=train_tensors['coords'],
        valid_coord=valid_tensors['coords'],
        train_tensors=train_tensors,
        valid_tensors=valid_tensors,
        train_data=train_data,
        valid_data=valid_data,
        axes=axes
    )

    return data_dict