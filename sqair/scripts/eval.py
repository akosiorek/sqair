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

"""Evaluation script for SQAIR.
"""
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

import sys
sys.path.append('../')

import os
from os import path as osp

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

from sqair.experiment_tools import load, get_session, parse_flags, assert_all_flags_parsed, _load_flags, FLAG_FILE, json_load, _restore_flags
from sqair import tf_flags as flags

flags.DEFINE_string('data_config', 'configs/seq_mnist_data.py', '')
flags.DEFINE_string('model_config', 'configs/apdr.py', '')
flags.DEFINE_string('checkpoint_dir', '../checkpoints', '')

flags.DEFINE_integer('batch_size', 5, '')

flags.DEFINE_integer('every_nth_checkpoint', 1, 'takes 1 in nth checkpoints to evaluate; takes only the last checkpoint if -1')
flags.DEFINE_integer('from_itr', 0, 'Evaluates only checkpoints with training iteration greater than `from_itr`')

flags.DEFINE_string('dataset', 'valid', 'test or valid')

flags.DEFINE_boolean('logp', True, '')
flags.DEFINE_boolean('vae', True, '')
flags.DEFINE_boolean('num_step_acc', True, '')
flags.DEFINE_boolean('rec', True, '')
flags.DEFINE_boolean('kl', True, '')

flags.DEFINE_boolean('resume', False, 'Tries to resume if True. Throws an error if False and any of the log files exist'
                                      ' unless F.overwrite is True')

flags.DEFINE_boolean('overwrite', False, '')

flags.DEFINE_string('gpu', '0', 'Id of the gpu to allocate')
flags.DEFINE_boolean('debug', False, 'Adds a lot of summaries if True')


F = flags.FLAGS
os.environ['CUDA_VISIBLE_DEVICES'] = F.gpu

if __name__ == '__main__':

    _load_flags(F.model_config, F.data_config)
    flags = parse_flags()
    assert_all_flags_parsed()

    flag_path = os.path.join(F.checkpoint_dir, FLAG_FILE)
    restored_flags = json_load(flag_path)
    flags.update(restored_flags)
    _restore_flags(flags)

    print 'Processing:', F.checkpoint_dir
    checkpoint_state = tf.train.get_checkpoint_state(F.checkpoint_dir)
    if checkpoint_state is None:
        print 'No checkpoints found in {}'.format(F.checkpoint_dir)

    checkpoint_paths = checkpoint_state.all_model_checkpoint_paths

    if F.from_itr > 0:
        itrs = [(int(p.split('-')[-1]), p) for p in checkpoint_paths]
        itrs = sorted(itrs, key=lambda x: x[0])

        for i, (itr, _) in enumerate(itrs):
            if itr >= F.from_itr:
                break

        itrs = itrs[i:]
        checkpoint_paths = [i[1] for i in itrs]

    last_checkpoint = checkpoint_paths[-1]

    if F.every_nth_checkpoint >= 0:
        checkpoint_paths = checkpoint_paths[::F.every_nth_checkpoint]
        if checkpoint_paths[-1] != last_checkpoint:
            checkpoint_paths.append(last_checkpoint)
    elif F.every_nth_checkpoint == -1:
        checkpoint_paths = [last_checkpoint]
    else:
        raise ValueError('every_nth_checkpoint has an invalid value of {}'.format(F.every_nth_checkpoint))

    tf.reset_default_graph()

    data_dict = load(F.data_config, F.batch_size)

    # mean img
    imgs = data_dict.train_data.imgs
    mean_img = imgs.mean(tuple(range(len(imgs.shape) - 2)))
    assert len(mean_img.shape) == 2

    try:
        coords = data_dict.train_coord
    except AttributeError:
        coords = None

    if F.dataset == 'train':
        n_batches = data_dict.train_data.imgs.shape[1] // F.batch_size
        img, num, coords = data_dict.train_img, data_dict.train_num, data_dict.train_coord
    else:
        n_batches = data_dict.valid_data.imgs.shape[1] // F.batch_size
        img, num, coords = data_dict.valid_img, data_dict.valid_num, data_dict.valid_coord

    model = load(F.model_config, img=img, coords=coords, num=num, mean_img=mean_img)

    saver = tf.train.Saver()
    sess = get_session()
    sess.run(tf.global_variables_initializer())

    evaluated_checkpoints = set()

    files = {}
    estimates = {}
    tensors = {}

    def check_logfile(tag, path, init_value, tensor):
        global evaluated_checkpoints
        global files
        global estimates
        global values
        global tensors

        if os.path.exists(path):
            if not F.resume and not F.overwrite:
                raise RuntimeError('Log file {} exists!'.format(path))
            elif F.resume:
                results = np.loadtxt(path, delimiter=': ')
                iters = set(results[:, 0])
                evaluated_checkpoints = evaluated_checkpoints.union(iters)
            elif F.overwrite:
                os.rmdir(path)

        files[tag] = path
        estimates[tag] = init_value
        tensors[tag] = tensor

    if F.logp:
        log_p_x_file = os.path.join(F.checkpoint_dir, 'logpx_{}.txt'.format(F.dataset))
        check_logfile('logp', log_p_x_file, 0., model.elbo_iwae)

    if F.vae:
        vae_file = os.path.join(F.checkpoint_dir, 'vae_{}.txt'.format(F.dataset))
        check_logfile('vae', vae_file, 0., model.elbo_vae)

    if F.num_step_acc:
        num_step_acc_file = os.path.join(F.checkpoint_dir, 'num_step_acc_{}.txt'.format(F.dataset))
        check_logfile('num_step_accuracy', num_step_acc_file, 0., model.num_step_accuracy)

    if F.rec:
        rec_file = os.path.join(F.checkpoint_dir, 'rec_{}.txt'.format(F.dataset))
        check_logfile('rec', rec_file, 0., model.data_ll)

    if F.kl:
        kl_file = os.path.join(F.checkpoint_dir, 'kl_{}.txt'.format(F.dataset))
        check_logfile('kl', kl_file, 0., model.kl)

    for checkpoint_path in checkpoint_paths:
        n_itr = int(checkpoint_path.split('-')[-1])

        if n_itr in evaluated_checkpoints:
            print 'Skipping checkpoint:', n_itr
            continue

        print 'Processing checkpoint:', n_itr,
        sys.stdout.flush()

        saver.restore(sess, checkpoint_path)

        log_p_x_estimate = 0.
        vae_estimate = 0.
        num_step_acc_estimate = 0.
        rec_estimate = 0.
        kl_estimate = 0.

        start = time.time()
        print 'num_batches', n_batches
        for batch_num in xrange(n_batches):
            print 'batch_num', batch_num
            values = sess.run(tensors)
            for k, v in values.iteritems():
                estimates[k] += v

        for k, v in estimates.iteritems():
            estimates[k] = v / n_batches

            with open(files[k], 'a') as f:
                f.write('{}: {}\n'.format(n_itr, estimates[k]))

        duration = time.time() - start
        print 'took {}s'.format(duration)