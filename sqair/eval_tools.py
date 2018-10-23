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

"""Tools used for model evaluation.
"""
import collections
import os.path as osp
import time
import itertools

import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from attrdict import AttrDict

from modules import SpatialTransformer


bbox_colors = """
    #a6cee3
    #1f78b4
    #b2df8a
    #33a02c
    #fb9a99
    #e31a1c
    #fdbf6f
    #ff7f00
    #cab2d6
    #6a3d9a
    #ffff99
    #b15928""".split()

bbox_colors = [c.strip() for c in bbox_colors]
bbox_colors = bbox_colors[1::2] + bbox_colors[::2]


def rect(bbox, c=None, facecolor='none', label=None, ax=None, line_width=1):
    r = Rectangle((bbox[1], bbox[0]), bbox[3], bbox[2], linewidth=line_width,
                  edgecolor=c, facecolor=facecolor, label=label)

    if ax is not None:
        ax.add_patch(r)
    return r


def rect_stn(ax, width, height, stn_params, c=None, line_width=3):
    bbox = SpatialTransformer.stn_to_pixel_coords(stn_params, (height, width))
    rect(bbox, c, ax=ax, line_width=line_width)


class ProgressFig(object):
    """Plots SQAIR results directly from SQAIR outputs.
    """
    _BBOX_COLORS = 'rgbymcw'

    def __init__(self, air, sess, checkpoint_dir=None, n_samples=10, seq_n_samples=3, dpi=300,
                 fig_scale=1.5):

        self.air = air
        self.sess = sess
        self.checkpoint_dir = checkpoint_dir
        self.n_samples = n_samples
        self.seq_n_samples = seq_n_samples
        self.dpi = dpi
        self.fig_scale = fig_scale

        self.n_steps = self.air.sequence._max_steps
        self.height, self.width = air.img_size[:2]

    def plot_all(self, global_step=None, save=True):
        self.plot_still(global_step, save)
        self.plot_seq(global_step, save)

    def plot_still(self, global_step=None, save=True):

        o = self._air_outputs(single_timestep=True)
        cmap = self._cmap(o.obs, with_time=False)
        fig, axes = self._make_fig(self.n_steps + 2, self.n_samples)

        # ground-truth
        for i, ax in enumerate(axes[0]):
            ax.imshow(o.obs[i], cmap=cmap, vmin=0, vmax=1)

        # reconstructions with marked steps
        for i, ax in enumerate(axes[1]):
            ax.imshow(o.canvas[i], cmap=cmap, vmin=0, vmax=1)
            for j, c in zip(xrange(self.n_steps), self._BBOX_COLORS):
                if o.presence[i, j] > .5:
                    self._rect(ax, o.where[i, j], c)

        # glimpses
        for i, ax_row in enumerate(axes[2:]):
            for j, ax in enumerate(ax_row):
                ax.imshow(o.presence[j, i] * o.glimpse[j, i], cmap=cmap)
                ax.set_title('{:d} with p({:d}) = {:.02f}'.format(int(o.presence[j, i]), i + 1,
                                                                  o.presence_prob[j, i]),
                             fontsize=4 * self.fig_scale)

                if o.presence[j, i] > .5:
                    for spine in 'bottom top left right'.split():
                        ax.spines[spine].set_color(self._BBOX_COLORS[i])
                        ax.spines[spine].set_linewidth(2.)

                ax_row[0].set_ylabel('glimpse #{}'.format(i + 1))

        for ax in axes.flatten():
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

        axes[0, 0].set_ylabel('ground-truth')
        axes[1, 0].set_ylabel('reconstruction')

        self._maybe_save_fig(fig, global_step, save, 'still_fig')

    def plot_seq(self, global_step=None, save=True):

        o, n_timesteps = self._air_outputs(n_samples=self.seq_n_samples)
        fig, axes = self._make_fig(2 * self.seq_n_samples, n_timesteps)
        axes = axes.reshape((2 * self.seq_n_samples, n_timesteps))

        unique_ids = np.unique(o.obj_id)[1:] # remove id == -1
        color_by_id = {i: c for i, c in zip(unique_ids, itertools.cycle(self._BBOX_COLORS))}
        color_by_id[-1] = 'k'

        cmap = self._cmap(o.obs)
        for t, ax in enumerate(axes.T):
            for n in xrange(self.seq_n_samples):
                pres_time = o.presence[t, n, :]
                obj_id_time = o.obj_id[t, n, :]
                ax[2 * n].imshow(o.obs[t, n], cmap=cmap, vmin=0., vmax=1.)

                n_obj = str(int(np.round(pres_time.sum())))
                id_string = ('{}{}'.format(color_by_id[i], i) for i in o.obj_id[t, n] if i > -1)
                id_string = ', '.join(id_string)
                title = '{}: {}'.format(n_obj, id_string)

                ax[2 * n + 1].set_title(title, fontsize=6 * self.fig_scale)
                ax[2 * n + 1].imshow(o.canvas[t, n], cmap=cmap, vmin=0., vmax=1.)
                for i, (p, o_id) in enumerate(zip(pres_time, obj_id_time)):
                    c = color_by_id[o_id]
                    if p > .5:
                        self._rect(ax[2 * n + 1], o.where[t, n, i], c, line_width=1.)

        for n in xrange(self.seq_n_samples):
            axes[2 * n, 0].set_ylabel('gt #{:d}'.format(n))
            axes[2 * n + 1, 0].set_ylabel('rec #{:d}'.format(n))

        for a in axes.flatten():
            a.grid(False)
            a.set_xticks([])
            a.set_yticks([])

        self._maybe_save_fig(fig, global_step, save, 'seq_fig')

    def _maybe_save_fig(self, fig, global_step, save, root_name):
        if save and self.checkpoint_dir is not None:
            fig_name = osp.join(self.checkpoint_dir, '{}_{}.png'.format(root_name, global_step))
            fig.savefig(fig_name, dpi=self.dpi)
            plt.close(fig)

    def _air_outputs(self, n_samples=None, single_timestep=False):

        if n_samples is None:
            n_samples = self.n_samples

        if not getattr(self, '_air_tensors', None):
            names = 'canvas glimpse presence_prob presence where obj_id'.split()
            tensors = {name: getattr(self.air, 'resampled_' + name, getattr(self.air, name)) for name in names}
            tensors = AttrDict(tensors)
            tensors.presence_prob = tensors.presence_prob
            tensors.obj_id = tf.to_int32(tensors.obj_id)

            # logits to coords
            tensors.where = SpatialTransformer.to_coords(tensors.where)
            tensors['obs'] = self.air.obs
            tensors['canvas'] = tf.clip_by_value(tensors['canvas'], 0., 1.)
            tensors['glimpse'] = tf.clip_by_value(tensors['glimpse'], 0., 1.)
            self._air_tensors = tensors

        res = self.sess.run(self._air_tensors)

        bs = np.random.choice(self.air.batch_size, size=n_samples, replace=False)

        ts = slice(None)
        if single_timestep:
            n_timesteps = res.obs.shape[0]
            ts = np.random.choice(n_timesteps, size=self.n_samples, replace=True)

        for k, v in res.iteritems():
            if v.shape[-1] == 1:
                v = v[..., 0]

            res[k] = v[ts, bs]

        if not single_timestep:
            n_timesteps = res['canvas'].shape[0]
            res = res, n_timesteps

        return res

    def _rect(self, ax, coords, color, line_width=2.):
        rect_stn(ax, self.width, self.height, coords, color, line_width=2.)

    def _make_fig(self, h, w, *args, **kwargs):
        figsize = self.fig_scale * np.asarray((w, h))
        return plt.subplots(h, w, figsize=figsize)

    def _cmap(self, obs, with_time=True):

        ndims = len(obs.shape)
        cmap = None
        if ndims == (3 + with_time) or (ndims == (4 + with_time) and obs.shape[-1] == 1):
            cmap = 'gray'
            
        return cmap


def make_logger(model, sess, writer, train_tensor, num_train_batches, test_tensor, num_test_batches, eval_on_train):
    exprs = {
        'vae': model.elbo_vae,
        'iwae': model.elbo_iwae,
        'steps': model.num_steps,
        'steps_acc': model.num_step_accuracy,
        'ess': model.ess,
        'data_ll': model.data_ll,
        'log_p_z': model.log_p_z,
        'log_q_z_given_x': model.log_q_z_given_x,
        'kl': model.kl,
        'mse': model.mse,
        'raw_mse': model.raw_mse
    }

    maybe_exprs = {
        # 'ess_alpha': lambda: model.alpha_ess,
        # 'alpha': lambda: model.alpha,
        'num_disc_steps': lambda: model.num_disc_steps,
        'num_prop_steps': lambda: model.num_prop_steps,
        'normalised_vae': lambda: model.normalised_elbo_vae,
        'normalised_iwae': lambda: model.normalised_elbo_iwae,
    }

    for k, expr in maybe_exprs.iteritems():
        try:
            exprs[k] = expr()
        except AttributeError:
            pass

    data_dict = {
        train_tensor['imgs']: test_tensor['imgs'],
        train_tensor['nums']: test_tensor['nums'],
    }

    try:
        data_dict[train_tensor['coords']] = test_tensor['coords']
    except KeyError:
        pass

    test_log = make_expr_logger(sess, num_test_batches, exprs,
                                writer=writer, name='test', data_dict=data_dict)

    if eval_on_train:
        train_log = make_expr_logger(sess, num_train_batches, exprs,
                                     writer=writer, name='train')

        def log(train_itr):
            train_log(train_itr)
            test_log(train_itr)
            print
    else:
        def log(train_itr):
            test_log(train_itr)
            print

    return log


def make_expr_logger(sess, num_batches, expr_dict, name, data_dict=None,
                     constants_dict=None, measure_time=True, writer=None):
    """

    :param sess:
    :param writer:
    :param num_batches:
    :param expr:
    :param name:
    :param data_dict:
    :param constants_dict:
    :return:
    """

    expr_dict = collections.OrderedDict(sorted(expr_dict.items()))

    tags = {k: '/'.join((k, name)) for k in expr_dict}
    data_name = 'Data {}'.format(name)
    log_string = ', '.join((''.join((k + ' = {', k, ':.4f}')) for k in expr_dict))
    log_string = ' '.join(('Step {},', data_name, log_string))

    if measure_time:
        log_string += ', eval time = {:.4}s'

        def make_log_string(itr, l, t):
            return log_string.format(itr, t, **l)
    else:
        def make_log_string(itr, l, t):
            return log_string.format(itr, **l)

    def log(itr, l, t):
        try:
            return make_log_string(itr, l, t)
        except ValueError as err:
            print err.message
            print '\tLogging items'
            for k, v in l.iteritems():
                print '{}: {}'.format(k, type(v))

    def logger(itr=0, num_batches_to_eval=None, write=True, writer=writer):
        l = {k: 0. for k in expr_dict}
        start = time.time()
        if num_batches_to_eval is None:
            num_batches_to_eval = num_batches

        for i in xrange(num_batches_to_eval):
            if data_dict is not None:
                vals = sess.run(data_dict.values())
                feed_dict = {k: v for k, v in zip(data_dict.keys(), vals)}
                if constants_dict:
                    feed_dict.update(constants_dict)
            else:
                feed_dict = constants_dict

            r = sess.run(expr_dict, feed_dict)
            for k, v in r.iteritems():
                l[k] += v

        for k, v in l.iteritems():
            l[k] /= num_batches_to_eval
        t = time.time() - start
        print log(itr, l, t)

        if writer is not None and write:
            log_values(writer, itr, [tags[k] for k in l.keys()], l.values())

        return l

    return logger


def log_ratio(var_tuple, name='ratio', eps=1e-8):
    """

    :param var_tuple:
    :param name:
    :param which_name:
    :param eps:
    :return:
    """
    a, b = var_tuple
    ratio = tf.reduce_mean(abs(a) / (abs(b) + eps))
    tf.summary.scalar(name, ratio)


def log_norm(expr_list, name):
    """

    :param expr_list:
    :param name:
    :return:
    """
    n_elems = 0
    norm = 0.
    for e in nest.flatten(expr_list):
        n_elems += tf.reduce_prod(tf.shape(e))
        norm += tf.reduce_sum(e ** 2)
    norm /= tf.to_float(n_elems)
    tf.summary.scalar(name, norm)
    return norm


def log_values(writer, itr, tags=None, values=None, dict=None):
    if dict is not None:
        assert tags is None and values is None
        tags = dict.keys()
        values = dict.values()
    else:

        if not nest.is_sequence(tags):
            tags, values = [tags], [values]

        elif len(tags) != len(values):
            raise ValueError('tag and value have different lenghts:'
                             ' {} vs {}'.format(len(tags), len(values)))

    for t, v in zip(tags, values):
        summary = tf.Summary.Value(tag=t, simple_value=v)
        summary = tf.Summary(value=[summary])
        writer.add_summary(summary, itr)


def gradient_summaries(gvs, norm=True, ratio=True, histogram=True):
    """Register gradient summaries.

    Logs the global norm of the gradient, ratios of gradient_norm/uariable_norm and
    histograms of gradients.

    :param gvs: list of (gradient, variable) tuples
    :param norm: boolean, logs norm of the gradient if True
    :param ratio: boolean, logs ratios if True
    :param histogram: boolean, logs gradient histograms if True
    """

    with tf.name_scope('grad_summary'):
        if norm:
            grad_norm = tf.global_norm([gv[0] for gv in gvs])
            tf.summary.scalar('grad_norm', grad_norm)

        for g, v in gvs:
            var_name = v.name.split(':')[0]
            if g is None:
                print 'Gradient for variable {} is None'.format(var_name)
                continue

            if ratio:
                log_ratio((g, v), '/'.join(('grad_ratio', var_name)))

            if histogram:
                tf.summary.histogram('/'.join(('grad_hist', var_name)), g)
