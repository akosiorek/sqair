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

import itertools
import numpy as np
from tensorflow.python.util import nest

from scipy.misc import imresize


def constrain_dims(a, b, DIM):
    ai = 0 if a >= 0 else -a
    d = min(DIM - b, 0)
    bi = b - a + d
    return ai, max(bi, 0)


def convert_img_dtype(imgs, dtype):
    if dtype == np.uint8:
        imgs = (imgs - imgs.min()) / (imgs.max() / 255.)
        imgs = imgs.astype(np.uint8)
    return imgs


class TemplateDataset(object):
    """Creates a dataset of floating templates."""

    def __init__(self, canvas_size, n_timesteps):
        """
        :param canvas_size: tuple of ints, size of the canvas that the templates will be placed in
        :param n_timesteps: int, number of timesteps of a sequence
        """

        super(TemplateDataset, self).__init__()
        self._canvas_size = tuple(canvas_size)
        self.n_timesteps = n_timesteps

    def create(self, coords, templates, dtype=np.uint8):
        n_samples = len(templates)
        canvas = np.zeros((self.n_timesteps, n_samples) + self._canvas_size, dtype=np.float32)

        for i, (tjs, seq_templates) in enumerate(itertools.izip(coords, templates)):
            for tj, template in zip(tjs, seq_templates):
                for t in xrange(len(tj)):
                    self._blend(canvas[t, i], template, tj[t])

        return convert_img_dtype(canvas, dtype)

    def _blend(self, canvas, template, pos):
        """Blends `template` into `canvas` at position given by `pos`

        :param canvas:
        :param template:
        :param pos:
        """

        template_shape = template.shape[:2]
        height, width = canvas.shape[:2]

        pos = np.round(pos)
        y0, x0 = pos
        y1, x1 = pos + template_shape
        y0, x0, y1, x1 = (int(i) for i in (y0, x0, y1, x1))

        yt0, yt1 = constrain_dims(y0, y1, height)
        xt0, xt1 = constrain_dims(x0, x1, width)

        y0, y1 = min(max(y0, 0), height), max(min(y1, height), 0)
        x0, x1 = min(max(x0, 0), width), max(min(x1, width), 0)

        self._blend_slice(canvas, template, (y0, y1, x0, x1), (yt0, yt1, xt0, xt1))

    @staticmethod
    def _blend_slice(canvas, template, dst, src):
        """Merges the slice of `template` given by indices in `src` into the slice of `canvas` given by indices `dst`.

        :param canvas:
        :param template:
        :param dst:
        :param src:
        """
        current = canvas[dst[0]:dst[1], dst[2]:dst[3]]
        target = template[src[0]:src[1], src[2]:src[3]]
        canvas[dst[0]:dst[1], dst[2]:dst[3]] = np.maximum(current, target)