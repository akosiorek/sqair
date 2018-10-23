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

import os
import numpy as np
import cPickle as pickle
from tensorflow.python.util import nest

from trajectory import NoisyAccelerationTrajectory
from template import TemplateDataset

from data import create_mnist, _MNIST_PATH


def trajectory_from_coords(coords, n_timesteps, canvas_size=None, template_size=None,
                           overlap=None, trajectory=None):

    coords = list(coords)
    flat_coords = np.asarray(nest.flatten(coords))
    n_samples = len(flat_coords)

    if trajectory is None:
        template_size = np.asarray(template_size)
        allowed_region = np.asarray(canvas_size) - overlap * template_size
        y_bounds = [-overlap * template_size[0], allowed_region[0]]
        x_bounds = [-overlap * template_size[1], allowed_region[1]]
        bounds = [y_bounds, x_bounds]

        trajectory = NoisyAccelerationTrajectory(
            noise_std=.01,
            n_dim=2,
            pos_bounds=bounds,
            max_speed=10,
            max_acc=3,
            bounce=True
        )

    tjs = trajectory.create(n_timesteps, n_samples, init_from=flat_coords)
    tjs = np.hsplit(tjs, tjs.shape[1])
    tjs = [tj.squeeze() for tj in tjs]
    tjs = np.asarray(nest.pack_sequence_as(coords, tjs))
    return tjs


def fix_data(data_dict, trajectories, img_seq):
    """
    1. Transpose nums to have shape (1, N, max_objects+1)
    2. Convert list of coords into a numpy array with zeros for absent objects and replace coords in `data_dict`
    3. Replace images from `data_dict` with the image sequence
    4. Remove templates from `data_dict`

    :param data_dict: dict, data dictionary
    :return: None, works in place
    """
    data_dict['nums'] = data_dict['nums'].T
    nums = data_dict['nums'].astype(np.int32).sum(-1)
    n_max = nums.max()
    n_timesteps, n_samples = img_seq.shape[:2]
    coords = np.zeros((n_timesteps, n_samples, n_max, 4))
    for i, coord in enumerate(trajectories):
        for num in xrange(nums[0, i]):
            coords[:, i, num, :2] = trajectories[i][num]
            coords[:, i, num, 2:] = data_dict['templates'][i][num].shape
    data_dict['coords'] = coords

    data['imgs'] = img_seq
    del data['templates']

if __name__ == '__main__':
    partitions = ['train', 'validation']
    nums = [60000, 10000]
    # nums = [n//100 for n in nums]

    name = 'seq_mnist'

    # seq parameters
    n_timesteps = 10
    overlap = 0.

    # mnist parameters
    canvas_size = (50, 50)
    obj_size = (28, 28)
    n_objects = (0, 2)
    init_fraction_outside_canvas = 0.
    include_templates = True
    include_coords = True

    for p, n in zip(partitions, nums):
        print 'Processing partition "{}"'.format(p)
        print 'Creating static data'
        data = create_mnist(p, canvas_size, obj_size, n_objects,
                            n_samples=n,
                            fraction_outside_canvas=init_fraction_outside_canvas,
                            include_coords=include_coords,
                            include_templates=include_templates
                            )

        print 'Creating sequences'
        tjs = trajectory_from_coords(data['coords'], n_timesteps, canvas_size,
                                     obj_size, overlap)

        td = TemplateDataset(canvas_size, n_timesteps)
        img_seq = td.create(tjs, data['templates'])

        fix_data(data, tjs, img_seq)
        filename = '{}_{}.pickle'.format(name, p)
        filename = os.path.join(_MNIST_PATH, filename)

        print 'saving to "{}"'.format(filename)
        with open(filename, 'w') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
