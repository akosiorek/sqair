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

"""Propagate and Discover modules for SQAIR."""
import itertools

import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from tensorflow.python.util import nest

import orderedattrdict

from core import DiscoveryCore
from modules import ConditionedNormalAdaptor, RecurrentNormal
from neural import MLP
import index
import nested
from prior import NumStepsDistribution


class BaseSQAIRModule(snt.AbstractModule):
    """Base class for Discover and PropagateOnlyTimestep modules."""
    _where_posterior = tfd.Normal

    def __init__(self):
        super(BaseSQAIRModule, self).__init__()

    def _make_posteriors(self, hidden_outputs):
        """Creates the posterior distributions.

        :param hidden_outputs: Iterable, outputs of a corresponding RNNCore.
        :return: 3-tuple of posterior distributions.
        """

        steps_posterior = self._make_step_posterior(hidden_outputs.presence_prob, hidden_outputs.presence_logit)
        where_posterior = self._where_posterior(hidden_outputs.where_loc, hidden_outputs.where_scale)
        what_posterior = tfd.Normal(hidden_outputs.what_loc, hidden_outputs.what_scale)

        return what_posterior, where_posterior, steps_posterior

    def _reduce_prob(self, x):
        return tf.reduce_sum(sum(x[:2]), -1) + x[-1]


class Discover(BaseSQAIRModule):
    """Discovery module."""

    def __init__(self, n_steps, cell, step_success_prob, where_mean=(-2., -2., 0., 0.),
                 where_std=(1., 1., 1., 1.), disc_prior_type='geom', rec_where_prior=False):
        super(Discover, self).__init__()

        self._n_steps = n_steps
        self._cell = cell
        self._init_disc_step_success_prob = step_success_prob
        self._what_prior = tfd.Normal(0., 1.)
        self._disc_prior_type = disc_prior_type

        with self._enter_variable_scope():
            if rec_where_prior:
                init = list(where_mean) + list(where_std)
                init = {'b': tf.constant_initializer(init)}
                self._where_prior = RecurrentNormal(4, 128, conditional=True, output_initializers=init)
            else:
                self._where_prior = ConditionedNormalAdaptor(where_mean, where_std)

    @property
    def n_what(self):
        return self._cell.n_what

    def initial_z(self, batch_size, n_steps):
        return self._cell.initial_z(batch_size, n_steps)

    def _build(self, img, n_present_obj, conditioning_from_prop=None, time_step=0,
               prior_conditioning=None, sample_from_prior=False, do_generate=False):
        """

        :param img: `Tensor` of shape `[B, H, W, C]` of images.
        :param n_present_obj: `Tensor` of integer numbers (but dtype=tf.float32) of shape `[B]` representing
            number of already present object for every data example in the batch.
        :param conditioning_from_prop: `Tensor` of shape `[B, n]` representing summury of propagated latent variables.
        :param time_step: Scalar tensor.
        :param prior_conditioning: `Tensor` of shape `[B, m]`, additional conditioning passed to prior distributions.
        :param sample_from_prior: Boolean; if True samples from the prior instead of the inference network.
        :param do_generate: if True, replaces sample from the posterior with a sample from the prior. Useful for
            conditional generation from a few observations (i.e. prediction).
        :return: AttrDict of results.
        """

        max_disc_steps = self._n_steps - n_present_obj

        hidden_outputs, num_steps = self._discover(img, max_disc_steps, conditioning_from_prop, time_step)
        hidden_outputs, log_probs = self._compute_log_probs(hidden_outputs, num_steps, time_step,
                                                            conditioning_from_prop, prior_conditioning,
                                                            sample_from_prior, do_generate)

        outputs = orderedattrdict.AttrDict(
            hidden_outputs=hidden_outputs,
            num_steps=num_steps,
            max_disc_steps=max_disc_steps
        )
        outputs.update(hidden_outputs)
        outputs.update(log_probs)

        return outputs

    def _discover(self, img, max_disc_steps, conditioning, time_step):  # pylint: disable=unused-variable
        """Performs object discovery.
        """

        initial_state = self._cell.initial_state(img)
        if conditioning is None:
            conditioning = tf.zeros((int(img.shape[0]), 1))

        seq_len_inpt = []
        for t in xrange(self._n_steps):
            exists = tf.greater(max_disc_steps, t)
            seq_len_inpt.append(tf.expand_dims(tf.to_float(exists), -1))

        inpt = [[conditioning, s] for s in seq_len_inpt]

        hidden_outputs, hidden_state = tf.nn.static_rnn(self._cell, inpt, initial_state)
        hidden_outputs = self._cell.outputs_by_name(hidden_outputs)

        num_steps = tf.reduce_sum(tf.squeeze(hidden_outputs.presence, -1), -1)

        return hidden_outputs, num_steps

    def _compute_log_probs(self, hidden_outputs, num_steps, time_step, conditioning_from_prop,
                           prior_conditioning, sample_from_prior, do_generate):
        """Computes log probabilities of latent variables from discovery under both q and p.
        """

        where_conditioning = tf.concat((conditioning_from_prop, prior_conditioning), -1)

        priors = self._make_priors(time_step, prior_conditioning)
        if sample_from_prior:

            what = priors[0].sample(hidden_outputs.what.shape)
            where = priors[1].sample(hidden_outputs.where.shape[:-1], conditioning=where_conditioning)

            pres_sample = priors[2].sample()
            pres_sample = tf.sequence_mask(pres_sample, maxlen=self._n_steps, dtype=tf.float32)
            pres_sample = tf.expand_dims(pres_sample, -1) * 0.

            dg = tf.to_float(do_generate)
            ndg = 1. - dg
            hidden_outputs.what = dg * what + ndg * hidden_outputs.what
            hidden_outputs.where = dg * where + ndg * hidden_outputs.where
            hidden_outputs.presence = dg * pres_sample + ndg * hidden_outputs.presence

        squeezed_presence = tf.squeeze(hidden_outputs.presence, -1)

        # outputs; short name due to frequent usage
        o = orderedattrdict.AttrDict()

        posteriors = self._make_posteriors(hidden_outputs)
        samples = [hidden_outputs.what, hidden_outputs.where, num_steps]
        posterior_log_probs = [distrib.log_prob(sample) for (distrib, sample) in zip(posteriors, samples)]

        kwargs = [dict(), {'conditioning': where_conditioning}, dict()]
        prior_log_probs = [distrib.log_prob(sample, **kw) for (distrib, sample, kw) in zip(priors, samples, kwargs)]

        for probs in (posterior_log_probs, prior_log_probs):
            for i in xrange(2):
                probs[i] = tf.reduce_sum(probs[i], -1) * squeezed_presence

        o.q_z_given_x = self._reduce_prob(posterior_log_probs)
        o.p_z = self._reduce_prob(prior_log_probs)

        for i, k in enumerate('what where num_step'.split()):
            o['{}_log_prob'.format(k)] = posterior_log_probs[i]
            o['{}_prior_log_prob'.format(k)] = prior_log_probs[i]

        o.num_steps_prob = posteriors[-1].probs

        return hidden_outputs, o

    def _make_priors(self, time_step, prior_conditioning):
        """Instantiates prior distributions for discovery.
        """

        is_first_timestep = tf.to_float(tf.equal(time_step, 0))

        if self._disc_prior_type == 'geom':
            num_steps_prior = tfd.Geometric(probs=1. - self._init_disc_step_success_prob)

        elif self._disc_prior_type == 'cat':
            init = [0.] * (self._n_steps + 1)
            step_logits = tf.Variable(init, trainable=True, dtype=tf.float32, name='step_prior_bias')

            # increase probability of zero steps when t>0
            init = [10.] + [0] * self._n_steps
            timstep_bias = tf.Variable(init, trainable=True, dtype=tf.float32, name='step_prior_timestep_bias')
            step_logits += (1. - is_first_timestep) * timstep_bias

            if prior_conditioning is not None:
                step_logits = tf.expand_dims(step_logits, 0) + MLP(10, n_out=self._n_steps + 1)(prior_conditioning)

            step_logits = tf.nn.elu(step_logits)
            num_steps_prior = tfd.Categorical(logits=step_logits)

        else:
            raise ValueError('Invalid prior type: {}'.format(self._disc_prior_type))

        return self._what_prior, self._where_prior, num_steps_prior

    def _make_step_posterior(self, presence_prob, presence_logit):  # pylint: disable=unused-variable
        return NumStepsDistribution(tf.squeeze(presence_prob, -1))


class Propagate(BaseSQAIRModule):
    """Propagation module."""

    def __init__(self, ssm, prior):
        """Initialises the module

        :param ssm: RNNCore, a state space model, see propagate.py for an example.
        :param prior: RNNCore for the propagation prior, see propagate.py for an example.
        """

        super(Propagate, self).__init__()
        self._ssm = ssm
        self._prior = prior
        self._where_posterior = self._ssm._cell._where_distrib

    def prior_init_state(self, batch_size, trainable=True, initializer=None):
        return self._prior.initial_state(batch_size, trainable, initializer)

    def _build(self, img, z_tm1, temporal_state, prior_state, sample_from_prior=False, do_generate=False):
        """

        :param img: `Tensor` of shape `[B, H, W, C]` representing images.
        :param z_tm1: 4-tuple of [what, where, presence, presence_logit] at the previous time-step.
        :param temporal_state: Hidden state of the temporal RNN.
        :param prior_state: Hidden state of the prior RNN.
        :param sample_from_prior: see Discovery class.
        :param do_generate: see Discovery class.
        :return: AttrDict of results.
        """

        presence_tm1 = z_tm1[2]
        prior_stats, prior_state = self._prior(z_tm1, prior_state)

        hidden_outputs, num_steps, delta_what, delta_where = self._ssm(img, z_tm1, temporal_state)
        hidden_outputs, log_probs = self._compute_log_probs(presence_tm1, hidden_outputs, prior_stats, delta_what,
                                                            delta_where, sample_from_prior=sample_from_prior,
                                                            do_generate=do_generate)

        outputs = orderedattrdict.AttrDict(
            prior_stats=prior_stats,
            prior_state=prior_state,
            hidden_outputs=hidden_outputs,
            num_steps=num_steps,
        )

        outputs.update(hidden_outputs)
        outputs.update(log_probs)
        return outputs

    def _compute_log_probs(self, presence_tm1, hidden_outputs, prior_stats, delta_what,
                           delta_where, sample_from_prior=False, do_generate=False):
        """Computes log probabilities, see Discovery class.
        """

        presence = tf.squeeze(hidden_outputs.presence, -1)
        presence_tm1 = tf.squeeze(presence_tm1, -1)
        o = orderedattrdict.AttrDict()

        posteriors = self._make_posteriors(hidden_outputs)
        priors = self._prior.make_distribs(prior_stats)

        samples = [delta_what, delta_where, presence]
        if sample_from_prior:
            samples = [p.sample() for p in priors]
            dg = tf.to_float(do_generate)
            ndg = 1. - dg
            hidden_outputs.what = dg * samples[0] + ndg * hidden_outputs.what
            hidden_outputs.where = dg * samples[1] + ndg * hidden_outputs.where

            pres = tf.to_float(tf.expand_dims(samples[2], -1))
            hidden_outputs.presence = dg * pres + ndg * hidden_outputs.presence

        posterior_log_probs = [distrib.log_prob(sample) for (distrib, sample) in zip(posteriors, samples)]

        samples = [hidden_outputs.what, hidden_outputs.where, presence]
        prior_log_probs = [distrib.log_prob(sample) for (distrib, sample) in zip(priors, samples)]
        o.prop_prob = tf.exp(posterior_log_probs[-1]) * presence_tm1

        for probs in (posterior_log_probs, prior_log_probs):
            for i in xrange(2):
                if probs[i].shape.ndims == 3:
                    probs[i] = tf.reduce_sum(probs[i], -1)

                probs[i] = probs[i] * presence_tm1 * presence

            probs[-1] = tf.reduce_sum(probs[-1] * presence_tm1, -1)

        o.q_z_given_x = self._reduce_prob(posterior_log_probs)
        o.p_z = self._reduce_prob(prior_log_probs)

        for i, k in enumerate('what where prop'.split()):
            o['{}_log_prob'.format(k)] = posterior_log_probs[i]
            o['{}_prior_log_prob'.format(k)] = prior_log_probs[i]

        return hidden_outputs, o

    def _make_step_posterior(self, presence_prob, presence_logit):  # pylint disable=unused-variable
        return tfd.Bernoulli(logits=tf.squeeze(presence_logit, -1))


class AbstractTimstepModule(snt.AbstractModule):
    """Abstract base-class for modules handling a single time-step of a sequence.
    """

    def __init__(self, n_steps, n_latent_code=0, relation_embedding=False):
        """Initialises the module.

        :param n_steps: Integer, number of inference steps to perform at this time-step.
        :param n_latent_code:  Integer, dimensionality of summary of latent variables.
        :param relation_embedding: Boolean; computes DeepSet-like embedding of latent variables if True.
        """
        super(AbstractTimstepModule, self).__init__()
        self._n_steps = n_steps
        self._n_latent_code = n_latent_code
        self._relation_embedding = relation_embedding

        with self._enter_variable_scope():
            if n_latent_code > 0:
                self._latent_encoder = MLP([n_latent_code] * 2)

    def initial_prior_state(self, batch_size):

        if not hasattr(self, '_initial_prior_state'):
            prior_init_state = self._propagate.prior_init_state(batch_size, trainable=True)
            self._initial_prior_state = nested.tile_along_newaxis(prior_init_state, self._n_steps, 1)

        return self._initial_prior_state

    def initial_temporal_state(self, *args, **kwargs):

        if not hasattr(self, '_initial_temporal_state'):
            state = self._propagate._ssm._cell._temporal_cell.initial_state(*args, **kwargs)
            self._initial_temporal_state = nested.tile_along_newaxis(state, self._n_steps, 1)

        return self._initial_temporal_state

    def _encode_latents(self, what, where, presence):
        """Encodes latent variables.
        """
        inpts = tf.concat((what, where), -1)

        if self._relation_embedding:
            def combinations(tensor):
                tensor = tf.split(tensor, self._n_steps, -2)
                tensor = itertools.combinations(tensor, 2)
                tensor = [tf.concat(t, -1) for t in tensor]
                tensor = tf.concat(tensor, -2)
                return tensor

            inpts = combinations(inpts)
            presence = tf.reduce_prod(combinations(presence), -1, keep_dims=True)

        features = snt.BatchApply(self._latent_encoder)(inpts) * presence
        return tf.reduce_sum(features, -2)


class PropagateOnlyTimestep(AbstractTimstepModule):
    """Mock for propagation-only model.

    This class was used in the development stage of the project, where the inference was initialized
    with ground-truth positions and presence of objects. Very useful to debug propagation.
    """

    def __init__(self, n_steps, propagate, time_cell=None, decoder=None, relation_embedding=False):
        n_units = nest.flatten(time_cell.state_size)[0] if time_cell else 0
        if isinstance(n_units, tf.TensorShape):
            n_units = n_units[0]

        super(PropagateOnlyTimestep, self).__init__(n_steps, n_units, relation_embedding)
        self._propagate = propagate
        self._time_cell = time_cell
        self._decoder = decoder

    def _build(self, img, z_tm1, temporal_hidden_state, prop_prior_state,
               time_step=0, sample_from_prior=False, do_generate=False):

        outputs = self._propagate(img, z_tm1, temporal_hidden_state, prop_prior_state,
                                  sample_from_prior, do_generate)

        outputs.z_t = (outputs.what, outputs.where, outputs.presence, outputs.presence_logit)
        outputs.prop_prior_state = prop_prior_state
        outputs.temporal_hidden_state = temporal_hidden_state
        return outputs


class SQAIRTimestep(AbstractTimstepModule):
    """Implements one time-step of propagation and discovery - full SQAIR model.
    """

    def __init__(self, n_steps, discover, propagate, time_cell, relation_embedding=False):
        """

        :param n_steps: Integer, total number of inference steps per time-step.
        :param discover: Discovery module.
        :param propagate: Propagate module.
        :param time_cell: RNNCell.
        :param relation_embedding: Boolean, see AbstractTimstepModule.
        """
        n_units = nest.flatten(discover._cell.state_size)[-1]
        if isinstance(n_units, tf.TensorShape):
            n_units = n_units[0]

        super(SQAIRTimestep, self).__init__(n_steps, n_units, relation_embedding)
        self._discover = discover
        self._propagate = propagate
        self._time_cell = time_cell

    @property
    def n_what(self):
        return self._discover.n_what

    def initial_z(self, batch_size):
        return self._discover.initial_z(batch_size, self._n_steps)

    def _build(self, img, z_tm1, temporal_hidden_state, prop_prior_state, highest_used_ids, prev_ids,
               time_step=0, sample_from_prior=False, do_generate=False):
        """

        :param img: `Tensor` of size `[B, H, W, C]` representing images.
        :param z_tm1: 4-tuple of [what, where, presence, presence_logit] from previous time-step.
        :param temporal_hidden_state: Hidden state of the time_cell.
        :param prop_prior_state: Hidden state of the propagation prior.
        :param highest_used_ids: Integer `Tensor` of size `[B]`, where each entry represent the highest used object
            ID for the corresponding data example in the batch.
        :param prev_ids: Integer `Tensor` of size `[B, n_steps]`, with each entry representing object ID of the
            corresponding object at the previous time-step.
        :param time_step: Integer.
        :param sample_from_prior: Boolean; if True samples from the prior instead of the inference network.
        :param do_generate: if True, replaces sample from the posterior with a sample from the prior. Useful for
            conditional generation from a few observations (i.e. prediction).
        :return: AttrDict of results.
        """

        batch_size = int(img.shape[0])
        prop_output, disc_output = \
            self._propagate_and_discover(img, z_tm1, temporal_hidden_state, prop_prior_state,
                                         time_step, sample_from_prior, do_generate)

        hidden_outputs, z_t, obj_ids, prop_prior_state, temporal_hidden_state, highest_used_ids = \
            self._choose_latents(batch_size, prop_output, disc_output, highest_used_ids, prev_ids)

        outputs = orderedattrdict.AttrDict(
            hidden_outputs=hidden_outputs,
            obj_ids=obj_ids,
            z_t=z_t,
            prop_prior_state=prop_prior_state,
            ids=obj_ids,
            highest_used_ids=highest_used_ids,
            prop=prop_output,
            disc=disc_output,
            temporal_hidden_state=temporal_hidden_state,
            presence_log_prob=prop_output.prop_log_prob + disc_output.num_step_log_prob,
            p_z=disc_output.p_z + prop_output.p_z,
            q_z_given_x=disc_output.q_z_given_x + prop_output.q_z_given_x
        )
        outputs.update(hidden_outputs)
        outputs.num_steps = tf.reduce_sum(tf.squeeze(outputs.presence, -1), -1)

        return outputs

    def _propagate_and_discover(self, img, z_tm1, temporal_hidden_state, prop_prior_state,
                                time_step, sample_from_prior, do_generate):
        """Propagates and discovers object. See self._build for argument docs.

        :return: AttrDicts returned by propagation and discovery.
        """

        prop_output = self._propagate(img, z_tm1, temporal_hidden_state, prop_prior_state,
                                      sample_from_prior, do_generate)
        conditioning_from_prop = self._encode_latents(prop_output.what, prop_output.where, prop_output.presence)

        discovery_inpt_img = img

        prop_prior_step_logits = tf.squeeze(prop_output.prior_stats[-1], -1)
        prop_prior_step_probs = (tf.nn.sigmoid(prop_prior_step_logits) - 0.5) / self._n_steps
        expected_prop_prior_num_step = tf.reduce_sum(prop_prior_step_probs, axis=-1, keep_dims=True)

        disc_output = self._discover(discovery_inpt_img, prop_output.num_steps, conditioning_from_prop, time_step,
                                     expected_prop_prior_num_step, sample_from_prior, do_generate)

        return prop_output, disc_output

    def _choose_latents(self, batch_size, prop_output, disc_output, highest_used_ids, prev_ids):
        """Picks outputs of propagation and discovery based on the presence values.

        This method is the work-horse of SQAIR. Assume that axis=1 is the object axis. Given dictionaries of outputs of
        propagation and discovery, and corresponding presence values (they are inside the dictionaries) this method:

        1) Concatenates prop, disc outputs in that order.
        2) Rearranges values in every tensor such that entries 1:k correspond to only present objects and entries
            k+1:self._n_steps to absent objects. Relative ordering between two objects and between two absent objects
            always remains the same.
        3) Truncates outputs to self._n_steps entries along the object axis, so that they can be used in future
            time-steps.
        4) Assigns new IDs to newly discovered objects and initialises corresponding temporal hidden states.
        5) Increments ID counters in `highest_used_ids` and updated 'prev_ids'.

        The above is not necessarily done in that order.

        :param batch_size: Integer, batch size (often denoted as `B` in docs).
        :param prop_output: AttrDict, output of propagation.
        :param disc_output: AttrDict, output of discovery.
        :param highest_used_ids: `Tensor`, highest used object IDs, see self._build for explanation.
        :param prev_ids: `Tensor`, previous object IDs, see self_build for explanation.

        :return: A bunch of stuff, see the code. It is appropriately reordered and truncated to self._n_steps
            along axis=1.
        """
        # 3) merge outputs of the two models

        # extract temporal prop and prior states from prop outputs and append newly initialised states
        # for any discovered objects
        prop_temporal_state = prop_output.temporal_state
        del prop_output.temporal_state
        prop_and_disc_temporal_state = nested.concat([prop_temporal_state, self.initial_temporal_state()], -2)

        prop_prior_rnn_state = prop_output.prior_state
        prop_and_disc_prior_hidden_states =\
            nested.concat([prop_prior_rnn_state, self.initial_prior_state(batch_size)], -2)

        # concat prop and disc outputs along the object axis, in that order
        outputs = [o.hidden_outputs.values() for o in (prop_output, disc_output)]
        hidden_outputs = [tf.concat((p, d), -2) for p, d in zip(*outputs)]
        hidden_outputs = DiscoveryCore.outputs_by_name(hidden_outputs, stack=False)

        # compute object ids based on ordering; discovered objects get new ids and propagated get the same as at the
        # previous timestep
        highest_used_ids, new_obj_id = index.compute_object_ids(highest_used_ids, prev_ids,
                                                                prop_output.presence, disc_output.presence)
        # gather all variables that are to be shuffled according to presence: vars for present objects get shifted
        # to the beginning and for absent ones to the end
        variables_to_partition = list(hidden_outputs.values()) + \
            [
                new_obj_id,
                prop_and_disc_prior_hidden_states,
                prop_and_disc_temporal_state,
            ]

        # # merge, partition, split to avoid partitioning each vec separately
        variables_to_partition = index.select_present_nested(variables_to_partition,
                                                             tf.squeeze(hidden_outputs.presence, -1), batch_size)
        variables_to_partition = nested.map_nested(lambda x: x[:, :self._n_steps], variables_to_partition)

        n_hiddens = len(hidden_outputs)
        hidden_outputs, (obj_ids, prop_prior_rnn_state, temporal_state)\
            = variables_to_partition[:n_hiddens], variables_to_partition[n_hiddens:]

        hidden_outputs = DiscoveryCore.outputs_by_name(hidden_outputs, stack=False)
        z_t = [hidden_outputs.what, hidden_outputs.where, hidden_outputs.presence, hidden_outputs.presence_logit]

        return hidden_outputs, z_t, obj_ids, prop_prior_rnn_state, temporal_state, highest_used_ids
