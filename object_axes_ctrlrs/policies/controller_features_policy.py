import numpy as np
import tensorflow as tf

from stable_baselines.common.policies import ActorCriticPolicy
from .utils import linear, PassThroughDiagGaussianProbabilityDistributionType, PassThroughCategoricalProbabilityDistributionType


class ControllerFeaturesIndependentScoresPolicy(ActorCriticPolicy):
    """
    For the case of priority and no expand_MDP

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param obs_enc_layers: (list) Layer sizes for observation encoder
    :param ctrl_enc_layers: (list) Layer sizes for controller encoder
    :param vf_pi_layers: (list) Layer sizes for value function and policy
    :param act_fun: (tf.func) the activation function to use in the neural network.
    The following parameters are required:
    :param N_obs_dim: (int) dimension of non-controller observations.
    :param N_ctrlrs: (int) how many controllers
    :param N_ctrlr_dim: (int) dimension of each controller feature
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, 
                obs_enc_layers=None, ctrl_enc_layers=None, vf_pi_layers=None, act_fun=tf.tanh, 
                N_obs_dim=-1, N_ctrlrs=-1, N_ctrlr_dim=-1):
        super(ControllerFeaturesIndependentScoresPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse)

        self._pdtype = PassThroughDiagGaussianProbabilityDistributionType(ac_space.shape[0])

        if obs_enc_layers is None:
            obs_enc_layers = [64, 64]
        if ctrl_enc_layers is None:
            ctrl_enc_layers = [64, 64]
        if vf_pi_layers is None:
            vf_pi_layers = [64]

        with tf.variable_scope("model", reuse=reuse):
            # Pass obs and ctrlr into separate encoders

            # all_obs should be:
            # B x (N_obs_dim + N_ctrlrs * N_ctrlr_dim)
            all_obs = tf.layers.flatten(self.processed_obs)

            obs_latent = all_obs[:, :N_obs_dim]
            for idx, layer in enumerate(obs_enc_layers):
                obs_latent = act_fun(linear(obs_latent, 'obs_enc{}'.format(idx), layer, init_scale=np.sqrt(2)))

            # Same controller encoding for all controllers
            ctrlrs_latent = []
            for n_ctrlr in range(N_ctrlrs):
                start_idx = n_ctrlr * N_ctrlr_dim
                end_idx = (n_ctrlr + 1) * N_ctrlr_dim
                latent_ctrlr = all_obs[:, N_obs_dim + start_idx: N_obs_dim + end_idx]
                
                for idx, layer in enumerate(ctrl_enc_layers):
                    latent_ctrlr = act_fun(linear(latent_ctrlr, 'ctrlr_enc{}'.format(idx), layer, init_scale=np.sqrt(2)))

                ctrlrs_latent.append(latent_ctrlr)

            # Concat latent obs and ctrlrs and pass through all pi's and vfs
            latents = [tf.concat([obs_latent, latent_ctrlr], axis=1) for latent_ctrlr in ctrlrs_latent]

            # pi_latents, vf_latents = [], []
            pi_latents = []
            for n_ctrlr in range(N_ctrlrs):
                # pi_latent, vf_latent = latents[n_ctrlr], latents[n_ctrlr]
                pi_latent = latents[n_ctrlr]

                for idx, layer in enumerate(vf_pi_layers):
                    pi_latent = act_fun(linear(pi_latent, 'pi{}'.format(idx), layer, init_scale=np.sqrt(2)))
                    # vf_latent = act_fun(linear(vf_latent, 'vf{}'.format(idx), layer, init_scale=np.sqrt(2)))

                pi_latents.append(linear(pi_latent, 'pi_out', 1))
                # vf_latents.append(vf_latent)

            pi_latent = tf.concat(pi_latents, axis=1)
            # pi_soft_attention = tf.math.sigmoid(pi_latent)
            # vf_latent = tf.math.reduce_sum(tf.stack(vf_latents, axis=2) * pi_soft_attention, axis=2)

            vf_latent = obs_latent
            for idx, layer in enumerate(vf_pi_layers):
                vf_latent = act_fun(linear(vf_latent, 'vf{}'.format(idx), layer, init_scale=np.sqrt(2)))

            # Process into actual pi and vf
            self._value_fn = linear(vf_latent, 'vf', 1)
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class ControllerFeaturesSingleNetworkDiscretePolicy(ActorCriticPolicy):
    """
    For the case of discrete_one_controller and no expand_MDP

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param obs_enc_layers: (list) Layer sizes for observation encoder
    :param ctrl_enc_layers: (list) Layer sizes for controller encoder
    :param vf_pi_layers: (list) Layer sizes for value function and policy
    :param act_fun: (tf.func) the activation function to use in the neural network.
    The following parameters are required:
    :param N_obs_dim: (int) dimension of non-controller observations.
    :param N_ctrlrs: (int) how many controllers
    :param N_ctrlr_dim: (int) dimension of each controller feature
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, 
                obs_enc_layers=None, ctrl_enc_layers=None, vf_pi_layers=None, act_fun=tf.tanh, 
                N_obs_dim=-1, N_ctrlrs=-1, N_ctrlr_dim=-1):
        super(ControllerFeaturesSingleNetworkDiscretePolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse)

        self._pdtype = PassThroughCategoricalProbabilityDistributionType(ac_space.n)

        if obs_enc_layers is None:
            obs_enc_layers = [64, 64]
        if ctrl_enc_layers is None:
            ctrl_enc_layers = [64, 64]
        if vf_pi_layers is None:
            vf_pi_layers = [64]

        with tf.variable_scope("model", reuse=reuse):
            all_obs = tf.layers.flatten(self.processed_obs)

            # Split obs, all ctrlrs
            obs_latent = all_obs[:, :N_obs_dim]
            ctrlrs_latent = []
            for n_ctrlr in range(N_ctrlrs):
                start_idx = n_ctrlr * N_ctrlr_dim
                end_idx = (n_ctrlr + 1) * N_ctrlr_dim
                ctrlrs_latent.append(all_obs[:, N_obs_dim + start_idx : N_obs_dim + end_idx])

            # Encode observations
            for idx, layer in enumerate(obs_enc_layers):
                obs_latent = act_fun(linear(obs_latent, 'obs_enc{}'.format(idx), layer, init_scale=np.sqrt(2)))

            # Encode controllers
            for n_ctrlr in range(N_ctrlrs):                
                for idx, layer in enumerate(ctrl_enc_layers):
                    ctrlrs_latent[n_ctrlr] = act_fun(linear(ctrlrs_latent[n_ctrlr], 'ctrlr_enc{}'.format(idx), layer, init_scale=np.sqrt(2)))

            # Produce all controller outputs for each conditioned pi and vf
            pi_latents = []
            for ctrlr_latent in ctrlrs_latent:
                fused_latent = tf.concat([obs_latent, ctrlr_latent], axis=1)
                pi_latent = fused_latent

                for idx, layer in enumerate(vf_pi_layers):
                    pi_latent = act_fun(linear(pi_latent, 'pi_{}'.format(idx), layer, init_scale=np.sqrt(2)))

                pi_latent = linear(pi_latent, 'pi', 1)
                pi_latents.append(pi_latent)
            pi_latent = tf.concat(pi_latents, axis=1)

            vf_latent = obs_latent
            for idx, layer in enumerate(vf_pi_layers):
                vf_latent = act_fun(linear(vf_latent, 'vf{}'.format(idx), layer, init_scale=np.sqrt(2)))

            # Process into actual pi and vf
            self._value_fn = linear(vf_latent, 'vf', 1)
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class ControllerFeaturesMultiNetworkScoresPolicy(ActorCriticPolicy):
    """
    For the case of discrete_one_controller and expand_MDP

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param obs_enc_layers: (list) Layer sizes for observation encoder
    :param ctrl_enc_layers: (list) Layer sizes for controller encoder
    :param vf_layers: (list) Layer sizes for value function
    :param pi_layers: (list) Layer sizes for policy
    :param vf_pi_layers : (list) layer sizes for shared vf pi layers. only used if fuse_backbones is True
    :param act_fun: (tf.func) the activation function to use in the neural network.
    The following parameters are required:
    :param N_action_steps: (int) the number of controller selection steps
    :param N_obs_dim: (int) dimension of non-controller observations.
    :param N_ctrlrs: (int) how many controllers
    :param N_ctrlr_dim: (int) dimension of each controller feature
    :param fuse_backbones: (bool) whether or not to use common backbone for vf and pi
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, 
                obs_enc_layers=None, ctrl_enc_layers=None, vf_layers=None, pi_layers=None, vf_pi_layers=None, act_fun=tf.tanh, 
                N_action_steps=-1, N_obs_dim=-1, N_ctrlrs=-1, N_ctrlr_dim=-1, fuse_backbones=False):
        super(ControllerFeaturesMultiNetworkScoresPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse)

        self._pdtype = PassThroughCategoricalProbabilityDistributionType(ac_space.n)

        N_conditioned_steps = N_action_steps - 1

        if obs_enc_layers is None:
            obs_enc_layers = [64, 64]
        if ctrl_enc_layers is None:
            ctrl_enc_layers = [64, 64]
        if vf_layers is None:
            vf_layers = [256, 256]
        if pi_layers is None:
            pi_layers = [32]
        if vf_pi_layers is None:
            vf_pi_layers = [64]

        with tf.variable_scope("model", reuse=reuse):
            all_obs = tf.layers.flatten(self.processed_obs)

            # Split obs, all ctrlrs, selected ctrlrs, and mask
            obs_latent = all_obs[:, :N_obs_dim]
            ctrlrs_latent = []
            for n_ctrlr in range(N_ctrlrs):
                start_idx = n_ctrlr * N_ctrlr_dim
                end_idx = (n_ctrlr + 1) * N_ctrlr_dim
                ctrlrs_latent.append(all_obs[:, N_obs_dim + start_idx : N_obs_dim + end_idx])

            selected_ctrlrs_latent = []
            for n_selected in range(N_conditioned_steps):
                start_idx = n_selected * N_ctrlr_dim
                end_idx = (n_selected + 1) * N_ctrlr_dim
                selected_ctrlrs_latent.append(all_obs[:, 
                    N_obs_dim + N_ctrlrs * N_ctrlr_dim + start_idx : N_obs_dim + N_ctrlrs * N_ctrlr_dim + end_idx
                ])

            action_step_mask = all_obs[:, -N_action_steps:]

            # Encode observations
            for idx, layer in enumerate(obs_enc_layers):
                obs_latent = act_fun(linear(obs_latent, 'obs_enc{}'.format(idx), layer, init_scale=np.sqrt(2)))

            # Encode controllers
            for n_ctrlr in range(N_ctrlrs):                
                for idx, layer in enumerate(ctrl_enc_layers):
                    ctrlrs_latent[n_ctrlr] = act_fun(linear(ctrlrs_latent[n_ctrlr], 'ctrlr_enc{}'.format(idx), layer, init_scale=np.sqrt(2)))

            # Encode selected controllers
            for n_selected in range(N_conditioned_steps):                
                for idx, layer in enumerate(ctrl_enc_layers):
                    selected_ctrlrs_latent[n_selected] = act_fun(linear(selected_ctrlrs_latent[n_selected], 'ctrlr_enc{}'.format(idx), layer, init_scale=np.sqrt(2)))

            if fuse_backbones:
                # Produce all controller outputs for each conditioned pi and vf
                pi_latents, vf_latents = [], []
                for n_step in range(N_action_steps):
                    conditioned_latents = selected_ctrlrs_latent[:n_step]
                    pi_latents.append([])
                    vf_latents.append([])

                    for ctrlr_latent in ctrlrs_latent:
                        fused_latent = tf.concat([obs_latent, ctrlr_latent] + conditioned_latents, axis=1)
                        pi_latent, vf_latent = fused_latent, fused_latent

                        for idx, layer in enumerate(vf_pi_layers):
                            pi_latent = act_fun(linear(pi_latent, 'pi_{}_{}'.format(n_step, idx), layer, init_scale=np.sqrt(2)))
                            vf_latent = act_fun(linear(vf_latent, 'vf_{}_{}'.format(n_step, idx), layer, init_scale=np.sqrt(2)))

                        pi_latent = linear(pi_latent, 'pi{}'.format(n_step), 1)
                        vf_latent = linear(vf_latent, 'vf{}'.format(n_step), 1)

                        pi_latents[n_step].append(pi_latent)
                        vf_latents[n_step].append(vf_latent)

                pi_pds = []
                for n_step in range(N_action_steps):
                    pi_latents[n_step] = tf.concat(pi_latents[n_step], axis=1)
                    vf_latents[n_step] = tf.concat(vf_latents[n_step], axis=1)

                # Use a mask to isolate the current network
                mask_0 = tf.reshape(action_step_mask[:, 0], (-1, 1))
                pi_pd = pi_latents[0] * mask_0
                vf_latent = vf_latents[0] * mask_0
                for i in range(1, N_action_steps):
                    mask_i = tf.reshape(action_step_mask[:, i], (-1, 1))
                    pi_pd += pi_latents[i] * mask_i
                    vf_latent += vf_latents[i] * mask_i

                # Process into actual pi and vf
                self._value_fn = tf.math.reduce_max(vf_latent, axis=1, keepdims=True)
                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(pi_pd, vf_latent, init_scale=0.01)
            else:
                # Produce all controller outputs for each conditioned pi
                pi_latents = []
                for n_step in range(N_action_steps):
                    conditioned_latents = selected_ctrlrs_latent[:n_step]
                    pi_latents.append([])

                    for ctrlr_latent in ctrlrs_latent:
                        fused_latent = tf.concat([obs_latent, ctrlr_latent] + conditioned_latents, axis=1)
                        pi_latent = fused_latent

                        for idx, layer in enumerate(pi_layers):
                            pi_latent = act_fun(linear(pi_latent, 'pi_{}_{}'.format(n_step, idx), layer, init_scale=np.sqrt(2)))

                        pi_latents[n_step].append(pi_latent)
                
                pi_pds = []
                for n_step in range(N_action_steps):
                    pds = []
                    for pi_latent in pi_latents[n_step]:
                        pds.append(linear(pi_latent, 'pi{}'.format(n_step), 1))
                    pi_pds.append(tf.concat(pds, axis=1))

                # Use a mask to isolate the current network
                mask_0 = tf.reshape(action_step_mask[:, 0], (-1, 1))
                pi_pd = pi_pds[0] * mask_0
                for i in range(1, N_action_steps):
                    mask_i = tf.reshape(action_step_mask[:, i], (-1, 1))
                    pi_pd += pi_pds[i] * mask_i

                vf_latent = obs_latent
                for idx, layer in enumerate(vf_layers):
                    vf_latent = act_fun(linear(vf_latent, 'vf{}'.format(idx), layer, init_scale=np.sqrt(2)))

                # Process into actual pi and vf
                self._value_fn = linear(vf_latent, 'vf', 1)
                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(pi_pd, vf_latent, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})