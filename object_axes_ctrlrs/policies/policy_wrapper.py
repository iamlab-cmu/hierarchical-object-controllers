import numpy as np
from abc import ABC
import os
from collections import deque

from stable_baselines import PPO2
from autolab_core import YamlConfig
from carbongym_utils.draw import draw_transforms

import copy

from object_axes_ctrlrs.exp.exp_utils import (str2bool, create_env_with_klass,
                                   get_ppo_clipfrac_exp_schedule, get_ppo_clipfrac_linear_schedule,
                                   create_model, load_cfg_for_model_cp,
                                   verify_current_cfg_with_model_cp_cfg, path_to_logdir,
                                   reset_config_to_test_mode)

from object_axes_ctrlrs.exp.stable_baseline_envs import create_franka_door_open_env as create_env
from object_axes_ctrlrs.envs.env_wrappers import ExpandMDPVecEnvWrapper


class ArgsStruct:
    '''Convert dictionary to obj artributes'''
    def __init__(self, **entries):
        self.__dict__.update(entries)


class PolicyWrapper(ABC):
    '''Wrapper for learned policy'''

    def __init__(self, cfg, model_cp, env_type, seed=0):
        assert env_type in ['door_open', 'hex_screw', 'block_tumble'], "Env type {} not supported".format(env_type)
        self._env_type = env_type
        self._cfg = YamlConfig(cfg)
        self._use_expandMDP = self._cfg['task']['expand_MDP']
        self._model_cp = model_cp
        assert os.path.exists(model_cp), "Model weight {} doesn't exist.".format(model_cp)
        self._seed = seed
        self._model_state = None

        self._build_model()

    def _build_model(self):
        if self._env_type == 'door_open':
            from object_axes_ctrlrs.exp.stable_baseline_envs import create_franka_door_open_env as create_env
        elif self._env_type == 'hex_screw':
            from object_axes_ctrlrs.exp.stable_baseline_envs import create_franka_hex_screw_env as create_env
        elif self._env_type == 'block_tumble':
            from object_axes_ctrlrs.exp.stable_baseline_envs import create_franka_block_tumble_env as create_env

        def custom_draws(scene):
            for env_idx, env_ptr in enumerate(scene.env_ptrs):
                ee_transform = vec_env._frankas[env_idx].get_ee_transform(env_ptr, 'franka0')
                desired_ee_transform = vec_env._frankas[env_idx].get_desired_ee_transform(env_idx, 'franka0')

                draw_transforms(scene.gym, scene.viewer, [env_ptr],
                                [ee_transform, desired_ee_transform])

        def inter_step_cb(vec_env, t_inter_step, n_inter_steps):
            vec_env.render(custom_draws=custom_draws)

        args = {'seed': self._seed, 'model_cp': self._model_cp, 'train': False, 'logdir': None}
        args_oba = ArgsStruct(**args)

        self._cfg['scene']['gui'] = 0
        vec_env = create_env(args_obj, self._cfg, inter_step_cb)
        self._vec_env = vec_env
        print("Did create simulated env")

        self._model = create_model(args_obj, vec_env, self._cfg)
        self._model.load_parameters(self._model_cp)

    def predict(self, all_obs):
        if all_obs.shape[0] != self._model.n_envs:
            all_obs = np.repeat(all_obs[0].reshape(1, -1), self._model.n_envs, axis=0)
        if not self._use_expandMDP:
            all_action, self._model_state = self._model.predict(all_obs, state=self._model_state, deterministic=True)
            return all_action
        else:
            self._num_steps = self._vec_env.venv.num_steps
            assert isinstance(self._vec_env.venv, ExpandMDPVecEnvWrapper)
            update_obs_cb = self._vec_env.venv.get_update_obs_cb()
            update_obs_cb_first = self._vec_env.venv.add_initial_controller_state_to_obs
            expanded_MDP_action = []

            self._vec_env.venv._action_deque = deque([], self._num_steps)
            self._vec_env.venv._action_step_counts = [1]

            for i in range(self._num_steps):
                if i == 0:
                    all_obs_expanded = update_obs_cb_first(all_obs)
                else:
                    self._vec_env.venv._action_deque.append(expanded_MDP_action[-1])
                    all_obs_expanded = update_obs_cb(all_obs, expanded_MDP_action[-1])
                    self._vec_env.venv._action_step_counts[0] += 1

                all_action, self._model_state = self._model.predict(all_obs_expanded, state=self._model_state,
                                                                    deterministic=True)
                expanded_MDP_action.append(all_action)
            return np.array(expanded_MDP_action).T
