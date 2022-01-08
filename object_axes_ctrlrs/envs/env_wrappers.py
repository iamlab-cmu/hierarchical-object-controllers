import numpy as np
import h5py
import os
import os.path as osp
import pickle
from collections import deque, OrderedDict

from gym.spaces import Box, Discrete
from stable_baselines.common.vec_env.base_vec_env import VecEnvWrapper

from object_axes_ctrlrs.utils.hdf5_utils import recursively_save_dict_contents_to_group


def get_env_controller_info(venv):

    def _convert_np_array_to_list(x):
        if type(x) is np.ndarray:
            return x.tolist()
        return x
    controller_info = dict()

    if hasattr(venv, '_high_level_controllers'):
        high_level_controller = venv._high_level_controllers[0]
        low_level_controllers = high_level_controller.low_level_controllers_dict

        low_level_controller_info = {}
        ctrlr_idx = 0
        for ctrlr_key, ctrlr in low_level_controllers.items():
            klass = str(ctrlr.__class__)
            axis = _convert_np_array_to_list(ctrlr.axis)
            target = _convert_np_array_to_list(ctrlr.target)

            low_level_controller_info[ctrlr_idx] = dict(
                klass=klass,
                axis=axis,
                target=target,
                ctrlr_key=ctrlr_key,
            )
            if hasattr(ctrlr, 'mag'):
                mag = _convert_np_array_to_list(ctrlr.mag)
                low_level_controller_info[ctrlr_idx]['mag'] = mag
            ctrlr_idx += 1
        
        controller_info['low_level_controller_info'] = low_level_controller_info

    if hasattr(venv, '_low_level_controller_combos'):
        controller_info['low_level_controller_combos'] = venv._low_level_controller_combos

    return controller_info


class SaveInteractionEnvWrapper(VecEnvWrapper):

    def __init__(self, venv, logdir):
        super(SaveInteractionEnvWrapper, self).__init__(venv)
        self.history_by_episode_dict = dict()
        self.curr_episode = -1

        results_dir = os.path.join(logdir, 'evaluation_results')
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        print(f'Will save results to: {results_dir}')
        self.h5_path = os.path.join(results_dir, 'interaction_data.h5')
        self.pkl_path = os.path.join(results_dir, 'interaction_info.pkl')
        self.env_data = dict()

        self.env_data['low_level_controller_info'] = get_env_controller_info(self.venv)

        with open(self.pkl_path, 'wb') as pkl_f:
            pickle.dump(self.env_data, pkl_f, protocol=2)

    def step_async(self, actions):
        raise ValueError("Should invoke step method directly.")

    def step_wait(self):
        raise ValueError("Should invoke step method directly.")

    def reset(self):
        observations = self.venv.reset()
        self.curr_episode += 1
        self._actions = None
        self._observations = observations

        if self.curr_episode > 0:
            last_episode = self.curr_episode - 1
            for k in self.history_by_episode_dict[str(last_episode)].keys():
                v = self.history_by_episode_dict[str(last_episode)][k]
                self.history_by_episode_dict[str(last_episode)][k] = np.array(v).squeeze()

            h5f = h5py.File(self.h5_path, 'w')
            recursively_save_dict_contents_to_group(h5f, '/', self.history_by_episode_dict)
            h5f.flush()
            h5f.close()
            with open(self.pkl_path, 'wb') as pkl_f:
                pickle.dump(self.env_data, pkl_f, protocol=2)
            print(f"Did save \n"
                  f"traj data: {self.h5_path}\n"
                  f" env_data: {self.pkl_path}")

        self.history_by_episode_dict[str(self.curr_episode)] = {
            'action': [],
            'obs': [],
            'reward': [],
            'done': [],
        }
        self.history_by_episode_dict[str(self.curr_episode)]['obs'].append(
            np.copy(observations))

        return observations
    
    def step(self, all_actions):
        all_obs, all_rews, all_dones, all_infos = self.venv.step(all_actions)
        key_to_data = [
            ('action', all_actions),
            ('obs', all_obs),
            ('reward', all_rews),
            ('done', all_dones)
        ]
        for key, value in key_to_data:
            self.history_by_episode_dict[str(self.curr_episode)][key].append(
                np.copy(value))
        
        return all_obs, all_rews, all_dones, all_infos


class SaveInteractionVecEnvWrapper(VecEnvWrapper):

    def __init__(self, venv, logdir):
        super(SaveInteractionVecEnvWrapper, self).__init__(venv)
        self.n_envs = venv.n_envs
        self.episode_idx_by_env = np.zeros((self.n_envs), dtype=np.int32)

        self.history_by_episode_dict = {
            'obs': deque([], 1001),
            'next_obs': deque([], 1001),
            'action': deque([], 1001),
            'reward': deque([], 1001),
            'done': deque([], 1001)
        }
        self.history_by_env_by_episode = dict()
        for i in range(self.n_envs):
            self.history_by_env_by_episode[str(i)] = dict()

        self.reset_save_obs_data_path(osp.join(logdir, 'evaluation_results'))

        self.env_data = dict()

        self.env_data['controller_info'] = get_env_controller_info(self.venv)

        with open(self.pkl_path, 'wb') as pkl_f:
            pickle.dump(self.env_data, pkl_f, protocol=2)
        
        self.curr_obs = None
    
    def reset_save_obs_data_path(self, logdir):
        results_dir = osp.join(logdir, 'evaluation_results')
        if not osp.exists(logdir):
            os.makedirs(logdir)
        if not osp.exists(results_dir):
            os.mkdir(results_dir)
        print(f'Will save results to: {results_dir}')
        self.h5_path = osp.join(results_dir, 'interaction_data.h5')
        self.pkl_path = osp.join(results_dir, 'interaction_info.pkl')

    def step_async(self, actions):
        raise ValueError("Should invoke step method directly.")

    def step_wait(self):
        raise ValueError("Should invoke step method directly.")
    
    def _clear_data(self):
        for hist_deque in self.history_by_episode_dict.values():
            hist_deque.clear()

    def reset(self):
        observations = self.venv.reset()
        self.curr_obs = np.copy(observations)
        return observations
    
    def get_last_done_index(self, env_idx):
        dones = self.history_by_episode_dict['done']
        for i in range(len(dones) - 2, -1, -1):
            if dones[i][env_idx]:
                return i
        return -1

    def step(self, all_actions):
        all_obs, all_rews, all_dones, all_infos = self.venv.step(all_actions)
        key_to_data = [
            ('action', all_actions),
            ('obs', self.curr_obs),
            ('next_obs', all_obs),
            ('reward', all_rews),
            ('done', all_dones)
        ]
        for key, value in key_to_data:
            self.history_by_episode_dict[key].append(np.copy(value))

        envs_done = (np.where(all_dones)[0]).tolist()
        for env_idx in envs_done:
            last_done_idx = self.get_last_done_index(env_idx) 
            env_data = {}
            for key, _ in key_to_data:
                all_envs_data = self.history_by_episode_dict[key]
                d = [all_envs_data[i][env_idx] for i in range(last_done_idx+1, len(all_envs_data))]
                env_data[key] = np.array(d)

            episode_idx = self.episode_idx_by_env[env_idx]
            self.history_by_env_by_episode[str(env_idx)][str(episode_idx)] = env_data
            self.episode_idx_by_env[env_idx] += 1
        
        if len(envs_done) > 0:
            # Some envs were finished, let us save the data.
            self.save_h5_data()

        self.curr_obs = np.copy(all_obs)
        
        return all_obs, all_rews, all_dones, all_infos
    
    def save_h5_data(self):
        h5f = h5py.File(self.h5_path, 'w')
        recursively_save_dict_contents_to_group(
            h5f, '/', self.history_by_env_by_episode)
        h5f.flush()
        h5f.close()
        with open(self.pkl_path, 'wb') as pkl_f:
            pickle.dump(self.env_data, pkl_f, protocol=2)
        print(f"Did save \n"
              f"traj data: {self.h5_path}\n"
              f" env_data: {self.pkl_path}")

class SeparatePoliciesVecEnvWrapper(VecEnvWrapper):
    def __init__(self, vec_env, all_models, cfg):
        super().__init__(vec_env)
        self.all_models = all_models
        self.vec_env = vec_env
        cfg["action"]["type"] = "combo"
        self.n_envs = vec_env.n_envs
        self.vec_env.change_action_space(cfg)

    def step_wait(self):
        raise ValueError("Should invoke step method directly.")

    def step(self,all_actions,index_to_learn=None, obs=None, states=None, dones=None):
        #pick out other 2 indices, get their values. all_actions refers to the actions for the index to learn
        model_indices = list(range(len(self.all_models)))
        actions = []
        for model_idx in model_indices:
            if model_idx != index_to_learn:
                test_time_actions, _, _, _ = self.all_models[model_idx].step(obs, states, dones)
                actions.append(test_time_actions)
            else:
                actions.append(all_actions)

        combined_action = np.stack(actions, axis=1) #properly combine the action
        return self.vec_env.step(combined_action) #requires it to be in the combo space.

    def reset(self, env_idxs=None):
        obs = self.venv.reset(env_idxs=env_idxs)
        return obs

class ExpandMDPVecEnvWrapper(VecEnvWrapper):

    def __init__(self, venv, num_steps, same_ctrlr_reward, controller_obs_type='single_one_hot'):
        super(ExpandMDPVecEnvWrapper, self).__init__(venv)
        self._num_steps = num_steps
        self._action_as_obs_id = -1
        self.n_envs = self.venv.n_envs
        self._same_ctrlr_reward = same_ctrlr_reward
        self.add_reward_for_same_controller = True

        self._controller_obs_types = (
            'single_one_hot',
            'multi_one_hot',
            'features'
        )
        assert controller_obs_type in self._controller_obs_types
        self._obs_type = controller_obs_type

        self.observation_space = self._init_obs_space()

        # The original MDP that the "RL agent" sees is the low-level expanded
        # MDP hence the original _step_count corresponds to that MDP. In here
        # we manually keep track of the original high-level MDP.
        self._action_step_counts = np.zeros((self.n_envs), dtype=np.int32)
        self._action_step_counts_curr = np.zeros((self.n_envs), dtype=np.int32)
        self._last_obs = None
        # Deque used to store last N actions.
        self._action_deque = deque([], num_steps)
        # Used only for book-keeping and verification purpose
        self._did_apply_action_for_env = np.zeros((self.n_envs), dtype=np.int32)

    @property
    def num_steps(self):
        return self._num_steps

    def _init_obs_space(self):
        obs_space = self.venv.observation_space
        n_low_level_controllers = self.venv._n_low_level_controllers

        if self._obs_type == 'single_one_hot':
            limits_low = np.zeros(n_low_level_controllers)
            limits_high = np.ones(n_low_level_controllers)
        elif self._obs_type == 'multi_one_hot':
            n_steps = self._num_steps - 1
            limits_low = np.zeros(n_low_level_controllers * n_steps)
            limits_high = np.ones(n_low_level_controllers * n_steps)
        elif self._obs_type == 'features':
            conditioned_steps = self.num_steps - 1
            limits_low = np.concatenate([np.repeat(self.venv._controller_feature_extractor[0].low, conditioned_steps), np.zeros(self.num_steps)])
            limits_high = np.concatenate([np.repeat(self.venv._controller_feature_extractor[0].high, conditioned_steps), np.ones(self.num_steps)])
        else:
            raise ValueError(f"Invalid obs type: {self._obs_type}")

        self._ctrl_obs_space = Box(limits_low, limits_high, dtype=np.float32)

        limits_low = np.concatenate([obs_space.low, limits_low])
        limits_high = np.concatenate([obs_space.high, limits_high])

        obs_space = Box(limits_low, limits_high, dtype=np.float32)
        return obs_space

    def step_async(self, actions):
        raise ValueError("Should invoke step method directly.")

    def step_wait(self):
        raise ValueError("Should invoke step method directly.")

    def add_initial_controller_state_to_obs(self, all_obs):
        n_envs = self.n_envs
        n_low_level_controllers = self.venv._n_low_level_controllers
        if self._obs_type == 'single_one_hot':
            expand_mdp_obs = np.zeros((n_envs, n_low_level_controllers))
        elif self._obs_type == 'multi_one_hot':
            expand_mdp_obs = np.zeros((n_envs, (self._num_steps - 1 ) * n_low_level_controllers))
        elif self._obs_type == 'features':
            n_ctrlr_obs = (self.num_steps - 1) * self.venv._controller_feature_extractor[0].dim
            expand_mdp_obs = np.zeros((n_envs, n_ctrlr_obs + self.num_steps))
            expand_mdp_obs[:, n_ctrlr_obs] = 1
        else:
            raise ValueError(f"Invalid obs type: {self._obs_type}")

        all_obs = np.concatenate((all_obs, expand_mdp_obs), axis=1)
        return all_obs

    def get_update_obs_cb(self):
        obs_type = self._obs_type

        def f_update_obs_cb(all_obs, all_actions):
            col_idx = all_obs.shape[1]
            assert len(self._ctrl_obs_space.shape) == 1
            ctrlr_obs_size = self._ctrl_obs_space.shape[0]
            all_obs = np.concatenate(
                [all_obs, np.zeros((all_obs.shape[0], ctrlr_obs_size))], axis=1)

            num_action_steps = self._action_step_counts[0]
            n_low_level_controllers = self.venv._n_low_level_controllers
            step_mask = np.zeros(self.num_steps)
            step_mask[num_action_steps] = 1

            for env_idx in range(self.n_envs):
                curr_action = all_actions[env_idx]

                if obs_type == 'single_one_hot':
                    if num_action_steps == 0:
                        raise ValueError("Invalid value, num_action_steps cannot be 0.")
                    elif num_action_steps == 1:
                        all_obs[env_idx, col_idx + curr_action] = 1
                    elif num_action_steps == 2:
                        all_obs[env_idx, col_idx + curr_action] = 1
                        prev_action = self._action_deque[-2][env_idx]
                        all_obs[env_idx, col_idx + prev_action] = 1
                    else:
                        raise ValueError(f"Invalid value, num_action_steps cannot be {num_action_steps}.")
                elif obs_type == 'multi_one_hot':
                    if num_action_steps == 0:
                        raise ValueError("Invalid value, num_action_steps cannot be 0.")
                    elif num_action_steps == 1:
                        all_obs[env_idx, col_idx + curr_action] = 1
                    elif num_action_steps == 2:
                        prev_action = self._action_deque[-2][env_idx]
                        all_obs[env_idx, col_idx + prev_action] = 1
                        all_obs[env_idx, col_idx + n_low_level_controllers + curr_action] = 1
                    else:
                        raise ValueError(f"Invalid value, num_action_steps cannot be {num_action_steps}.")
                elif obs_type == 'features':
                    ctrlr_feature_dim = self.venv._controller_feature_extractor[0].dim
                    start_idx = col_idx + (num_action_steps - 1) * ctrlr_feature_dim
                    end_idx = col_idx + num_action_steps * ctrlr_feature_dim
                    all_obs[env_idx, start_idx:end_idx] = self.venv._controller_feature_extractor[env_idx].latest_features[curr_action]
                    all_obs[env_idx, col_idx + (self.num_steps - 1) * ctrlr_feature_dim:] = step_mask
                else:
                    raise ValueError(f"Invalid obs type: {obs_type}")

            return all_obs
        return f_update_obs_cb

    def step(self, all_actions):
        self._did_apply_action_for_env[:] = 0
        self._action_deque.append(all_actions.copy())

        self._action_step_counts += 1

        assert len(np.unique(self._action_step_counts)) == 1, \
            "Enviroments are out of phase"
        if np.all(self._action_step_counts == self._num_steps):
            expanded_mdp_actions = []
            for t in range(-self._num_steps, 0):
                expanded_mdp_actions.append(self._action_deque[t])
            expanded_mdp_actions = np.vstack(expanded_mdp_actions).T

            all_obs, all_rews, all_dones, all_infos = self.venv.step(expanded_mdp_actions)

            # Add the initial data to observations? Maybe move this method
            # to vec_env.
            all_obs = self.add_initial_controller_state_to_obs(all_obs)

            if self.add_reward_for_same_controller:
                all_rews = self.update_reward_for_controller_selection(
                    all_obs, all_actions, all_rews, all_dones, all_infos)

            self._action_step_counts[:] = 0
        else:
            # Update intermediate controllers being chosen for this vec env.
            update_obs_cb = self.get_update_obs_cb()
            # TODO: We don't need this function anymore but I will keep this for now.
            all_obs, all_rews, all_dones, all_infos = self.venv.step_expanded_mdp(
                all_actions, 
                update_obs_cb)

            if self.add_reward_for_same_controller:
                all_rews = self.update_reward_for_controller_selection(
                    all_obs, all_actions, all_rews, all_dones, all_infos)
            
        return all_obs, all_rews, all_dones, all_infos
    
    def reset(self, env_idxs=None):
        if env_idxs is None:
            self._action_step_counts[:] = 0
        else:
            self._action_step_counts[env_idxs] = 0

        obs = self.venv.reset(env_idxs=env_idxs)
        assert np.all(self._action_step_counts == 0)

        obs = self.add_initial_controller_state_to_obs(obs)
        return obs

    def update_reward_for_controller_selection(self, all_obs, all_action, 
                                               all_rews, all_dones, all_infos):

        curr_steps = self._action_step_counts[0]
        max_steps = self._num_steps

        assert curr_steps > 0 and curr_steps <= max_steps, "Invalid step count."
        if curr_steps == 1:
            return all_rews

        # Start from second last action i.e. the action before current action
        # and move back to see that we did not choose the same controller
        for i in range(2, curr_steps + 1):
            step_i_action = self._action_deque[-i] 
            for env_idx in range(self.n_envs):
                if all_action[env_idx] == step_i_action[env_idx] \
                    and all_action[env_idx] != 0:
                    all_rews[env_idx] += self._same_ctrlr_reward
        
        return all_rews
