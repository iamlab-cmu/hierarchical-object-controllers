import numpy as np
import os
import pickle
import copy

import tensorflow as tf
from stable_baselines.common.callbacks import BaseCallback


class PPOTensorboardCallback(BaseCallback):

    def __init__(self, results_save_dir, n_envs, train_in_multi_env, verbose=0):
        super(PPOTensorboardCallback, self).__init__(verbose)

        self.success_count = 0
        self.episode_count = 0
        self.num_rollouts = 0
        self.results_save_dir = results_save_dir

        self._n_envs = n_envs
        self._train_in_multi_env = train_in_multi_env
        self.train_stats_by_rollout_idx = dict()
        self.rollout_idx = 0

        if train_in_multi_env:
            self.success_count_by_env_idx = np.zeros((n_envs), dtype=np.int32)
            self.episode_count_by_env_idx = np.zeros((n_envs), dtype=np.int32)

        self.success_count_by_env_type = None
        self.episode_count_by_env_type = None
        self.wall_config_by_env_type = None

    def _on_step(self):
        infos = self.training_env.last_infos
        success_count = sum([info['episode']['is_success'] for info in infos])
        dones = self.model.runner.dones
        done_count = np.sum(dones)

        if self._train_in_multi_env:
            done_env_idx = (np.where(dones)[0]).tolist()
            for env_idx in done_env_idx:
                self.success_count_by_env_idx[env_idx] += \
                    infos[env_idx]['episode']['is_success']
                self.episode_count_by_env_idx[env_idx] += 1

        self.success_count += success_count
        self.episode_count += done_count

        return True

    def _on_rollout_start(self):
        self.success_count = 0
        self.episode_count = 0
        self.num_rollouts += 1
        if self._train_in_multi_env:
            self.success_count_by_env_idx[:] = 0
            self.episode_count_by_env_idx[:] = 0

    def _on_rollout_end(self):
        writer = self.locals['writer']
        summary_1 = tf.Summary(
            value=[tf.Summary.Value(tag='episode-success/success_count',
                                    simple_value=self.success_count)]
        )
        writer.add_summary(summary_1, self.num_rollouts)

        summary_2 = tf.Summary(
            value=[tf.Summary.Value(tag='episode-success/episode_count',
                                    simple_value=self.episode_count)]
        )
        writer.add_summary(summary_2, self.num_rollouts)

        success_mean = 0
        if self.episode_count > 0:
            success_mean = self.success_count / self.episode_count
        summary_3 = tf.Summary(
            value=[tf.Summary.Value(tag='episode-success/success_mean',
                                    simple_value=success_mean)]
        )
        writer.add_summary(summary_3, self.num_rollouts)

        self.rollout_idx += 1
        self.train_stats_by_rollout_idx[self.rollout_idx] = dict(
            episode_count=self.episode_count,
            success_count=self.success_count,
            success_mean=success_mean,
        )

        print(f"Rollout end ep_count: {self.episode_count}, \t "
              f"succ_count: {self.success_count}, \t "
              f"mean: {success_mean:.4f}")

        if self._train_in_multi_env:
            success_count_by_env_type = {}
            episode_count_by_env_type = {}
            wall_config_by_env_type = {}
            for env_idx, env_type in enumerate(self.training_env._env_type_by_env_idx):
                if success_count_by_env_type.get(env_type) is None:
                    success_count_by_env_type[env_type] = 0
                    episode_count_by_env_type[env_type] = 0
                    wall_config_by_env_type[env_type] = self.training_env.get_wall_config_name(env_idx)
                episode_count_by_env_type[env_type] += self.episode_count_by_env_idx[env_idx]
                success_count_by_env_type[env_type] += self.success_count_by_env_idx[env_idx]

            self.train_stats_by_rollout_idx[self.rollout_idx].update(dict(
                episode_count_by_env_type=episode_count_by_env_type,
                success_count_by_env_type=success_count_by_env_type,
                wall_config_by_env_type=wall_config_by_env_type,
            ))
            # Save for later
            self.episode_count_by_env_type = copy.deepcopy(episode_count_by_env_type)
            self.success_count_by_env_type = copy.deepcopy(success_count_by_env_type)
            self.wall_config_by_env_type = copy.deepcopy(wall_config_by_env_type)

            for env_type in episode_count_by_env_type.keys():
                ep_count = episode_count_by_env_type[env_type]
                success_count = success_count_by_env_type[env_type]
                env_name = wall_config_by_env_type[env_type]
                summary_1 = tf.Summary(
                    value=[tf.Summary.Value(tag=f'episode-success/{env_name}/success_count',
                                            simple_value=success_count)]
                )
                writer.add_summary(summary_1, self.num_rollouts)

                summary_2 = tf.Summary(
                    value=[tf.Summary.Value(tag=f'episode-success/{env_name}/episode_count',
                                            simple_value=ep_count)]
                )
                writer.add_summary(summary_2, self.num_rollouts)

                success_mean = 0
                if ep_count > 0:
                    success_mean = float(success_count) / ep_count
                summary_3 = tf.Summary(
                    value=[tf.Summary.Value(tag=f'episode-success/{env_name}/success_mean',
                                            simple_value=success_mean)]
                )
                writer.add_summary(summary_3, self.num_rollouts)

                print(f"\t \t env: {env_name} ep_count: {ep_count}, \t "
                      f"succ_count: {success_count}, \t "
                      f"mean: {success_mean:.4f}")

        pkl_path = os.path.join(self.results_save_dir, 'ppo_agent_success.pkl')
        with open(pkl_path, 'wb') as pkl_f:
            pickle.dump(self.train_stats_by_rollout_idx, pkl_f, protocol=2)
            print(f"Did save ppo_agent_success: {pkl_path}")


class EnvRenderCallback(BaseCallback):

    def __init__(self, fn_cb, verbose=0):
        super(EnvRenderCallback, self).__init__(verbose=verbose)
        self._fn_cb = fn_cb

    def on_step(self) -> bool:
        self._fn_cb(None, None)
        return True


class SaveTFModelCallback(BaseCallback):

    def __init__(self, save_dir: str, verbose: int = 0):
        super(SaveTFModelCallback, self).__init__(verbose=verbose)
        self.save_dir = save_dir
        assert os.path.exists(save_dir), "Dir to save tf models does not exist"

    def on_step(self) -> bool:
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps

        save_path = os.path.join(self.save_dir, f'cp_{self.num_timesteps}.cpkt')

        # Save the TF model
        parameters = self.model.get_parameters()
        self.model._save_to_file(save_path, data=None, params=parameters,
                                 cloudpickle=False)
        print(f"Did save model: {save_path}")
        return True
