import argparse
import random
import numpy as np
from autolab_core import YamlConfig
import os
import logging
import copy
import time
import torch

import tensorflow as tf

from carbongym_utils.draw import draw_transforms
from stable_baselines.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines.common.evaluation import evaluate_policy


from object_axes_ctrlrs.exp.callbacks import (PPOTensorboardCallback, DqnTensorboardCallback,
                                   EnvRenderCallback, SaveTFModelCallback)
from object_axes_ctrlrs.exp.exp_utils import (str2bool, create_env_with_klass,
                                   get_ppo_clipfrac_exp_schedule, get_ppo_clipfrac_linear_schedule,
                                   create_model, load_cfg_for_model_cp,
                                   verify_current_cfg_with_model_cp_cfg, path_to_logdir,
                                   reset_config_to_test_mode)
from object_axes_ctrlrs.exp.stable_baseline_envs import create_franka_door_open_env as create_env
from carbongym_utils.math_utils import transform_to_np, vec3_to_np


class FrankaDoorPPOTensorboardCallback(PPOTensorboardCallback):

    def __init__(self, results_save_dir, n_envs, train_in_multi_env, verbose=0):
        super(FrankaDoorPPOTensorboardCallback, self).__init__(
            results_save_dir, n_envs, train_in_multi_env, verbose)
    
    def _on_rollout_end(self):
        super()._on_rollout_end()
        
        writer = self.locals['writer']
        env = self.training_env
        print("Separate Rewards:")
        for rew_key, rew_value in env._log_debug_rews_dict.items():
            if len(rew_value) > 0:
                summary_1 = tf.Summary(
                    value=[tf.Summary.Value(tag=f'episode-reward-info/{rew_key}-mean', 
                                            simple_value=np.mean(rew_value))]
                )
                writer.add_summary(summary_1, self.num_rollouts)
                summary_2 = tf.Summary(
                    value=[tf.Summary.Value(tag=f'episode-reward-info/{rew_key}-std', 
                                            simple_value=np.std(rew_value))]
                )
                writer.add_summary(summary_2, self.num_rollouts)
                print(f" \t \t {rew_key}: \t   {np.mean(rew_value):.4f}, {np.std(rew_value):.4f}")
        
        for rew_key in env._log_debug_rews_dict.keys():
            if len(env._log_debug_rews_dict[rew_key]) > 0:
                env._log_debug_rews_dict[rew_key] = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, 
                        help='Set seed to control stochasticity.')
    parser.add_argument('--cfg', '-c', type=str, default='cfg/franka_door_opening/train_franka_door_open.yaml')
    parser.add_argument('--logdir', '-l', type=str, default='outs/tb')
    parser.add_argument('--model_cp', type=str, default='', 
                        help='Path to saved model params')
    parser.add_argument('--train', type=str2bool, default=True, 
                        help='Train model or test provided model cp')
    parser.add_argument('--save_obs_data', type=str2bool, default=False, 
                        help='Save interaction data.')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # np.seterr(all='raise')

    print(f"Using cfg path: {args.cfg}")
    cfg = YamlConfig(args.cfg)
    if not args.train:
        cfg = reset_config_to_test_mode(cfg)
        print(f"Will test model with cp: {args.model_cp}")
    
    colors = np.random.rand(10, 3).astype(np.float32)

    # Some custom drawings on env for debugging. Can be safely disabled (ignored).
    def custom_draw_lines(scene):
        door_handle_transforms = vec_env.get_door_handle_transform()
        for env_idx, env_ptr in enumerate(scene.env_ptrs):
            door_kps = vec_env._door_keypoints[env_idx]
            door_tf = door_handle_transforms[env_idx]
            door_pos = vec3_to_np(door_tf.p)
            final_door_kps = door_kps + door_pos
            # verts = final_door_kps[[0, 2, 3, 5, 6, 8]].astype(np.float32)
            verts = final_door_kps[[0, 2, 3, 5, 6, 8]].astype(np.float32)
            # print(np.array_str(verts, suppress_small=True, precision=3, max_line_width=120))
            num_lines = verts.shape[0] // 2

            scene._gym.add_lines(scene._viewer, env_ptr, num_lines, verts, colors[:num_lines])

    def custom_draws(scene):
        for env_idx, env_ptr in enumerate(scene.env_ptrs):
            ee_transform = vec_env._frankas[env_idx].get_ee_transform(env_ptr, 'franka0')
            desired_ee_transform = vec_env._frankas[env_idx].get_desired_ee_transform(env_idx, 'franka0')
            # door_ah = scene.ah_map[env_idx]['door0']
            # handle_idx = vec_env._door.rb_names_map['door_handle']
            # handle_transform = vec_env._door.get_rb_transforms(env_ptr, door_ah)[handle_idx]

            transforms = [ee_transform, desired_ee_transform]
            if 'hand_cam0' in scene.ch_map[env_idx]:
                ch = scene.ch_map[env_idx]['hand_cam0']
                cam_transform = vec_env._cam_objects['hand_cam0'].get_transform(ch, env_idx)
                # transforms.append(cam_transform)
                transforms = [cam_transform]
            
            door_name = vec_env._door_names[env_idx]
            door_ah = vec_env._scene.ah_map[env_idx][door_name]
            door_asset = vec_env._doors[env_idx]
            pole_idx = door_asset.rb_names_map['door_pole']
            axel_idx = door_asset.rb_names_map['door_handle_axel']
            handle_idx = door_asset.rb_names_map['door_handle']
            pole_tf = door_asset.get_rb_transforms(env_ptr, door_ah)[pole_idx]
            axel_tf = door_asset.get_rb_transforms(env_ptr, door_ah)[axel_idx]
            handle_transform = door_asset.get_rb_transforms(env_ptr, door_ah)[handle_idx]
            transforms += [pole_tf, axel_tf, handle_transform]

            draw_transforms(scene.gym, scene.viewer, [env_ptr], transforms)

    step = [0, 0]
    def inter_step_cb(vec_env, t_inter_step, n_inter_steps):
        vec_env.render(custom_draws=custom_draws, custom_lines=custom_draw_lines)

    vec_env = create_env(args, cfg, inter_step_cb)
    print("Did create env")
    
    save_cpkt_dir = os.path.join(args.logdir, 'models')
    if not os.path.exists(save_cpkt_dir):
        os.makedirs(save_cpkt_dir)

    save_model_cb = EveryNTimesteps(
        1000 * cfg['task']['max_steps'],
        SaveTFModelCallback(save_cpkt_dir, verbose=0))

    train_info_results_dir = os.path.join(args.logdir, 'train_info')
    if not os.path.exists(train_info_results_dir):
        os.makedirs(train_info_results_dir)
    
    callback_list = [save_model_cb]
    if cfg['rl']['algo'] in ('dqn', 'vec_dqn'):
        tb_callback = DqnTensorboardCallback()
        callback_list.append(tb_callback)
    elif cfg['rl']['algo'] in ('ppo'):
        tb_callback =  FrankaDoorPPOTensorboardCallback(
            train_info_results_dir,
            cfg['scene']['n_envs'],
            cfg['task'].get('train_in_multi_env', False),
        )
        callback_list.append(tb_callback)

    model = create_model(args, vec_env, cfg)

    if args.train:
        save_cfg_path = os.path.join(train_info_results_dir, 'train_cfg.yaml')
        import shutil
        # Save file's other metadata including modtime etc.
        shutil.copy2(args.cfg, save_cfg_path)

        save_cfg_path = os.path.join(train_info_results_dir, 'train_cfg_exec.yaml')
        cfg.save(save_cfg_path)
        print(f"Did save train cfg: {save_cfg_path}")

        if len(args.model_cp) > 0:
            assert os.path.exists(args.model_cp), f"Model cp does not exist: {args.model_cp}"
            model.load_parameters(args.model_cp)

        model.learn(
            total_timesteps=int(cfg['rl']['total_timesteps']),
            callback=callback_list,
            log_interval=1,
            reset_num_timesteps=False)
    else:

        if cfg['task']['evaluate_verify_cfg']:
            old_cfg, old_cfg_path = load_cfg_for_model_cp(args.model_cp, cfg)
            train_test_cfg_similar = verify_current_cfg_with_model_cp_cfg(
                cfg, old_cfg)
            if not train_test_cfg_similar:
                raise ValueError("Train and test config are not similar\n"
                                 f"     Train config: \t {old_cfg_path}\n"
                                 f"      Test config: \t {args.cfg}\n")
            else:
                print("Train and test config are similar\n"
                      f"     Train config: \t {old_cfg_path}\n"
                      f"      Test config: \t {args.cfg}\n")


        assert os.path.exists(args.model_cp), f"Test model cp does not exist: {args.model_cp}"
        model.load_parameters(args.model_cp)

        if cfg['task']['evaluate_multi_envs']:
            envs_to_evaluate = copy.deepcopy(cfg['walls']['multi_env_types'])
        else:
            envs_to_evaluate = [cfg['walls']['type']]
        
        for wall_config_name in envs_to_evaluate:
            vec_env.close()
            time.sleep(4.0)
            # Load old config and set the right env.
            test_cfg = YamlConfig(args.cfg)
            test_cfg = reset_config_to_test_mode(test_cfg)
            test_cfg['walls']['type'] = wall_config_name
            # Close the old env.
            vec_env = create_env(args, test_cfg, inter_step_cb)
            model.env = vec_env

            episode_rewards, episode_lengths = evaluate_policy(
                model, 
                vec_env, 
                n_eval_episodes=5,
                deterministic=True,
                # callback=learn_cb,
                return_episode_rewards=True
            )
            mean_rewards, std_rewards = np.mean(episode_rewards), np.std(episode_rewards)
            mean_length, std_length = np.mean(episode_lengths), np.std(episode_lengths)

            print(f"Did evaluate model:  {args.model_cp} \n"
                f"     for env:          {wall_config_name}\n"
                f" Reward Mean:          {mean_rewards:.4f}\n"
                f"  Reward std:          {std_rewards:.4f}\n"
                f" Length mean:          {mean_length:.4f}\n"
                f"  Length std:          {std_length:.4f}\n"
            )
