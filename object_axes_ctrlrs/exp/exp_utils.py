import numpy as np
from autolab_core import YamlConfig
import os
import copy
import argparse

from object_axes_ctrlrs.envs.env_wrappers import (SaveInteractionEnvWrapper,
                                       SaveInteractionVecEnvWrapper,
                                       ExpandMDPVecEnvWrapper)
from object_axes_ctrlrs.policies import ControllerFeaturesMultiNetworkScoresPolicy, \
                            ControllerFeaturesIndependentScoresPolicy, \
                            ControllerFeaturesSingleNetworkDiscretePolicy
from object_axes_ctrlrs.envs.block2d import GymBlock2DVecEnv


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise ValueError("Invalid")


def create_env_with_klass(klass, args, cfg, inter_step_cb, separate_policies=False, all_models=None):
    use_expand_MDP_wrapper = True

    if use_expand_MDP_wrapper:
        expand_MDP = cfg['task']['expand_MDP']
        # Set expand MDP to false within the vec env
        cfg['task']['expand_MDP'] = False

    vec_env = klass(cfg, inter_step_cb=inter_step_cb, **cfg['env'])
    vec_env.seed(args.seed)

    if use_expand_MDP_wrapper and expand_MDP:
        vec_env = ExpandMDPVecEnvWrapper(
            vec_env, 3, cfg['rews']['expand_MDP_same_controller_reward'],
            controller_obs_type=cfg['task']['expand_MDP_obs_type'])
        vec_env.seed(args.seed)
    osp = os.path
    if args.model_cp is not None and len(args.model_cp) > 0:
        # basename will be "cp_{}.cpkt". Remove ".cpkt"
        cp_name = osp.basename(args.model_cp)[:-5]
        if 'walls' in cfg:
            env_name = cfg['walls']['type']
            cp_name += f'_{env_name}'

    if not args.train:
        main_result_dir = osp.abspath(os.path.join(args.model_cp, '../..'))
        eval_results_dir = osp.join(main_result_dir, f'result_{cp_name}')
        vec_env = SaveInteractionEnvWrapper(vec_env, eval_results_dir)
    else:
        if args.save_obs_data:
            eval_results_dir = osp.join(args.save_obs_data_path , 'results_obs_data')
            vec_env = SaveInteractionVecEnvWrapper(vec_env, eval_results_dir)

    return vec_env

def get_ppo_clipfrac_exp_schedule(cfg):
    clip_start = cfg['rl']['ppo']['cliprange']
    def ppo_clipfrac_schedule(frac):
        '''frac: Float. Goes from 1 to 0 over the entire training schedule.'''
        return clip_start * np.exp((frac - 1.) * 1.5)

    return ppo_clipfrac_schedule


def get_ppo_clipfrac_linear_schedule(cfg):
    clip_start = cfg['rl']['ppo']['cliprange']
    clip_end = 0.04
    clip_diff = clip_end - clip_start
    def ppo_clipfrac_schedule(frac):
        '''frac: Float. Goes from 1 to 0 over the entire training schedule.'''
        return clip_start + (1.0 - frac) * clip_diff

    return ppo_clipfrac_schedule


def create_model(args, vec_env, cfg, logdir=None, index_to_learn=None):
    action_type = cfg['action']['type']
    algo = cfg['rl']['algo']
    if logdir is None:
        logdir = args.logdir
    if algo == 'ppo':
        ppo_args = copy.deepcopy(cfg['rl']['ppo'])
        cliprange = get_ppo_clipfrac_linear_schedule(cfg)
        ppo_args['cliprange'] = cliprange
        if index_to_learn is not None:
            from stable_baselines_custom.ppo2 import PPO2
            from stable_baselines_custom.common.policies import MlpPolicy
        else:
            from stable_baselines.ppo2 import PPO2
            from stable_baselines.common.policies import MlpPolicy

        if cfg['task']['use_controller_features']:
            if cfg['action']['type'] == 'discrete_one_controller':
                if isinstance(vec_env, ExpandMDPVecEnvWrapper):
                    assert cfg['task']['expand_MDP_obs_type'] == 'features'
                    Policy = ControllerFeaturesMultiNetworkScoresPolicy
                    ppo_args['policy_kwargs'] = {
                        'N_action_steps': vec_env.num_steps,
                        'N_obs_dim': vec_env.venv.n_obs_wo_ctrlr,
                        'N_ctrlrs': vec_env.venv.n_low_level_controllers,
                        'N_ctrlr_dim': vec_env.venv.n_ctrlr_features
                    }
                    if isinstance(vec_env.venv, GymBlock2DVecEnv):
                        ppo_args['policy_kwargs']['fuse_backbones'] = True
                    else:
                        ppo_args['policy_kwargs']['fuse_backbones'] = False
                else:
                    Policy = ControllerFeaturesSingleNetworkDiscretePolicy
                    ppo_args['policy_kwargs'] = {
                        'N_obs_dim': vec_env.n_obs_wo_ctrlr,
                        'N_ctrlrs': vec_env.n_low_level_controllers,
                        'N_ctrlr_dim': vec_env.n_ctrlr_features
                    }
            elif cfg['action']['type'] == 'priority':
                assert not cfg['task']['expand_MDP'], \
                    'Must not use expand_MDP when using controller features and priority'
                Policy = ControllerFeaturesIndependentScoresPolicy
                ppo_args['policy_kwargs'] = {
                    'N_obs_dim': vec_env.n_obs_wo_ctrlr,
                    'N_ctrlrs': vec_env.n_low_level_controllers,
                    'N_ctrlr_dim': vec_env.n_ctrlr_features
                }
        else:
            Policy = MlpPolicy

        model = PPO2(
            Policy,
            env=vec_env,
            verbose=1,
            tensorboard_log=logdir,
            **ppo_args)
        model.n_envs = cfg["scene"]["n_envs"]
    else:
        raise ValueError(f"RL model {algo} not impl. for action_type: {action_type}")

    model.set_random_seed(args.seed)

    return model


def load_cfg_for_model_cp(model_cp_path, cfg):
    osp = os.path
    main_result_dir = osp.abspath(os.path.join(model_cp_path, '../..'))
    model_cp_cfg_path = osp.join(main_result_dir, 'train_cfg.yaml')
    assert os.path.exists(model_cp_cfg_path), \
        f"Model cp config does not exist: {model_cp_cfg_path}"

    model_cp_cfg = YamlConfig(model_cp_cfg_path)
    return model_cp_cfg, model_cp_cfg_path

def path_to_logdir(args, idx=None):
    if idx is None:
        return args.logdir
    else:
        return "{}_{}".format(args.logdir, idx)


def reset_config_to_test_mode(cfg):
    cfg['scene']['n_envs'] = 1
    cfg['scene']['gui'] = 1
    cfg['task']['train_in_multi_env'] = False
    return cfg
