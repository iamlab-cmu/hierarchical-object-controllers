from abc import ABC, abstractmethod
import numpy as np
from itertools import permutations
from collections import deque, OrderedDict

from isaac_gym import gymapi
from isaac_gym_utils.assets import GymBoxAsset
from isaac_gym_utils.math_utils import (rpy_to_quat, quat_to_rot, rot_from_np_quat,
                                        transform_to_RigidTransform, rot_to_quat,
                                        transform_to_np, vec3_to_np, slerp_quat, np_to_vec3)
from isaac_gym_utils.rl import GymVecEnv
from gym.spaces import Box, Discrete

from object_axes_ctrlrs.controllers.projected_axes_attractors import \
    (PositionAttractorController, RotationAttractorController, 
     ForceAttractorController, NullAttractorController,
     ComposedAttractorControllers, ErrorAxisPositionAttractorController,
     CurlPositionAttractorController, PositionAxisAttractorController, 
     ControllersListFeatureExtractor)

from object_axes_ctrlrs.utils.math_utils import (points_on_same_side, angle_between_axis,
                                      check_underflow)


class GymBlock2DVecEnv(GymVecEnv, ABC):

    @property
    def is_2d(self):
        return True

    @staticmethod
    def get_xz_positions_for_segment(size, theta, origin=(0, 0)):
        '''In Isaac x points towards the y-axis and z towards x-axis.'''
        theta_rad = np.deg2rad(theta)
        pos_from_origin = size * np.cos(theta_rad), -size * np.sin(theta_rad)
        world_pos = np.array([origin[i] + pos_from_origin[i] for i in range(2)])
        world_end_pos = np.array([origin[i] + 2 * pos_from_origin[i] for i in range(2)])
        return world_pos, world_end_pos

    def _parse_wall_object_position_properties(self, cfg, parse_multi_configs=False):
        '''Get wall positions and orientations.'''
        if parse_multi_configs:
            wall_config_names = cfg['walls']['multi_env_types']
        else:
            wall_config_names = [cfg['walls']['type']]

        self._wall_half_sizes_dict = {}
        self._wall_pos_dict = {}
        self._wall_end_dict = {}
        self._wall_half_width_dict = {}
        self._wall_angles_dict = {}
        self._wall_corners_dict = {}

        for wall_config_name in wall_config_names:
            self._wall_half_width_dict[wall_config_name] = \
                cfg['walls'][wall_config_name]['half_width']
            self._wall_angles_dict[wall_config_name] = \
                cfg['walls'][wall_config_name]['wall_angles']
            self._wall_half_sizes_dict[wall_config_name] = {}
            self._wall_pos_dict[wall_config_name] = {}
            self._wall_end_dict[wall_config_name] = {}
            
            wall_config = cfg['walls'][wall_config_name]
            wall_names = ['wall{}'.format(i) for i in range(wall_config['n_walls'])]

            wall_end = cfg['walls'][wall_config_name]['origin']
            self._wall_corners_dict[wall_config_name] = [wall_end]
            for wall_name in wall_names:
                half_width = self._wall_half_width_dict[wall_config_name][wall_name]
                self._wall_half_sizes_dict[wall_config_name][wall_name] = np.array([half_width, 0, cfg['wall']['dims']['depth'] / 2])
                angle = self._wall_angles_dict[wall_config_name][wall_name]

                wall_pos_xz, wall_end = GymBlock2DVecEnv.get_xz_positions_for_segment(
                    half_width, angle, wall_end)
                self._wall_pos_dict[wall_config_name][wall_name] = wall_pos_xz
                self._wall_end_dict[wall_config_name][wall_name] = wall_end
                self._wall_corners_dict[wall_config_name].append(wall_end)
            self._wall_corners_dict[wall_config_name] = np.array(self._wall_corners_dict[wall_config_name])
            
    def _fill_scene(self, cfg):
        self._train_in_multi_env = cfg['task'].get('train_in_multi_env', False)
        self._parse_wall_object_position_properties(
            cfg, parse_multi_configs=self._train_in_multi_env)

        if self._train_in_multi_env:
            self._multi_env_types = cfg['walls']['multi_env_types']

            num_envs = cfg['scene']['n_envs']
            assert num_envs % len(self._multi_env_types) == 0
            envs_by_type = num_envs // len(self._multi_env_types)
            self._env_type_by_env_idx = [i for i in range(len(self._multi_env_types))
                                           for _ in range(envs_by_type)]

        self._expand_MDP = cfg['task']['expand_MDP']
        self._expand_MDP_num_action_steps = 3
        self._expand_MDP_action_as_obs_id = -1
        self._add_y_axis_to_obs = cfg['task']['add_y_axis_to_obs']
        self._min_y_bound, self._max_y_bound = -0.05, 0.1
        self._use_adaptive_env_bounds = cfg['task']['use_adaptive_env_bounds']

        self._walls = {}
        for wall_config_name, wall_config_half_width_dict in self._wall_half_width_dict.items():
            self._walls[wall_config_name] = {}
            for wall_name in wall_config_half_width_dict.keys():
                self._walls[wall_config_name][wall_name] = GymBoxAsset(
                    self._scene.gym, 
                    self._scene.sim,
                    **cfg['wall']['dims'], 
                    width=2 * wall_config_half_width_dict[wall_name],
                    shape_props=cfg['wall']['shape_props'],
                    rb_props=cfg['wall']['rb_props'],
                    asset_options=cfg['wall']['asset_options']
                )
        if not cfg['task']['train_in_multi_env'] and len(self._walls.keys()) == 1:
            self._walls = self._walls[list(self._walls.keys())[0]]

        # create block asset
        self._block = GymBoxAsset(self._scene.gym, self._scene.sim,
            **cfg['block']['dims'],
            shape_props=cfg['block']['shape_props'],
            rb_props=cfg['block']['rb_props'],
            asset_options=cfg['block']['asset_options']
        )

        # add assets to scene
        wall_y = cfg['wall']['dims']['height'] / 2
        for env_idx in range(self.n_envs):
            wall_config_name = self.get_wall_config_name(env_idx)
            wall_angles_dict = self._wall_angles_dict[wall_config_name]
            wall_pos_dict = self._wall_pos_dict[wall_config_name]

            for wall_name, wall_pos in wall_pos_dict.items():
                wall_angle = np.deg2rad(wall_angles_dict[wall_name])
                if self._train_in_multi_env:
                    wall_asset = self._walls[wall_config_name][wall_name]
                    wall_asset_name = f'{wall_config_name}_{wall_name}'
                else:
                    wall_asset = self._walls[wall_name]
                    wall_asset_name = wall_name
                self._scene.add_asset(
                    wall_asset_name, 
                    wall_asset,
                    gymapi.Transform(
                        p=gymapi.Vec3(wall_pos[0], wall_y, wall_pos[1]),
                        r=rpy_to_quat([0, wall_angle, 0])
                    ),
                    envs=[env_idx]
                    )

        # we'll sample initial block transforms in reset
        self._block_name = 'block'
        self._scene.add_asset(self._block_name, self._block, gymapi.Transform())

        # add attractors to blocks
        self._attractor_handles = []
        self._attractor_targets = []
        for env_ptr in self._scene.env_ptrs:
            attractor_props = gymapi.AttractorProperties()
            attractor_props.stiffness = cfg['action']['attractor_stiffness']
            attractor_props.damping = 2 * np.sqrt(cfg['action']['attractor_stiffness'])
            attractor_props.axes = gymapi.AXIS_ALL

            block_rigid_handle = self._scene._gym.get_rigid_handle(env_ptr, self._block_name, 'box')
            attractor_props.rigid_handle = block_rigid_handle

            attractor_handle = self._scene._gym.create_rigid_body_attractor(env_ptr, attractor_props)
            self._attractor_handles.append(attractor_handle)
            self._attractor_targets.append(gymapi.Transform()) # these will be updated in reset

        # Declare all the variables that we might need in the future
        self._max_steps = cfg['task']['max_steps']
        self._prev_dists_to_target = None

        if self._train_in_multi_env:
            self._wall_config_obs_size = len(self._walls[self._multi_env_types[0]]) * 6  
        else:
            self._wall_config_obs_size = len(self._walls) * 6  # num_walls * [wall_pos, end]
        self._prev_infos = None

        if self._expand_MDP:
            self._expand_MDP_action_step_counts = np.zeros((self.n_envs), dtype=np.int32)
            self._expand_MDP_action_step_counts_curr = np.zeros((self.n_envs), dtype=np.int32)
            self._expand_MDP_last_obs = None
            self._expand_MDP_action_deque = deque([], self._expand_MDP_num_action_steps)
            self._did_apply_action_for_env = np.zeros((self.n_envs), dtype=np.int32)

        self._block_clearance = self._cfg['block']['dims']['width'] / np.sqrt(2) + 0.01

        self.completed_eplen_by_env_idx = [deque([], 1000) for _ in range(self.n_envs)]
    
    def get_success_eplen_by_env_type(self):
        '''Calculate the eplen of successfully completed episodes.
        '''
        success_eplen_by_env_type = dict()
        for env_idx, env_type in enumerate(self._env_type_by_env_idx):
            if success_eplen_by_env_type.get(env_type) is None:
                success_eplen_by_env_type[env_type] = []
            success_eplen_by_env_type[env_type] += list(self.completed_eplen_by_env_idx[env_idx])
        return success_eplen_by_env_type
    
    def reset_success_eplen_by_env_type(self):
        '''Reset eplen of successfully completed episodes. 
        This used to check multiple checkpoints sequentially. '''
        for eplen_deque in self.completed_eplen_by_env_idx:
            eplen_deque.clear()
    

    def _make_low_level_controllers_dict_for_env(self, action_cfg, env_idx, env_ptr):
        attractors_cfg = action_cfg['attractors']
        dx = attractors_cfg['limits']['dx']
        df = attractors_cfg['limits']['df']
        dtheta = np.deg2rad(attractors_cfg['limits']['dtheta'])
        dl = attractors_cfg['limits']['dl']

        low_level_ctrlr_dict = OrderedDict()
        low_level_ctrlr_dict['null'] = NullAttractorController()

        wall_assets_dict = self.get_wall_assets_dict(env_idx)
        wall_config_name = self.get_wall_config_name(env_idx)

        self._combo_wall_specific_ctrlrs = []

        for wall_name, wall_asset in wall_assets_dict.items():
            wall_asset_name = self.get_wall_asset_name(env_idx, wall_name)
            wall_ah = self._scene.ah_map[env_idx][wall_asset_name]
            wall_tf = wall_asset.get_rb_transforms(env_ptr, wall_ah)[0]

            wall_R = quat_to_rot(wall_tf.r)
            wall_tra = vec3_to_np(wall_tf.p)
            wall_normal = wall_R[:, 2]
            wall_end_dict = self._wall_end_dict[wall_config_name]
            wall_corner = np.array(
                [wall_end_dict[wall_name][0], 0.03, wall_end_dict[wall_name][1]])
            
            attractors_idx_by_name = OrderedDict()
            # pulls obj toward wall in normal direction
            if attractors_cfg['use']['pos_normal']:
                low_level_ctrlr_dict['{}/pos_normal'.format(wall_name)] = \
                    PositionAttractorController(wall_normal, wall_tra, dx)
                attractors_idx_by_name['pos_normal'] = len(low_level_ctrlr_dict) - 1
            # pulls obj toward wall in err direction
            if attractors_cfg['use']['pos_err']:
                low_level_ctrlr_dict['{}/pos_err'.format(wall_name)] = \
                    ErrorAxisPositionAttractorController(wall_tra, dx)
                attractors_idx_by_name['pos_err'] = len(low_level_ctrlr_dict) - 1
            # tries to align obj's z-axis w/ wall_normal
            if attractors_cfg['use']['rot_z']:
                low_level_ctrlr_dict['{}/rot_z'.format(wall_name)] = \
                    RotationAttractorController(np.array([0, 0, 1]), wall_normal, dtheta)
                attractors_idx_by_name['rot_z'] = len(low_level_ctrlr_dict) - 1
            # tries to align obj's x-axis w/ wall_normal
            if attractors_cfg['use']['rot_x']:
                low_level_ctrlr_dict['{}/rot_x'.format(wall_name)] = \
                    RotationAttractorController(np.array([1, 0, 0]), wall_normal, dtheta)
                attractors_idx_by_name['rot_x'] = len(low_level_ctrlr_dict) -1
            # tries to exert a constant force into a wall
            if attractors_cfg['use']['force']:
                wall_force = 10 * wall_normal
                low_level_ctrlr_dict['{}/force'.format(wall_name)] = \
                    ForceAttractorController(wall_normal, wall_force, df)
                attractors_idx_by_name['force'] = len(low_level_ctrlr_dict) - 1 
            # rotates obj in global frame around wall end corner
            if attractors_cfg['use']['pos_curl']:
                low_level_ctrlr_dict['{}/pos_curl'.format(wall_name)] = \
                    CurlPositionAttractorController(-wall_normal, wall_corner, dl)
                attractors_idx_by_name['pos_curl'] = len(low_level_ctrlr_dict) - 1
            # pulls obj toward wall end corner
            if attractors_cfg['use']['pos_corner']:
                low_level_ctrlr_dict['{}/pos_corner'.format(wall_name)] = \
                    ErrorAxisPositionAttractorController(wall_corner, dx)
                attractors_idx_by_name['pos_corner'] = len(low_level_ctrlr_dict) - 1 
            if attractors_cfg['use']['pos_side']:
                wall_perp = np.cross(wall_normal, np.array([0, 1, 0]))
                low_level_ctrlr_dict['{}/pos_side'.format(wall_name)] = \
                    PositionAxisAttractorController(wall_perp, dx/5)
                attractors_idx_by_name['pos_side'] = len(low_level_ctrlr_dict) - 1

        return low_level_ctrlr_dict

    def _init_controllers(self, cfg):
        '''Initialize the high level and low level controllers.'''
        self._high_level_controllers = []
        self._controller_feature_extractor = []
        for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
            low_level_ctrlr_dict = self._make_low_level_controllers_dict_for_env(cfg['action'], env_idx, env_ptr)
            high_level_ctrlr = ComposedAttractorControllers(low_level_ctrlr_dict)
            self._high_level_controllers.append(high_level_ctrlr)
            self._controller_feature_extractor.append(ControllersListFeatureExtractor(
                high_level_ctrlr.low_level_controllers, is_2d=True
            ))        
    
    @property
    def last_infos(self):
        return self._prev_infos
    
    def get_wall_pose(self, env_idx, wall_name):
        wall_asset = self.get_wall_asset(env_idx, wall_name)
        wall_asset_name = self.get_wall_asset_name(env_idx, wall_name)
        wall_ah = self._scene.ah_map[env_idx][wall_asset_name]
        env_ptr = self._scene.env_ptrs[env_idx]
        wall_tf = wall_asset.get_rb_transforms(env_ptr, wall_ah)[0]
        wall_R = quat_to_rot(wall_tf.r)
        wall_tra = vec3_to_np(wall_tf.p)
        return wall_tra, wall_R
    
    def get_wall_asset(self, env_idx, wall_name):
        wall_assets_dict = self.get_wall_assets_dict(env_idx)
        return wall_assets_dict[wall_name]
    
    def get_wall_assets_dict(self, env_idx):
        if self._train_in_multi_env:
            env_type = self._env_type_by_env_idx[env_idx]
            wall_config_name = self._multi_env_types[env_type]
            wall_assets_dict = self._walls[wall_config_name]
        else:
            wall_assets_dict = self._walls
        return wall_assets_dict
    
    def get_wall_config_name(self, env_idx):
        if self._train_in_multi_env:
            env_type = self._env_type_by_env_idx[env_idx]
            return self._multi_env_types[env_type]
        else:
            return self._cfg['walls']['type']
    
    def get_wall_asset_name(self, env_idx, wall_name):
        if self._train_in_multi_env:
            env_type = self._env_type_by_env_idx[env_idx]
            wall_config_name = self._multi_env_types[env_type]
            return f'{wall_config_name}_{wall_name}'
        else:
            return wall_name

    def filter_invalid_controller_combos(self, combos):
        """
        :param combos list of controller combination indices
        :return list of valid controller combos that can be sampled from the action space. currently just makes sure they
        are not both position and orientation but more can be added.
        """
        valid_combos = []
        for combo in combos:
            for controller_idx in combo:
                controller = self._low_level_controller_combos[controller_idx]
                if np.sum(np.abs(controller[:3])) > 1e-6:  # positon controller
                    if np.sum(np.abs(controller[3:])) > 1e-6:
                        continue
                else:
                    if np.sum(controller[3:]) > 1e-6:
                        continue

            valid_combos.append(combo)
        return valid_combos

    def change_action_space(self, cfg):
        self._init_action_space(cfg)

    def _init_action_space(self, cfg):
        self._action_type = cfg['action']['type']
        self._auto_clip_actions = cfg['action']['auto_clip_actions']
        self._raw_translation_goals = [gymapi.Vec3() for _ in range(self.n_envs)]
        self._raw_rotation_goals = [gymapi.Quat() for _ in range(self.n_envs)]
        self._current_controller_idxs = [[] for _ in range(self.n_envs)]
        self._interp_delta_translation = [gymapi.Vec3() for _ in range(self.n_envs)]
        self._interp_delta_rotations = [gymapi.Quat() for _ in range(self.n_envs)]
        
        if self._action_type == 'raw':
            # delta x, z, theta
            limits = np.array(cfg['action']['raw']['limits'])
            limits[2] = np.deg2rad(limits[2])
            # HACK: Stupid PGGaussian in rlpyt does not seem to support this not
            # sure why.
            # limits = np.array([0.1, 0.1, 0.1])
            action_space = Box(-limits, limits, dtype=np.float32)
        elif self._action_type in ('combo', 'priority', 'discrete_one_controller'):
            self._init_controllers(cfg)
            self._n_low_level_controllers = len(self._high_level_controllers[0].low_level_controllers)
            
            if self._action_type == 'combo':
                self._low_level_controller_combos = list(permutations(
                        # 2 zeros to allow multiple no movement controllers
                        [0, 0] + list(range(self._n_low_level_controllers))
                    , 3))
                self._low_level_controller_combos = self.filter_invalid_controller_combos(
                        self._low_level_controller_combos)
                self._low_level_controller_combos_map = {combo : i for i, combo in enumerate(
                    self._low_level_controller_combos)}
                action_space = Discrete(len(self._low_level_controller_combos))
            elif self._action_type == 'priority':
                action_space = Box(np.zeros(self._n_low_level_controllers),
                                   np.ones(self._n_low_level_controllers), dtype=np.float32)
            elif self._action_type == 'discrete_one_controller':
                action_space = Discrete(self._n_low_level_controllers)
            else:
                raise ValueError(f"Invalid action type: {self._action_type}")
        else:
            raise ValueError('Unknown action type: {}'.format(self._action_type))

        return action_space
    
    def get_all_wall_corners(self):
        '''Get corners for all different envs being currently used.'''
        if self._train_in_multi_env:
            wall_corners = []
            for _, env_type in enumerate(self._env_type_by_env_idx):
                env_config = self._multi_env_types[env_type]
                wall_corners.append(self._wall_corners_dict[env_config])
        else:
            wall_config = self.get_wall_config_name(0)
            wall_corners = list(self._wall_corners_dict.values())
        return wall_corners

    def _get_adaptive_obs_bounds(self, cfg, block_size):
        wall_corners_by_env = np.array(self.get_all_wall_corners())
        min_corner = wall_corners_by_env.min(axis=1)
        max_corner = wall_corners_by_env.max(axis=1)
        min_corner -= (block_size * cfg['task']['adaptive_env_bound_block_mult'])
        max_corner += (block_size * cfg['task']['adaptive_env_bound_block_mult'])
        if self._add_y_axis_to_obs:
            min_corner = np.hstack([min_corner[:, 0:1], 
                                    np.ones((min_corner.shape[0], 1)) * self._min_y_bound,
                                    min_corner[:, 1:2]])
            max_corner = np.hstack([max_corner[:, 0:1], 
                                    np.ones((max_corner.shape[0], 1)) * self._max_y_bound,
                                    max_corner[:, 1:2]])
        return min_corner, max_corner
    
    # Can override this method in sub-class.
    def get_adaptive_obs_bounds(self, cfg):
        return self._get_adaptive_obs_bounds(cfg, block_size=self._cfg['block']['dims']['width'])

    def _add_child_obs_space(self, cfg, obs_space):
        return obs_space

    def _init_obs_space(self, cfg):
        # x, z, theta, fx, fz
        if cfg['task']['use_adaptive_env_bounds']:
            self._min_env_bounds, self._max_env_bounds = self.get_adaptive_obs_bounds(cfg)
            min_x, max_x = -3.0, 3.0
            min_z, max_z = -3.0, 3.0
            assert (self._min_env_bounds[:, 0].min() >= min_x 
                    and self._min_env_bounds[:, 0].max() <= max_x)
            z_axis = 2 if self._add_y_axis_to_obs else 1
            assert (self._min_env_bounds[:, z_axis].min() >= min_z
                    and self._min_env_bounds[:, z_axis].max() <= max_z)
        else:
            all_wall_corners = np.concatenate(self.get_all_wall_corners())
            min_x, min_z = all_wall_corners[:, 0].min(), all_wall_corners[:, 1].min()
            max_x, max_z = all_wall_corners[:, 0].max() + self._block_clearance * 4, all_wall_corners[:, 1].max() + self._block_clearance * 4

        pos_limit_low = [min_x, self._min_y_bound, min_z] \
            if self._add_y_axis_to_obs else [min_x, min_z]
        pos_limit_high = [max_x, self._max_y_bound, max_z] \
            if self._add_y_axis_to_obs else [max_x, max_z]

        limits_low = np.array(pos_limit_low + [-np.pi, -100, -100])
        limits_high = np.array(pos_limit_high + [np.pi, 100, 100])

        # For a new observation until we have selected all actions we use a 
        # default value. After we select each action we add the selected action
        # into the state space.
        if self._expand_MDP:
            expanded_MDP_limits_low = np.zeros(self._expand_MDP_num_action_steps)
            expanded_MDP_limits_high = np.ones(self._expand_MDP_num_action_steps) * \
                                        self._n_low_level_controllers
            limits_low = np.concatenate([limits_low, expanded_MDP_limits_low])
            limits_high = np.concatenate([limits_high, expanded_MDP_limits_high])

        if cfg['task']['obs_walls']:
            wall_limits_low = np.array([-2] * self._wall_config_obs_size)
            wall_limits_high = np.array([2] * self._wall_config_obs_size)

            limits_low = np.concatenate((limits_low, wall_limits_low))
            limits_high = np.concatenate((limits_high, wall_limits_high))

        obs_space = self._add_child_obs_space(cfg, Box(limits_low, limits_high, dtype=np.float32))

        self._n_obs_wo_ctrlr = len(obs_space.low)
        if cfg['task']['use_controller_features']:
            limits_low = np.concatenate([obs_space.low, np.repeat(self._controller_feature_extractor[0].low, self.n_low_level_controllers)])
            limits_high = np.concatenate([obs_space.high, np.repeat(self._controller_feature_extractor[0].high, self.n_low_level_controllers)])
            obs_space = Box(limits_low, limits_high, dtype=np.float32)
        
        return obs_space

    def controller_combo_to_idx(self, controller_idxs):
        if self._cfg['action']['type'] in ('combo'):
            return self._low_level_controller_combos_map[tuple(controller_idxs)]
        else:
            raise ValueError('Can only call this function when using combo action type!')

    @property
    def n_low_level_controllers(self):
        if self._cfg['action']['type'] not in ('combo', 'priority', 'discrete_one_controller'):
            raise ValueError('Can only access this property when using combo'
                              ' or priority action type!')
        return self._n_low_level_controllers

    @property
    def n_ctrlr_features(self):
        return self._controller_feature_extractor[0].dim

    @property
    def n_obs_wo_ctrlr(self):
        return self._n_obs_wo_ctrlr

    def _inter_step_terminate(self, all_actions, t_inter_step, n_inter_steps):
        if self._cfg['action']['termination']['on_convergence']:
            for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
                block_ah = self._scene.ah_map[env_idx][self._block_name]
                block_vel = self._block.get_rb_vels_as_np_array(env_ptr, block_ah)
                
                if not np.all(np.isclose(block_vel, 0, atol=self._cfg['action']['termination']['converge_th'])):
                    return False
            return True
        else:
            return False

    def _apply_inter_actions(self, _, t_inter_step, n_inter_steps):
        if self._action_type == 'raw':
            n_interp_steps = self._cfg['action']['raw']['n_interp_steps']
        else:
            n_interp_steps = self._cfg['action']['attractors']['n_interp_steps']

        for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
            block_ah = self._scene.ah_map[env_idx][self._block_name]
            block_transform = self._block.get_rb_transforms(env_ptr, block_ah)[0]
            
            if t_inter_step % n_interp_steps == 0:
                if self._action_type == 'raw':
                    # Linear interpolation for raw actions throughout entire duration of intermediate steps
                    # Technically this code can be outside of the loop, but putting it here makes the code shorter
                    dt = 1 / n_inter_steps
                    delta_translation = gymapi.Transform(p=self._raw_translation_goals[env_idx].p * dt)
                    delta_rotation = gymapi.Transform(r=slerp_quat(gymapi.Quat(), self._raw_rotation_goals[env_idx].r, dt))
                else:
                    current_tra = vec3_to_np(block_transform.p)
                    current_R = quat_to_rot(block_transform.r)
                    block_force = vec3_to_np(self._block.get_rb_ct_forces(env_idx, self._block_name)[0])

                    # Update wall force controller direction so it's pointing into the wall
                    high_level_ctrlr = self._high_level_controllers[env_idx]
                    wall_config_name = self.get_wall_config_name(env_idx)
                    for wall_name in self.get_wall_assets_dict(env_idx):
                        wall_pos = self._wall_pos_dict[wall_config_name][wall_name]
                        wall_block_delta = current_tra - np.array([wall_pos[0], current_tra[1], wall_pos[1]])

                        if self._cfg['action']['attractors']['use']['force']:
                            wall_force_ctrlr = high_level_ctrlr.low_level_controllers_dict['{}/force'.format(wall_name)]
                            force_mag = np.linalg.norm(wall_force_ctrlr.target)
                            force_sign = -np.sign(wall_block_delta @ wall_force_ctrlr.axis)
                            wall_force_ctrlr.target = force_mag * force_sign * wall_force_ctrlr.axis

                    # Linear interpolation for controller actions every n_interp_steps
                    dt = 1 / n_interp_steps
                    delta_translation_target, delta_rotation_target = \
                        self._high_level_controllers[env_idx].compute_controls(
                            current_tra, current_R, block_force, self._current_controller_idxs[env_idx])

                    delta_translation = gymapi.Transform(p=delta_translation_target.p * dt)
                    delta_rotation = gymapi.Transform(r=slerp_quat(gymapi.Quat(), delta_rotation_target.r, dt))

                # constrain translation and rotation to be in 2D plane
                delta_translation.p.y = 0
                delta_R = quat_to_rot(delta_rotation.r)
                if not np.allclose(delta_R[1, :], [0, 1, 0]):
                    delta_R[:, 1] = [0, 1, 0]
                    delta_R[1, :] = [0, 1, 0]
                    delta_R[:, 0] /= np.linalg.norm(delta_R[:, 0])
                    delta_R[:, 2] /= np.linalg.norm(delta_R[:, 2])
                    delta_rotation.r = rot_to_quat(delta_R)

                self._interp_delta_translation[env_idx] = delta_translation
                self._interp_delta_rotations[env_idx] = delta_rotation

            attractor_target = self._interp_delta_translation[env_idx] * \
                                block_transform * \
                                self._interp_delta_rotations[env_idx]
            self._scene._gym.set_attractor_target(env_ptr,
                                                self._attractor_handles[env_idx],
                                                attractor_target)
            self._attractor_targets[env_idx] = attractor_target

    def _apply_actions(self, all_actions):
        # print(f"Action: {all_actions[0]}")
        if self._action_type == 'raw' and self._auto_clip_actions:
            all_actions = np.clip(all_actions,
                                  self.action_space.low,
                                  self.action_space.high)

        if self._expand_MDP:
            self._did_apply_action_for_env[:] = 0
            self._expand_MDP_action_deque.append(all_actions.copy())

        for env_idx in range(self.n_envs):
            action = all_actions[env_idx]

            if self._expand_MDP:
                self._expand_MDP_action_step_counts[env_idx] += 1

                if (self._expand_MDP_action_step_counts[env_idx]
                    == self._expand_MDP_num_action_steps):
                    # we have all actions so let's take a step in the env.
                    self._current_controller_idxs[env_idx] = []
                    for i in range(-self._expand_MDP_num_action_steps, 0):
                        ctrl_idx = self._expand_MDP_action_deque[i][env_idx]
                        self._current_controller_idxs[env_idx].append(ctrl_idx)
                    self._did_apply_action_for_env[env_idx] = 1

                    action = [
                        self._expand_MDP_action_deque[-3][env_idx],
                        self._expand_MDP_action_deque[-2][env_idx],
                        self._expand_MDP_action_deque[-1][env_idx],
                    ]

                else:
                    continue

            elif self._action_type == 'raw':
                self._raw_rotation_goals[env_idx] = gymapi.Transform(r=rpy_to_quat([0, action[2], 0]))
                self._raw_translation_goals[env_idx] = gymapi.Transform(gymapi.Vec3(action[0], 0, action[1]))
            elif self._action_type in ('combo', 'priority'):
                if self._action_type == 'combo':
                    if isinstance(action, (int, np.integer)):
                        self._current_controller_idxs[env_idx] = self._low_level_controller_combos[int(action)]
                    else:
                        self._current_controller_idxs[env_idx] = action

                elif self._action_type == 'priority':
                    self._current_controller_idxs[env_idx] = np.argsort(action)[:-4:-1]
                else:
                    raise ValueError("Invalid action type")

            elif self._action_type == 'discrete_one_controller':
                if type(action) is list:
                    self._current_controller_idxs[env_idx] = action
                elif type(action) is np.ndarray:
                    self._current_controller_idxs[env_idx] = action.tolist()
                elif type(action) in (int, np.int32, np.int64):
                    self._current_controller_idxs[env_idx] = [action]
                else:
                    raise ValueError(f"Invalid action type: {type(action)}")
            else:
                raise ValueError(f'Invalid action_type: {self._action_type}')
        
        if self._expand_MDP:
            action_applied_to_num_envs = np.sum(self._did_apply_action_for_env)
            # If we have output from all the controllers for all the environments
            # the following 1st if condition will be true. This implies that we
            # are ready to step in the simulation environment. Hence we set a 
            # non-zero value to `_n_inter_steps`, this leads us to calling _scene.step()
            # on isaac simulator. 
            if action_applied_to_num_envs == self.n_envs:
                self._n_inter_steps = self._cfg['env']['n_inter_steps']
            elif action_applied_to_num_envs == 0:
                # If we do not have all the controllers, then we assume that we 
                # do not # have the controllers for any of the envs. This condition
                # is always true since we always take N=3 controller steps and 
                # our time horizon is synced with 3*N-1. Also, we additionally
                # make sure that this condition holds true in code.
                self._n_inter_steps = 0
            else:
                # If some of the envs have all controllers but others do not
                # and we step into the envs, we will be stepping into all of the envs
                # since we do not have fine-grained control on which env to step and 
                # which not. This can create problems since the underlying obs
                # might change stochastically which will make learning more difficult.
                raise ValueError(f"Did not apply action for all envs. This can break some things?")
            
            self._expand_MDP_action_step_counts_curr[:] = self._expand_MDP_action_step_counts[:]

    def _compute_infos(self, all_obs, all_actions, all_rews, all_dones):
        all_infos = []
        for env_idx, _ in enumerate(self._scene.env_ptrs):
            info_dict = {}
            if all_dones[env_idx] and self._prev_did_reach_target[env_idx]:
                info_dict['is_success'] = 1
                self.completed_eplen_by_env_idx[env_idx].append(self.step_counts[env_idx] + 1)
            else:
                info_dict['is_success'] = 0

            # PPO2 code uses the episode key for some reason to save data.
            if 'rl' in self._cfg:
                if self._cfg['rl']['algo'] == 'ppo':
                    info_dict['r'] = self.episode_rewards[env_idx] + all_rews[env_idx]
                    info_dict['l'] = self.step_counts[env_idx] + 1
                    info_dict = {'episode': info_dict}

            all_infos.append(info_dict)
        self._prev_infos = all_infos
        return all_infos

    def step_expanded_mdp(self, all_actions, update_obs_cb):
        all_obs = self._compute_obs(all_actions, is_reset=False)

        all_obs = update_obs_cb(all_obs, all_actions)

        # Now compute_rewards
        all_rews = np.zeros((self.n_envs))

        assert np.sum(self._step_counts > self._max_steps) == 0, \
            "Cannot have episodes beyond max time"
        all_dones = np.zeros((self.n_envs), dtype=np.bool)

        all_infos = self._compute_infos(all_obs, all_actions, all_rews, all_dones)

        return all_obs, all_rews, all_dones, all_infos
    
    def _add_child_obs(self, all_actions, obs, is_reset=False):
        pass

    def _compute_obs(self, all_actions, is_reset=False):
        obs = np.zeros((self.n_envs, self.obs_space.shape[0]))
        if self._expand_MDP and all_actions is not None:
            assert max(all_actions) < self._n_low_level_controllers, \
                "Action space should be equal to the number of controllers"

        for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
            block_ah = self._scene.ah_map[env_idx][self._block_name]
            block_transform = self._block.get_rb_transforms(env_ptr, block_ah)[0]
            block_force = vec3_to_np(self._block.get_rb_ct_forces(env_idx, self._block_name)[0])

            obs_idx = 0
            obs[env_idx, obs_idx] = block_transform.p.x
            obs_idx += 1
            if self._add_y_axis_to_obs:
                obs[env_idx, obs_idx] = block_transform.p.y
                obs_idx += 1
            obs[env_idx, obs_idx] = block_transform.p.z
            obs_idx += 1

            # This is rotation around Y axis
            block_R = quat_to_rot(block_transform.r)
            sin_theta, cos_theta = block_R[0, 2], block_R[2, 2]
            # arctan2 has output in the range of [-pi, pi], all other np methods
            # such as np.arcos only output in 2 quadrant i.e. [0, pi].
            # theta_2 = np.arccos(block_R[0, 0]) # block's x-axis angle from world's x axis
            sin_theta = check_underflow(sin_theta, precision=1e-8, new_value=0.0)
            theta = np.arctan2(sin_theta, cos_theta)
            obs[env_idx, obs_idx] = theta
            obs_idx += 1
            obs[env_idx, obs_idx:obs_idx+2] = block_force[[0, 2]]
            obs_idx += 2

            # Previous observations should be till this value.
            start_obs_idx_for_expand_MDP = obs_idx

            if self._expand_MDP:
                self._expand_MDP_action_step_counts[env_idx] %= self._expand_MDP_num_action_steps

                num_action_steps = self._expand_MDP_action_step_counts[env_idx]
                # 0, 1, 2 are used for block position and orientation.
                col_idx = start_obs_idx_for_expand_MDP
                obs[env_idx, col_idx:col_idx+ self._expand_MDP_num_action_steps] = \
                    self._expand_MDP_action_as_obs_id
                # Offset by 1 since if we use 0 the input weights for this will
                # not really have any effect.

                # So some enviroment was reset if all_actions is None
                if all_actions is None:
                    reset_env_idxs = np.where(self.step_counts == 0)[0].tolist()
                    assert len(reset_env_idxs) > 0, "No actions and no envs to reset"
                    # This environment was just reset
                    if env_idx in reset_env_idxs:
                        continue
                    elif len(self._expand_MDP_action_deque) == 0:
                        continue
                    else:
                        last_action = self._expand_MDP_action_deque[-1]
                else:
                    last_action = all_actions
                    
                action_idx = last_action[env_idx] + 1
                if num_action_steps == 0:
                    pass
                elif num_action_steps == 1:
                    obs[env_idx, start_obs_idx_for_expand_MDP] = action_idx
                elif num_action_steps == 2:
                    obs[env_idx, start_obs_idx_for_expand_MDP] = \
                        self._expand_MDP_last_obs[env_idx, start_obs_idx_for_expand_MDP]
                    obs[env_idx, start_obs_idx_for_expand_MDP+1] = action_idx
                elif num_action_steps == 3:
                    obs[env_idx, start_obs_idx_for_expand_MDP] = \
                        self._expand_MDP_last_obs[env_idx, start_obs_idx_for_expand_MDP]
                    obs[env_idx, start_obs_idx_for_expand_MDP+1] = \
                        self._expand_MDP_last_obs[env_idx, start_obs_idx_for_expand_MDP+1]
                    obs[env_idx, start_obs_idx_for_expand_MDP+2] = action_idx
                else:
                    raise ValueError(
                        f"env: {env_idx} "
                        f"Invalid number of action steps: {num_action_steps}")

        if self._expand_MDP:
            obs_idx += self._expand_MDP_num_action_steps
            self._expand_MDP_last_obs = np.copy(obs)

        if self._cfg['task']['obs_walls']:
            start_wall_obs_idx = obs_idx
            for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
                wall_config_name = self.get_wall_config_name(env_idx)
                wall_assets_dict = self.get_wall_assets_dict(env_idx)

                for wall_i, (wall_name, wall_asset) in enumerate(wall_assets_dict.items()):
                    wall_asset_name = self.get_wall_asset_name(env_idx, wall_name)
                    wall_ah = self._scene.ah_map[env_idx][wall_asset_name]

                    # Returns a 2d array -- take it's 1st element only
                    wall_pose = wall_asset.get_rb_poses_as_np_array(env_ptr, wall_ah)[0]
                    wall_rot = rot_from_np_quat(wall_pose[3:7])
                    
                    wall_half_size_transf = wall_rot @ self._wall_half_sizes_dict[wall_config_name][wall_name]

                    obs_st_idx = start_wall_obs_idx + wall_i * 6
                    obs_end_idx = start_wall_obs_idx + (wall_i + 1) * 6
                    obs[env_idx, obs_st_idx:obs_end_idx] = [
                        wall_pose[0], wall_pose[2],
                        wall_pose[0] + wall_half_size_transf[0], 
                        wall_pose[2] + wall_half_size_transf[2],
                        wall_pose[0] - wall_half_size_transf[0], 
                        wall_pose[2] - wall_half_size_transf[2],
                    ]

        self._add_child_obs(all_actions, obs, is_reset=is_reset)

        # Add all controller features
        if self._cfg['task']['use_controller_features']:
            for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
                block_ah = self._scene.ah_map[env_idx][self._block_name]
                block_transform = self._block.get_rb_transforms(env_ptr, block_ah)[0]
                current_tra = vec3_to_np(block_transform.p)
                current_R = quat_to_rot(block_transform.r)
                current_force = vec3_to_np(self._block.get_rb_ct_forces(env_idx, self._block_name)[0])

                ctrlr_features = self._controller_feature_extractor[env_idx] \
                                    .get_features(current_tra, current_R, current_force)

                ctrl_feat_start_idx = self.n_obs_wo_ctrlr
                ctrl_feat_end_idx = ctrl_feat_start_idx + ctrlr_features.size
                obs[env_idx, ctrl_feat_start_idx:ctrl_feat_end_idx] = ctrlr_features.flatten()

        return obs
    
    def _compute_dones_for_expanded_mdp(self, all_obs, all_actions, all_rews):
        '''Compute if the expanded MDP is done just with respect to time.'''
        assert self._expand_MDP, "Only call this function for expanded MDPs"
        assert np.sum(self._step_counts > self._max_steps) == 0, \
            "Cannot have episodes beyond max time"
        # Make sure that envs that finish their time have all their actions,
        # so that we can reset these envs.
        assert np.all(self._expand_MDP_action_step_counts_curr[
            self._step_counts == self._max_steps] == self._expand_MDP_num_action_steps)
        all_action_dones = (self._expand_MDP_action_step_counts_curr == self._expand_MDP_num_action_steps)
        return all_action_dones

    def _compute_dones(self, all_obs, all_actions, all_rews):
        # done if block is out of bounds
        pos_range = 3 if self._add_y_axis_to_obs else 2
        if self._use_adaptive_env_bounds:
            out_of_bound_dones = np.any(
                all_obs[:, :pos_range] < self._min_env_bounds , axis=1) \
                | np.any(all_obs[:, :pos_range] > self._max_env_bounds, axis=1)
        else:
            out_of_bound_dones = np.any(
                all_obs[:, :pos_range] < self.obs_space.low[:pos_range], axis=1) \
                | np.any(all_obs[:, :pos_range] > self.obs_space.high[:pos_range], axis=1)
            
        time_dones = self._step_counts >= self._max_steps

        if self._expand_MDP:
            all_action_dones = self._compute_dones_for_expanded_mdp(
                all_obs, all_actions, all_rews)
            dones = all_action_dones & (out_of_bound_dones | time_dones)
        else:
            dones = out_of_bound_dones | time_dones
        
        return dones

    @abstractmethod
    def _sample_init_block_poses(self, env_idxs):
        pass

    def get_position_along_wall(self, env_idx, wall_name, 
                                pos_threshold_from_left_end,
                                block_width):
        wall_asset_name = self.get_wall_asset_name(env_idx, wall_name)
        wall_asset = self.get_wall_asset(env_idx, wall_name)
        wall_ah = self._scene.ah_map[env_idx][wall_asset_name]
        env_ptr = self._scene.env_ptrs[env_idx]
        wall_pose = wall_asset.get_rb_poses_as_np_array(env_ptr, wall_ah)
        wall_rot = rot_from_np_quat(wall_pose[0, 3:7])

        wall_config_name = self.get_wall_config_name(env_idx)
        wall_half_size = wall_rot @ self._wall_half_sizes_dict[wall_config_name][wall_name]
        wall_normal = wall_rot[:, 2]
        normal_towards_world = np.array([1, wall_normal[1], 0])

        angle = angle_between_axis(wall_normal, normal_towards_world)
        if angle > np.pi/2.0:
            wall_normal = -1 * wall_normal
        
        t = pos_threshold_from_left_end
        block_x = wall_pose[0, 0] + t * wall_half_size[0] + block_width * wall_normal[0]
        block_z = wall_pose[0, 2] + t * wall_half_size[2] + block_width * wall_normal[2]

        return block_x, block_z

    def _sample_init_block_poses_near_wall3(self, env_idxs, wall_name="wall3"):
        poses_2d = np.zeros((len(env_idxs), 3))
        npu = np.random.uniform

        if self._cfg['task']['init_block_rotation']:
            min_theta, max_theta = -np.pi, np.pi
        else:
            min_theta, max_theta = 0, 0

        for env_i, env_idx in enumerate(env_idxs):
            env_ptr = self._scene.env_ptrs[env_idx]
            wall_pose_dict = {}
            wall_rot_dict = {}
            wall_half_size_dict = {}

            wall_assets_dict = self.get_wall_assets_dict(env_idx)
            wall_config_name = self.get_wall_config_name(env_idx)
            for wall_name, wall_asset in wall_assets_dict.items():
                wall_asset_name = self.get_wall_asset_name(env_idx, wall_name)
                wall_ah = self._scene.ah_map[env_idx][wall_asset_name]
                wall_pose = wall_asset.get_rb_poses_as_np_array(env_ptr, wall_ah)
                wall_pose_dict[wall_name] = wall_pose

                wall_rot = rot_from_np_quat(wall_pose[0, 3:7])
                wall_rot_dict[wall_name] = wall_rot

                wall_half_size = wall_rot @ self._wall_half_sizes_dict[wall_config_name][wall_name]
                wall_half_size_dict[wall_name] = wall_half_size
            
            init_near_wall = self._cfg['task'].get('init_near_wall', wall_name)
            wall_normal = wall_rot_dict[init_near_wall][:, 2]
            # Since we know that vector towards positive x-axis is upwards i.e.
            # towards the world, make sure that the wall normal and x-axis have
            # an acute angle between them.
            normal_towards_world = np.array([1, wall_normal[1], 0])

            angle = angle_between_axis(wall_normal, normal_towards_world)
            if angle > np.pi/2.0:
                wall_normal = -1 * wall_normal

            def _sample_point_on_wall():
                t = npu(0.6, 0.99)
                block_x = wall_pose_dict[init_near_wall][0, 0] + t*wall_half_size_dict[wall_name][0]
                block_z = wall_pose_dict[init_near_wall][0, 2] + t*wall_half_size_dict[wall_name][2]
                
                # Now push this towards the world
                d = self._cfg['block']['dims']['width'] + 0.01
                block_x += d * wall_normal[0]
                block_z += d * wall_normal[2]

                return (block_x, block_z)
            
            block_pos = _sample_point_on_wall()

            poses_2d[env_i][0] = block_pos[0]
            poses_2d[env_i][1] = block_pos[1]
            poses_2d[env_i][2] = npu(min_theta, max_theta)

        return poses_2d

    def _reset(self, env_idxs):
        poses_2d = self._sample_init_block_poses(env_idxs)

        if self._expand_MDP:
            self._expand_MDP_action_step_counts[env_idxs] = 0

        for i, env_idx in enumerate(env_idxs):
            env_ptr = self._scene.env_ptrs[env_idx]
            block_ah = self._scene.ah_map[env_idx][self._block_name]
            pose_2d = poses_2d[i]

            block_transform = gymapi.Transform(
                p=gymapi.Vec3(
                    pose_2d[0],
                    # half block height + collision threshold
                    self._cfg['block']['dims']['height']/2 + 5e-3,
                    pose_2d[1]),
                r=rpy_to_quat([0, pose_2d[2], 0])
            )

            self._block.set_rb_transforms(env_ptr, block_ah, [block_transform])
            self._attractor_targets[env_idx] = block_transform
            self._scene._gym.set_attractor_target(env_ptr,
                                                  self._attractor_handles[env_idx],
                                                  block_transform)


class GymBlock2DFittingVecEnv(GymBlock2DVecEnv):

    def _init_rews(self, cfg):
        target_wall_positions = []
        target_wall_axes = []

        for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
            wall_name =  self._cfg['task'].get('target_wall', 'wall1')
            wall_asset = self.get_wall_asset(env_idx, wall_name)
            wall_asset_name = self.get_wall_asset_name(env_idx, wall_name)
            wall_ah = self._scene.ah_map[env_idx][wall_asset_name]
            wall_tf = wall_asset.get_rb_transforms(env_ptr, wall_ah)[0]
            wall_rigid_tf = transform_to_RigidTransform(wall_tf)
            target_wall_positions.append(wall_rigid_tf.position)
            target_wall_axes.append(wall_rigid_tf.rotation[:,2])

        self._target_wall_positions = np.array(target_wall_positions)[:, [0, 2]]
        self._target_wall_axes = np.array(target_wall_axes)

        self._prev_dists_to_target = None
        self._prev_angles_to_target = None
        self._prev_did_reach_target = None

        self._alive_penalty = cfg['rews']['alive_penalty']
        self._task_bonus = cfg['rews']['task_bonus']
        self._dist_weight = cfg['rews']['dist_weight']
        self._angle_weight = cfg['rews']['angle_weight']
        self._pos_dist_th = cfg['task']['pos_dist_th']
        self._angle_dist_th = np.deg2rad(cfg['task']['angle_dist_th'])


        if cfg['task']['expand_MDP']:
            self._expand_MDP_same_ctrlr_reward = cfg['rews']['expand_MDP_same_controller_reward']

    def _compute_dists_to_target(self, env_idxs):
        blocks_pos = []
        for env_idx in env_idxs:
            env_ptr = self._scene.env_ptrs[env_idx]
            block_ah = self._scene.ah_map[env_idx][self._block_name]
            block_tf = self._block.get_rb_transforms(env_ptr, block_ah)[0]
            blocks_pos.append(transform_to_np(block_tf)[[0, 2]])
        blocks_pos = np.vstack(blocks_pos)

        diffs_to_target = self._target_wall_positions[env_idxs] - blocks_pos
        dists_to_target = np.linalg.norm(diffs_to_target, axis=1)
        return dists_to_target
    
    def _compute_angles_to_target(self, env_idxs):
        blocks_axes = []
        for env_idx in env_idxs:
            env_ptr = self._scene.env_ptrs[env_idx]
            block_ah = self._scene.ah_map[env_idx][self._block_name]
            block_tf = self._block.get_rb_transforms(env_ptr, block_ah)[0]
            blocks_axes.append(quat_to_rot(block_tf.r)[:, 2])
        blocks_axes = np.vstack(blocks_axes)

        target = np.einsum('ni,ni->n', blocks_axes, self._target_wall_axes[env_idxs])
        target = np.clip(target, -1.0, 1.0, out=target)
        angles_to_target = np.arccos(target)

        return angles_to_target

    def _reset(self, env_idxs):
        super()._reset(env_idxs)
        if self._prev_dists_to_target is None:
            self._prev_dists_to_target = self._compute_dists_to_target(np.arange(self.n_envs))
        else:
            self._prev_dists_to_target[env_idxs] = self._compute_dists_to_target(env_idxs)

        if self._prev_angles_to_target is None:
            self._prev_angles_to_target = self._compute_angles_to_target(np.arange(self.n_envs))
        else:
            self._prev_angles_to_target[env_idxs] = self._compute_angles_to_target(env_idxs)

        if self._prev_did_reach_target is None:
            self._prev_did_reach_target = np.zeros((self.n_envs), dtype=np.bool)
        else:
            self._prev_did_reach_target[env_idxs] = False

    def _did_complete_task(self, dists_to_target, angles_to_target):
        return ((dists_to_target < self._pos_dist_th) & 
                (angles_to_target < self._angle_dist_th))

    def _compute_rews(self, all_obs, all_actions):
        if self._expand_MDP:
            rews = np.zeros(self.n_envs)

            # If we are using expanded MDP we should compute a reward only if
            # we have all the actions and we are ready to take a step in the env.
            max_step_count = self._expand_MDP_num_action_steps
            envs_with_all_actions = self._expand_MDP_action_step_counts_curr == max_step_count
            envs_without_all_actions = self._expand_MDP_action_step_counts_curr != max_step_count

            for env_idx in np.where(envs_without_all_actions)[0].tolist():
                assert max_step_count == 3, "The below code assumes this value."
                if self._expand_MDP_action_step_counts_curr[env_idx] == 1:
                    # We have only selected one controller for this env.
                    pass
                elif self._expand_MDP_action_step_counts_curr[env_idx] == 2:
                    # We have selected two controllers. If we select the same controller,
                    # or if the second controller-axis is the same as first controller axis
                    # we should get some negative reward.
                    first_ctrlr = self._expand_MDP_action_deque[-2][env_idx]
                    second_ctrlr = self._expand_MDP_action_deque[-1][env_idx]
                    if first_ctrlr == second_ctrlr and first_ctrlr != 0:
                        # we do not want to give a negative reward for null-movement 
                        # controller
                        rews[env_idx] += self._expand_MDP_same_ctrlr_reward
                else:
                    raise ValueError("Invalid action step count for expanded MDP")
            
            for env_idx in np.where(envs_with_all_actions)[0].tolist():
                first_ctrlr = self._expand_MDP_action_deque[-3][env_idx]
                second_ctrlr = self._expand_MDP_action_deque[-2][env_idx]
                third_ctrlr = self._expand_MDP_action_deque[-1][env_idx]
                if first_ctrlr == third_ctrlr and third_ctrlr != 0:
                    rews[env_idx] += self._expand_MDP_same_ctrlr_reward
                if second_ctrlr == third_ctrlr and third_ctrlr != 0:
                    rews[env_idx] += self._expand_MDP_same_ctrlr_reward

        cur_dists_to_target = self._compute_dists_to_target(np.arange(self.n_envs))
        dist_improvements = self._prev_dists_to_target - cur_dists_to_target

        cur_angles_to_target = self._compute_angles_to_target(np.arange(self.n_envs))
        angle_improvements = self._prev_angles_to_target - cur_angles_to_target

        task_bonus_arr = self._did_complete_task(cur_dists_to_target, 
                                                cur_angles_to_target)

        if self._expand_MDP and np.sum(envs_with_all_actions) > 0:
            rews[envs_with_all_actions] += \
                self._dist_weight * dist_improvements[envs_with_all_actions] + \
                self._angle_weight * angle_improvements[envs_with_all_actions] + \
                self._alive_penalty
            assert envs_with_all_actions.shape == task_bonus_arr.shape
            rews[envs_with_all_actions & task_bonus_arr] += self._task_bonus
        else:
            rews = self._dist_weight * dist_improvements + \
               self._angle_weight * angle_improvements + \
               self._alive_penalty
            rews[task_bonus_arr] += self._task_bonus
        
        self._prev_did_reach_target = task_bonus_arr

        self._prev_dists_to_target = cur_dists_to_target
        self._prev_angles_to_target = cur_angles_to_target        
        return rews

    def _compute_dones(self, all_obs, all_actions, all_rews):
        super_dones = super()._compute_dones(all_obs, all_actions, all_rews)
        task_dones = self._did_complete_task(self._prev_dists_to_target, 
                                            self._prev_angles_to_target)

        if self._expand_MDP:
            all_action_dones = self._compute_dones_for_expanded_mdp(
                    all_obs, all_actions, all_rews)
            return all_action_dones & (super_dones | task_dones)
        else:
            return super_dones | task_dones

    def _compute_infos(self, all_obs, all_actions, all_rews, all_dones):
        all_infos = []
        for env_idx, _ in enumerate(self._scene.env_ptrs):
            info_dict = {}
            if all_dones[env_idx] and self._prev_did_reach_target[env_idx]:
                info_dict['is_success'] = 1
                self.completed_eplen_by_env_idx[env_idx].append(self.step_counts[env_idx] + 1)
            else:
                info_dict['is_success'] = 0

            # PPO2 code uses the episode key for some reason to save data.
            if 'rl' in self._cfg and self._cfg['rl']['algo'] == 'ppo':
                # The reward for this step is added to episode_rewards after 
                # compute_info is called hence we need to add all_rews[env_idx].
                info_dict['r'] = self.episode_rewards[env_idx] + all_rews[env_idx]
                info_dict['l'] = self.step_counts[env_idx] + 1
                info_dict = {'episode': info_dict}

            all_infos.append(info_dict)
        
        # Save infos for logging.
        self._prev_infos = all_infos

        return all_infos

    def _sample_init_block_poses(self, env_idxs):
        return self._sample_init_block_poses_near_wall3(env_idxs)
    
    def _sample_init_block_poses_org(self, env_idxs):
        if self._cfg['task']['init_block_rotation']:
            min_theta, max_theta = -np.pi, np.pi
        else:
            min_theta, max_theta = 0, 0

        poses_2d = np.zeros((len(env_idxs), 3))
        for i, env_idx in enumerate(env_idxs):
            wall_config_name = self.get_wall_config_name(env_idx)
            wall2_end_pos = self._wall_end_dict[wall_config_name]['wall2']
            wall3_end_pos = self._wall_end_dict[wall_config_name]['wall3']
            min_x = max(wall2_end_pos[0], wall3_end_pos[0]) + self._block_clearance
            max_x = self.obs_space.high[0]
            min_z = wall2_end_pos[1]
            max_z = self.obs_space.high[1]

            poses_2d[i] = np.random.uniform([min_x, min_z, min_theta], [max_x, max_z, max_theta]) 
        return poses_2d


class GymBlock2DPushVecEnv(GymBlock2DVecEnv):

    def _fill_scene(self, cfg):
        super()._fill_scene(cfg)

        # create push block asset
        self._push_block = GymBoxAsset(self._scene.gym, self._scene.sim,
            **cfg['push_block']['dims'],
            shape_props=cfg['push_block']['shape_props'],
            rb_props=cfg['push_block']['rb_props'],
            asset_options=cfg['push_block']['asset_options']
        )

        self._push_block_name = 'push_block'
        self._scene.add_asset(self._push_block_name, self._push_block, gymapi.Transform())

        self._all_env_idx_list = np.arange(self.n_envs)

    def _init_rews(self, cfg):
        self._target_position = np.zeros((self.n_envs, 2))
        for env_idx in range(self.n_envs):
            wall_config_name = self.get_wall_config_name(env_idx)
            rew_wall = self._cfg['walls'][wall_config_name]['reward_wall']
            rew_direction = int(self._cfg['walls'][wall_config_name]['reward_direction'])
            assert rew_direction in (1, -1), f"Invalid reward direction: {rew_direction}"
            wall_mid_pos = self._wall_pos_dict[wall_config_name][rew_wall] 
            wall_end_pos = self._wall_end_dict[wall_config_name][rew_wall]
            wall_vector = wall_end_pos - wall_mid_pos
            wall_vector_norm = wall_vector / np.linalg.norm(wall_vector)
            target_pos_i = wall_end_pos - cfg['task']['target_dist_from_wall_end'] * wall_vector_norm

            wall_asset = self.get_wall_asset(env_idx, rew_wall)
            wall_asset_name = self.get_wall_asset_name(env_idx, rew_wall)
            wall_ah = self._scene.ah_map[env_idx][wall_asset_name]
            wall_tf = wall_asset.get_rb_transforms(self._scene.env_ptrs[env_idx], wall_ah)[0]
            wall_R = quat_to_rot(wall_tf.r)
            wall_normal = rew_direction * wall_R[:, 2]
            block_pos_away_from_wall = (cfg['push_block']['dims']['width']/2.) * wall_normal
            target_pos_i += block_pos_away_from_wall[[0, 2]]
        
            self._target_position[env_idx] = target_pos_i

        self._prev_dists_to_target = None
        self._prev_dists_to_block = None
        self._alive_penalty = cfg['rews']['alive_penalty']
        self._task_bonus = cfg['rews']['task_bonus']
        self._approach_weight = cfg['rews']['approach_weight']
        self._dist_weight = cfg['rews']['dist_weight']
        self._pos_dist_th = cfg['task']['pos_dist_th']

    # Can override this method in sub-class.
    def get_adaptive_obs_bounds(self, cfg):
        return self._get_adaptive_obs_bounds(cfg, block_size=self._cfg['push_block']['dims']['width'])

    def _init_obs_space(self, cfg):
        # x, y, theta for push_block
        super_obs_space = super()._init_obs_space(cfg)
        self._super_obs_space = super_obs_space
        limits_low = np.concatenate([super_obs_space.low, np.array([-1, -1, -np.pi])])
        limits_high = np.concatenate([super_obs_space.high, np.array([1, 1.5, np.pi])])
        obs_space = Box(limits_low, limits_high, dtype=np.float32)
        return obs_space

    def _make_low_level_controllers_dict_for_env(self, action_cfg, env_idx, env_ptr):
        low_level_ctrlr_dict = super()._make_low_level_controllers_dict_for_env(action_cfg, env_idx, env_ptr)
        push_block_ah = self._scene.ah_map[env_idx][self._push_block_name]
        push_block_tf = self._push_block.get_rb_transforms(env_ptr, push_block_ah)[0]
        push_block_tra = vec3_to_np(push_block_tf.p)
        low_level_ctrlr_dict['push_block_pos'] = ErrorAxisPositionAttractorController(push_block_tra, action_cfg['attractors']['limits']['dx'])
        self._combo_wall_specific_ctrlrs += [(len(low_level_ctrlr_dict) - 1, 0, 0)]

        return low_level_ctrlr_dict

    def _apply_inter_actions(self, all_actions, t_inter_step, n_inter_steps):
        if self._cfg['action']['type'] != 'raw':
            # Update push block controller targets
            for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
                push_block_ah = self._scene.ah_map[env_idx][self._push_block_name]
                push_block_tf = self._push_block.get_rb_transforms(env_ptr, push_block_ah)[0]
                push_block_tra = vec3_to_np(push_block_tf.p)

                high_level_ctrlr = self._high_level_controllers[env_idx]
                push_block_pos_ctrlr = high_level_ctrlr.low_level_controllers_dict['push_block_pos']
                push_block_pos_ctrlr.target = push_block_tra

        super()._apply_inter_actions(all_actions, t_inter_step, n_inter_steps)

    def _compute_dists_to_target(self, env_idxs):
        push_block_pos = []
        for env_idx in env_idxs:
            env_ptr = self._scene.env_ptrs[env_idx]
            push_block_ah = self._scene.ah_map[env_idx][self._push_block_name]
            push_block_tf = self._block.get_rb_transforms(env_ptr, push_block_ah)[0]
            push_block_pos.append(transform_to_np(push_block_tf)[[0, 2]])
        push_block_pos = np.vstack(push_block_pos)

        diffs_to_target = self._target_position[env_idxs] - push_block_pos
        dists_to_target = np.linalg.norm(diffs_to_target, axis=1)
        return dists_to_target

    def _compute_dists_to_target_and_block(self, env_idxs):
        push_block_pos, dists_to_block = [], []
        for env_idx in env_idxs:
            env_ptr = self._scene.env_ptrs[env_idx]
            push_block_ah = self._scene.ah_map[env_idx][self._push_block_name]
            push_block_tf = self._block.get_rb_transforms(env_ptr, push_block_ah)[0]

            push_block_pos.append(transform_to_np(push_block_tf)[[0, 2]])

            block_ah = self._scene.ah_map[env_idx][self._block_name]
            block_tf = self._block.get_rb_transforms(env_ptr, block_ah)[0]
            dist = np.linalg.norm(vec3_to_np(block_tf.p - push_block_tf.p))

            dists_to_block.append(dist)

        push_block_pos = np.vstack(push_block_pos)
        diffs_to_target = self._target_position[env_idxs] - push_block_pos
        dists_to_target = np.linalg.norm(diffs_to_target, axis=1)

        return dists_to_target, np.array(dists_to_block)

    def _compute_dists_to_block(self, env_idxs):
        dists_to_block = []
        for env_idx in env_idxs:
            env_ptr = self._scene.env_ptrs[env_idx]
            push_block_ah = self._scene.ah_map[env_idx][self._push_block_name]
            push_block_tf = self._block.get_rb_transforms(env_ptr, push_block_ah)[0]

            block_ah = self._scene.ah_map[env_idx][self._block_name]
            block_tf = self._block.get_rb_transforms(env_ptr, block_ah)[0]

            dist = np.linalg.norm(vec3_to_np(block_tf.p - push_block_tf.p))

            dists_to_block.append(dist)

        return np.array(dists_to_block)

    def _add_child_obs(self, all_actions, obs, is_reset=False):
        obs_idx = self._super_obs_space.shape[0]
        for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
            push_block_ah = self._scene.ah_map[env_idx][self._push_block_name]
            push_block_transform = self._push_block.get_rb_transforms(env_ptr, push_block_ah)[0]

            obs[env_idx, obs_idx] = push_block_transform.p.x
            obs[env_idx, obs_idx + 1] = push_block_transform.p.z

            # This is rotation around Y axis
            push_block_R = quat_to_rot(push_block_transform.r)
            sin_theta, cos_theta = push_block_R[0, 2], push_block_R[2, 2]
            # arctan2 has output in the of [-pi, pi], all other np methos
            # such as np.arcos only output in 1 quadrant i.e. [0, pi].self.
            # theta_2 = np.arccos(block_R[0, 0]) # block's x-axis angle from world's x axis
            theta = np.arctan2(sin_theta, cos_theta)
            obs[env_idx, obs_idx + 2] = theta

    def _compute_rews(self, all_obs, all_actions):
        # Reward is improvement in dists to target plus alive penalty to encourage faster improvement
        cur_dists_to_target, cur_dists_to_block = self._compute_dists_to_target_and_block(
            self._all_env_idx_list)
        block_improvements = self._prev_dists_to_target - cur_dists_to_target
        approach_improvements = self._prev_dists_to_block - cur_dists_to_block
        rews = self._dist_weight * block_improvements + \
               self._approach_weight * approach_improvements + \
               self._alive_penalty
        self._prev_did_reach_target = cur_dists_to_target < self._pos_dist_th
        rews[self._prev_did_reach_target] += self._task_bonus

        self._prev_dists_to_target = cur_dists_to_target
        self._prev_dists_to_block = cur_dists_to_block
        return rews

    def _compute_dones(self, all_obs, all_actions, all_rews):
        super_dones = super()._compute_dones(all_obs, all_actions, all_rews)

        obs_idx = self._super_obs_space.shape[0]
        push_block_pos = all_obs[:, obs_idx:obs_idx+2]
        assert self._add_y_axis_to_obs
        if self._use_adaptive_env_bounds:
            push_block_dones = np.any(push_block_pos < self._min_env_bounds[:, [0, 2]] , axis=1) \
                | np.any(push_block_pos > self._max_env_bounds[:, [0, 2]], axis=1)
        else:
            push_block_dones = np.any(
                push_block_pos < self.obs_space.low[obs_idx:obs_idx+2], axis=1) \
                | np.any(
                    push_block_pos > self.obs_space.high[obs_idx:obs_idx+2], axis=1)

        task_dones = self._prev_did_reach_target

        return super_dones | push_block_dones | task_dones

    def _sample_init_block_poses(self, env_idxs):
        return self._sample_init_block_poses_near_wall3(env_idxs, "wall2")

    def _reset(self, env_idxs):
        super()._reset(env_idxs)
        push_block_zs = np.random.uniform(0.6, 0.9, size=len(env_idxs))

        for i, env_idx in enumerate(env_idxs):
            env_ptr = self._scene.env_ptrs[env_idx]
            push_block_ah = self._scene.ah_map[env_idx][self._push_block_name]

            width = self._cfg['push_block']['dims']['width'] + 4e-2
            height = self._cfg['push_block']['dims']['height'] + 5e-3
            wall_config_name = self.get_wall_config_name(env_idx)
            width, height = self.get_position_along_wall(
                env_idx,
                self._cfg['walls'][wall_config_name]["push_block_init_wall"],
                -np.random.uniform(0.1, 0.3), 
                max(width, height))

            push_block_transform = gymapi.Transform(
                p=gymapi.Vec3(
                    width, # self._cfg['push_block']['dims']['width']/2 + 4e-2,
                    self._cfg['push_block']['dims']['height']/2 + 5e-3,
                    height
                    ),
            )

            self._block.set_rb_transforms(env_ptr, push_block_ah, [push_block_transform])

        if self._prev_dists_to_target is None:
            self._prev_dists_to_target = self._compute_dists_to_target(np.arange(self.n_envs))
        else:
            self._prev_dists_to_target[env_idxs] = self._compute_dists_to_target(env_idxs)
        if self._prev_dists_to_block is None:
            self._prev_dists_to_block = self._compute_dists_to_block(np.arange(self.n_envs))
        else:
            self._prev_dists_to_block[env_idxs] = self._compute_dists_to_block(env_idxs)
