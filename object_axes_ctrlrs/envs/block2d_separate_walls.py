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


class GymBlock2DSeparateVecEnv(GymVecEnv, ABC):

    @property
    def is_2d(self):
        return True

    @staticmethod
    def get_xz_positions_for_segment(size, theta, origin=(0, 0)):
        '''In Isaac x points towards the y-axis and z towards x-axis.'''
        # Assuming the angle is measured towards positive X-axis
        theta_rad = np.deg2rad(theta)
        pos_from_origin = size * np.cos(theta_rad), -size * np.sin(theta_rad)

        # Get the world pos
        world_pos = np.array([origin[i] + pos_from_origin[i] for i in range(2)])

        # Get the end position for this object.
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
        self._wall_corners_by_wall_dict = {}
        self._separate_walls = {}

        for wall_config_name in wall_config_names:
            self._wall_half_width_dict[wall_config_name] = \
                cfg['walls'][wall_config_name]['half_width']
            self._wall_angles_dict[wall_config_name] = \
                cfg['walls'][wall_config_name]['wall_angles']
            self._separate_walls[wall_config_name] = \
                cfg['walls'][wall_config_name]['separate_walls']

            self._wall_half_sizes_dict[wall_config_name] = {}
            self._wall_pos_dict[wall_config_name] = {}
            self._wall_end_dict[wall_config_name] = {}
            
            wall_config = cfg['walls'][wall_config_name]
            wall_names = ['wall{}'.format(i) for i in range(wall_config['n_walls'])]

            wall_start = np.array(cfg['walls'][wall_config_name]['origin'])
            wall_corners_dict = dict()
            # Go through walls in half_width (these walls should be sequential)
            for wall_name in cfg['walls'][wall_config_name]['half_width'].keys():
                half_width = self._wall_half_width_dict[wall_config_name][wall_name]
                self._wall_half_sizes_dict[wall_config_name][wall_name] = np.array([half_width, 0, cfg['wall']['dims']['depth'] / 2])
                angle = self._wall_angles_dict[wall_config_name][wall_name]

                wall_pos_xz, wall_end = GymBlock2DSeparateVecEnv.get_xz_positions_for_segment(
                    half_width, angle, wall_start)
                self._wall_pos_dict[wall_config_name][wall_name] = wall_pos_xz
                self._wall_end_dict[wall_config_name][wall_name] = wall_end
                wall_corners_dict[wall_name] = (wall_start, wall_end)
                wall_start = wall_end

            # Now go through wall properties for sequential walls
            for wall_name, wall_props in cfg['walls'][wall_config_name]['separate_walls'].items():
                wall_start, half_width, theta = np.array(wall_props['start']), wall_props['half_width'], wall_props['angle']
                theta_rad = np.deg2rad(theta)
                # Add half-width to dict with continuous wall half-widths
                assert self._wall_half_width_dict[wall_config_name].get(wall_name) is None
                self._wall_half_width_dict[wall_config_name][wall_name] = half_width
                # Add angle to dict with continuous wall angles
                assert self._wall_angles_dict[wall_config_name].get(wall_name) is None
                self._wall_angles_dict[wall_config_name][wall_name] = theta
                wall_pos = np.array([
                    wall_start[0] + half_width*np.cos(theta_rad),
                    wall_start[1] + half_width*np.sin(theta_rad)
                ])
                self._wall_half_sizes_dict[wall_config_name][wall_name] = np.array([half_width, 0, cfg['wall']['dims']['depth'] / 2])
                self._wall_pos_dict[wall_config_name][wall_name] = wall_pos
                wall_end = np.array([
                    wall_start[0] + 2*half_width*np.cos(theta_rad),
                    wall_start[1] + 2*half_width*np.sin(theta_rad)
                ])
                self._wall_end_dict[wall_config_name][wall_name] = wall_end
                wall_corners_dict[wall_name] = (wall_start, wall_end)
            
            wall_corners_list = []
            for wall_name in sorted(wall_corners_dict.keys()):
                wall_corners_list.append(
                    wall_corners_dict[wall_name][0].tolist() +
                    wall_corners_dict[wall_name][1].tolist())
            self._wall_corners_dict[wall_config_name] = np.array(wall_corners_list)
            self._wall_corners_by_wall_dict[wall_config_name] = wall_corners_dict
            # import ipdb; ipdb.set_trace()

            
    def _fill_scene(self, cfg):
        if cfg['action']['type'] == 'raw':
            assert cfg['env']['n_inter_steps'] == cfg['action']['raw']['n_interp_steps'], \
                "Number of interpolation steps must be same as number of " \
                    "intermediate steps for raw action space"

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

        # num_walls * [wall_pos, end]
        if self._train_in_multi_env:
            self._wall_config_obs_size = len(self._walls[self._multi_env_types[0]]) * 6  
        else:
            self._wall_config_obs_size = len(self._walls) * 6  # num_walls * [wall_pos, end]

        # Save previous step infos for logging/debugging. This is required since
        # vec-envs autoreset.
        self._prev_infos = None

        if self._expand_MDP:
            # The original MDP that the "RL agent" sees is the low-level expanded
            # MDP hence the original _step_count corresponds to that MDP. In here
            # we manually keep track of the original high-level MDP.
            self._expand_MDP_action_step_counts = np.zeros((self.n_envs), dtype=np.int32)
            self._expand_MDP_action_step_counts_curr = np.zeros((self.n_envs), dtype=np.int32)
            self._expand_MDP_last_obs = None
            # Deque used to store last N actions.
            self._expand_MDP_action_deque = deque([], self._expand_MDP_num_action_steps)
            # Used only for book-keeping and verification purpose
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
            # Convert a deque object into a list
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
        wall_corners_by_env_x = wall_corners_by_env[:, :, [0, 2]].reshape(self.n_envs, -1)
        wall_corners_by_env_z = wall_corners_by_env[:, :, [1, 3]].reshape(self.n_envs, -1)
        min_corner_x, max_corner_x = wall_corners_by_env_x.min(axis=1), wall_corners_by_env_x.max(axis=1)
        min_corner_z, max_corner_z = wall_corners_by_env_z.min(axis=1), wall_corners_by_env_z.max(axis=1)
        min_corner = np.vstack([min_corner_x, min_corner_z]).T
        max_corner = np.vstack([max_corner_x, max_corner_z]).T
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
        if self._cfg['action']['type'] == 'combo':
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
            if action_applied_to_num_envs == self.n_envs:
                self._n_inter_steps = self._cfg['env']['n_inter_steps']
            elif action_applied_to_num_envs == 0:
                self._n_inter_steps = 0
            else:
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
            normal_towards_world = np.array([0, wall_normal[1], 1])

            angle = angle_between_axis(wall_normal, normal_towards_world)
            if angle > np.pi/2.0:
                wall_normal = -1 * wall_normal
            
            wall_corners = self._wall_corners_by_wall_dict[wall_config_name][init_near_wall]
            wall_min_x = min(wall_corners[0][0], wall_corners[1][0])
            wall_max_z = max(wall_corners[0][1], wall_corners[1][1])

            d = self._cfg['block']['dims']['width']
            block_z = wall_max_z + d + npu(d/2, d)
            block_x = npu(wall_min_x - d/2, wall_min_x + d/2)

            poses_2d[env_i][0] = block_x
            poses_2d[env_i][1] = block_z
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
                    self._cfg['block']['dims']['height']/2 + 5e-3,
                    pose_2d[1]),
                r=rpy_to_quat([0, pose_2d[2], 0])
            )

            self._block.set_rb_transforms(env_ptr, block_ah, [block_transform])
            self._attractor_targets[env_idx] = block_transform
            self._scene._gym.set_attractor_target(env_ptr,
                                                  self._attractor_handles[env_idx],
                                                  block_transform)


class GymBlock2DSeparateFittingVecEnv(GymBlock2DSeparateVecEnv):

    def _init_rews(self, cfg):
        # Some intermediate target positions.
        wall1_wall3_midpoint_positions = []
        wall1_wall3_top_edge_midpoint_positions = []
        wall1_wall3_above_top_edge_midpoint_positions = []
        target_wall_positions = []
        target_wall_axes = []

        self._wall1_positions = np.zeros((self.n_envs, 2))
        self._wall3_positions = np.zeros((self.n_envs, 2))

        for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
            wall_name =  self._cfg['task'].get('target_wall')
            wall_asset = self.get_wall_asset(env_idx, wall_name)
            wall_asset_name = self.get_wall_asset_name(env_idx, wall_name)
            wall_ah = self._scene.ah_map[env_idx][wall_asset_name]
            wall_tf = wall_asset.get_rb_transforms(env_ptr, wall_ah)[0]
            wall_rigid_tf = transform_to_RigidTransform(wall_tf)

            wall1_pos, wall1_R = self.get_wall_pose(env_idx, "wall1")
            wall3_pos, wall3_R = self.get_wall_pose(env_idx, "wall2")
            self._wall1_positions[env_idx, :] = wall1_pos[[0, 2]]
            self._wall3_positions[env_idx, :] = wall3_pos[[0, 2]]

            # The first target position is the mid point of wall1 and wall3
            wall1_wall3_midpoint_positions.append((wall1_pos + wall3_pos) / 2.0)

            # Get the midpoint of the top edge between wall1 and wall3
            env_config = self._cfg['walls']['multi_env_types'][self._env_type_by_env_idx[env_idx]]
            wall1_start_pos = self._wall_corners_by_wall_dict[env_config]['wall1'][0]
            wall3_end_pos = self._wall_corners_by_wall_dict[env_config]['wall3'][1]
            wall1_wall3_top_edge_midpoint_positions.append((wall1_start_pos + wall3_end_pos) / 2.0)

            wall1_axes_along_wall = wall1_R[:, 0]
            # The X-axes points upwards
            axes_to_upper_wall = np.array([1, wall1_axes_along_wall[1], 0])
            angle = angle_between_axis(wall1_axes_along_wall, axes_to_upper_wall)
            if angle > np.pi/2.0:
                wall1_axes_along_wall *= -1
            d_half = self._cfg['block']['dims']['width'] / 2.0
            top_edge_pos = (wall1_start_pos + wall3_end_pos) / 2.0
            wall1_wall3_above_top_edge_midpoint_positions.append(
                [top_edge_pos[0] + d_half * wall1_axes_along_wall[0], 
                 top_edge_pos[1] + d_half * wall1_axes_along_wall[2]])

            target_wall_positions.append(wall_rigid_tf.position)
            target_wall_axes.append(wall_rigid_tf.rotation[:,2])

        self._target_wall_positions = np.array(target_wall_positions)[:, [0, 2]]
        self._target_wall_axes = np.array(target_wall_axes)
        self._wall1_wall3_midpoint_positions = np.array(wall1_wall3_midpoint_positions)[:, [0, 2]]
        self._wall1_wall3_top_edge_midpoint_positions = np.array(wall1_wall3_top_edge_midpoint_positions)
        self._wall1_wall3_above_top_edge_midpoint_positions = np.array(wall1_wall3_above_top_edge_midpoint_positions)
        self._initial_target_positions = self._wall1_wall3_above_top_edge_midpoint_positions

        self._prev_dists_to_target = None
        self._prev_angles_to_target = None
        self._prev_did_reach_target = None
        # Used to see if we reached the first goal or the second goal?
        self._did_reach_initial_position = None
        self._did_reach_final_position = None

        self._alive_penalty = cfg['rews']['alive_penalty']
        self._task_bonus = cfg['rews']['task_bonus']
        self._dist_weight = cfg['rews']['dist_weight']
        self._angle_weight = cfg['rews']['angle_weight']
        self._pos_dist_th = cfg['task']['pos_dist_th']
        self._angle_dist_th = np.deg2rad(cfg['task']['angle_dist_th'])


        if cfg['task']['expand_MDP']:
            self._expand_MDP_same_ctrlr_reward = cfg['rews']['expand_MDP_same_controller_reward']

        # Pre-allocate these so that we don't do this at "every step"
        self._blocks_positions = np.zeros((self.n_envs, 2))
        self._blocks_axes = np.zeros((self.n_envs, 3))
        self.n_env_array = np.arange(self.n_envs)
        self._did_reach_initial_goal = np.zeros((self.n_envs, 1), dtype=np.bool)
        self._prev_did_reach_initial_goal = np.zeros((self.n_envs, 1), dtype=np.bool)
        
    def _compute_dists_angles_to_target(self, env_idxs):
        for env_idx in env_idxs:
            env_ptr = self._scene.env_ptrs[env_idx]
            block_ah = self._scene.ah_map[env_idx][self._block_name]
            block_tf = self._block.get_rb_transforms(env_ptr, block_ah)[0]
            self._blocks_positions[env_idx, :] = transform_to_np(block_tf)[[0, 2]]
            self._blocks_axes[env_idx, :] = quat_to_rot(block_tf.r)[:, 2]

        target = np.einsum('ni,ni->n', self._blocks_axes[env_idxs], self._target_wall_axes[env_idxs])
        target = np.clip(target, -1.0, 1.0, out=target)
        angles_to_target = np.arccos(target)

        final_dists = np.linalg.norm(self._target_wall_positions[env_idxs] - self._blocks_positions[env_idxs], axis=1)
        initial_dists = np.linalg.norm(self._initial_target_positions[env_idxs] - self._blocks_positions[env_idxs], axis=1)

        # Check z-axis between walls
        z_axis_between_walls = \
            (self._blocks_positions[env_idxs, 1] > self._wall1_positions[env_idxs, 1]) & \
            (self._blocks_positions[env_idxs, 1] < self._wall3_positions[env_idxs, 1])

        # Check x-axis between walls
        x_axis_between_walls = \
            (self._blocks_positions[env_idxs, 0] > self._wall1_wall3_midpoint_positions[env_idxs, 0]) & \
            (self._blocks_positions[env_idxs, 0] < self._wall1_wall3_top_edge_midpoint_positions[env_idxs, 0])
        
        self._did_reach_initial_goal[env_idxs, 0] |= (x_axis_between_walls & z_axis_between_walls)

        dists_to_target = (1 - self._did_reach_initial_goal[env_idxs, 0]) * initial_dists \
            + self._did_reach_initial_goal[env_idxs, 0] * final_dists
        assert len(dists_to_target.shape) == 1

        return dists_to_target, angles_to_target

    def _reset(self, env_idxs):
        super()._reset(env_idxs)
        if self._prev_dists_to_target is None:
            self._prev_dists_to_target, self._prev_angles_to_target = self._compute_dists_angles_to_target(
                self.n_env_array)
        else:
            self._prev_dists_to_target[env_idxs], self._prev_angles_to_target[env_idxs] = \
                self._compute_dists_angles_to_target(env_idxs)
            self._did_reach_initial_goal[env_idxs, 0] = False

        if self._prev_did_reach_target is None:
            self._prev_did_reach_target = np.zeros((self.n_envs), dtype=np.bool)
        else:
            self._prev_did_reach_target[env_idxs] = False

    def _did_complete_task(self, dists_to_target, angles_to_target):
        return ((dists_to_target < self._pos_dist_th) & 
                (angles_to_target < self._angle_dist_th) & 
                (self._did_reach_initial_goal[:, 0]))
        

    def _compute_rews(self, all_obs, all_actions):
        if self._expand_MDP:
            rews = np.zeros(self.n_envs)

            max_step_count = self._expand_MDP_num_action_steps
            envs_with_all_actions = self._expand_MDP_action_step_counts_curr == max_step_count
            envs_without_all_actions = self._expand_MDP_action_step_counts_curr != max_step_count

            for env_idx in np.where(envs_without_all_actions)[0].tolist():
                assert max_step_count == 3, "The below code assumes this value."
                if self._expand_MDP_action_step_counts_curr[env_idx] == 1:
                    pass
                elif self._expand_MDP_action_step_counts_curr[env_idx] == 2:
                    first_ctrlr = self._expand_MDP_action_deque[-2][env_idx]
                    second_ctrlr = self._expand_MDP_action_deque[-1][env_idx]
                    if first_ctrlr == second_ctrlr and first_ctrlr != 0:
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


        cur_dists_to_target, cur_angles_to_target = \
            self._compute_dists_angles_to_target(self.n_env_array)
        
        dist_improvements = self._prev_dists_to_target - cur_dists_to_target
        angle_improvements = self._prev_angles_to_target - cur_angles_to_target

        newly_reached_initial_goal = \
            (self._prev_did_reach_initial_goal[:, 0] == 0) & (self._did_reach_initial_goal[:, 0] == 1)
        dist_improvements[newly_reached_initial_goal] = 1.0

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
        self._prev_did_reach_initial_goal[...] = self._did_reach_initial_goal[...]
        return rews

    def _compute_dones(self, all_obs, all_actions, all_rews):
        super_dones = super()._compute_dones(all_obs, all_actions, all_rews)
        task_dones = self._prev_did_reach_target

        if self._expand_MDP:
            # This dones is being applied twice, but this is fine for now.
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
