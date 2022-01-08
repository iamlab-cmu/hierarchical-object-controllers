import numpy as np
from itertools import permutations
from collections import OrderedDict

from isaac_gym import gymapi
from isaac_gym_utils.assets import GymURDFAsset, GymBoxAsset
from isaac_gym_utils.math_utils import (rpy_to_quat, transform_to_np, np_to_vec3, vec3_to_np,
                                        quat_to_rpy, quat_to_rot, slerp_quat)

from gym.spaces import Box, Discrete

from isaac_gym_utils.rl import GymFrankaVecEnv

from object_axes_ctrlrs.controllers.planar_box_world_high_level_controller import BoxWorldHighLevelController
from object_axes_ctrlrs.controllers.projected_axes_attractors import \
    PositionAttractorController, RotationAttractorController, \
    ForceAttractorController, NullAttractorController, \
    ComposedAttractorControllers, ErrorAxisPositionAttractorController, \
    FrankaGripperController, CurlPositionAttractorController, \
    ControllersListFeatureExtractor


class GymFrankaDoorOpenVecEnv(GymFrankaVecEnv):

    def _parse_door_multi_env_properties(self, cfg, parse_multi_configs=False):
        if parse_multi_configs:
            self._multi_env_types = cfg['door']['multi_env_types']
        else:
            self._multi_env_types = [cfg['door']['type']]

        self._door_urdf_dict = {}
        self._door_dof_props = {}

        for cfg_name in self._multi_env_types:
            self._door_urdf_dict[cfg_name] = cfg['door'][cfg_name]['urdf_path']
            self._door_dof_props[cfg_name] = cfg['door'][cfg_name]['dof_props']

    def _fill_scene(self, cfg):
        super()._fill_scene(cfg)

        self._train_in_multi_env = cfg['task'].get('train_in_multi_env', False)
        self._parse_door_multi_env_properties(cfg, self._train_in_multi_env)

        if self._train_in_multi_env:
            self._env_type_by_env_idx = []
            for env_idx in range(self.n_envs):
                self._env_type_by_env_idx.append(env_idx % len(self._multi_env_types))

        self._doors = {}
        self._door_names = {}
        for config_name in self._multi_env_types:
            self._doors[config_name] = GymURDFAsset(
                self._door_urdf_dict[config_name],
                self._scene.gym,
                self._scene.sim,
                dof_props=self._door_dof_props[config_name],
                asset_options=cfg['door']['asset_options'],
                assets_root=cfg['local_assets_path']
            )
            self._door_names[config_name] = f'{config_name}_door0'

        self._mean_door_pos = np.array(cfg['door']['position'])
        self._mean_door_rot = np.deg2rad(cfg['door']['y_rotation'])
        self._sample_door_pose = cfg['door']['sample_pose']
        self._door_pos_sample_range = np.array(cfg['door']['position_sample_range'])
        self._door_rot_sample_range = np.deg2rad(cfg['door']['y_rotation_sample_range'])

        # calc lock / unlock poses of block
        mean_lock_pos, mean_unlock_pos = self._calc_lock_unlock_pose(self._mean_door_pos, self._mean_door_rot)
        self._lock_poses = [mean_lock_pos for _ in range(self.n_envs)]
        self._unlock_poses = [mean_unlock_pos for _ in range(self.n_envs)]

        # add assets to scene
        self._lock = GymBoxAsset(
            self._scene.gym, self._scene.sim, **cfg['lock']['dims'],
            shape_props=cfg['lock']['shape_props'],
            rb_props=cfg['lock']['rb_props'],
            asset_options=cfg['lock']['asset_options']
        )
        self._lock_name = 'lock0'
        for env_idx in range(self.n_envs):
            door_config_name = self.get_door_config_name(env_idx)
            door_xyz = self._mean_door_pos
            door_rot = self._mean_door_rot

            door_pose = gymapi.Transform(
                p=np_to_vec3(np.array(door_xyz)),
                r=rpy_to_quat([0, door_rot, 0])
            )
            door_asset = self._doors[door_config_name]
            door_name = self._door_names[door_config_name]
            self._scene.add_asset(
                door_name,
                door_asset,
                door_pose,
                envs=[env_idx]
            )
            block_lock_pose = self._lock_poses[env_idx]
            self._scene.add_asset(
                self._lock_name,
                self._lock,
                block_lock_pose,
                envs=[env_idx]
            )

        self._prev_infos = None
        self._max_steps = cfg['task']['max_steps']
        # will sample ee pos in reset
        self._sample_init_ee_pos = cfg['task']['sample_init_ee_pos']
        self._ee_pos_sample_range = np.deg2rad(cfg['task']['ee_pos_sample_range'])

    def _calc_lock_unlock_pose(self, door_pos, door_y_rot):
        diff = np.deg2rad(4)
        lock_pose_p = np.array([0.4 * np.cos(door_y_rot - diff), 0, 0.05 - 0.4 * np.sin(door_y_rot - diff)]) + door_pos
        lock_pose = gymapi.Transform(
            p=np_to_vec3(lock_pose_p),
            r=rpy_to_quat([0, door_y_rot - diff, 0])
        )
        unlock_pose_p = np.array([0.8 * np.cos(door_y_rot - diff), 0, 0.05 - 0.8 * np.sin(door_y_rot - diff)]) + door_pos
        unlock_pose = gymapi.Transform(
            p=np_to_vec3(unlock_pose_p),
            r=rpy_to_quat([0, door_y_rot - diff, 0])
        )
        return lock_pose, unlock_pose

    def get_door_config_name(self, env_idx):
        if self._train_in_multi_env:
            env_type = self._env_type_by_env_idx[env_idx]
            return self._multi_env_types[env_type]
        else:
            return self._cfg['door']['type']

    def get_wall_config_name(self, env_idx):
        '''for compatibility of env wrappers..'''
        return self.get_door_config_name(env_idx)

    def _init_rews(self, cfg):
        self._prev_joints_diff = None
        self._alive_penalty = cfg['rews']['alive_penalty']
        self._task_bonus = cfg['rews']['task_bonus']
        self._handle_rotation_weight = cfg['rews']['handle_rotation_weight']
        self._pole_rotation_weight = cfg['rews']['pole_rotation_weight']
        self._force_penalty_weight = cfg['rews']['force_penalty_weight']
        self._approach_weight = cfg['rews']['approach_weight']

        self._target_handle_rotation = np.deg2rad(cfg['task']['target_handle_rotation'])
        self._target_pole_rotation = np.deg2rad(cfg['task']['target_pole_rotation'])
        self._handle_rotation_th = np.deg2rad(cfg['task']['pole_rotation_th'])
        self._pole_rotation_th = np.deg2rad(cfg['task']['handle_rotation_th'])

        self._debug_rews_dict = {
            'handle_rot': np.zeros((self.n_envs)),
            'pole_rot': np.zeros((self.n_envs)),
            'approach': np.zeros((self.n_envs)),
            'force_penalty': np.zeros((self.n_envs)),
            'alive': np.zeros((self.n_envs)),
            'task_bonus': np.zeros((self.n_envs)),
        }
        # For each key in the above rews dict, save a list of all the returns associated.
        self._log_debug_rews_dict = {}
        for k in self._debug_rews_dict.keys():
            self._log_debug_rews_dict[k] = []

    def _init_obs_space(self, cfg):
        obs_space = super()._init_obs_space(cfg)
        limits_low = np.concatenate([
            obs_space.low,
            [-np.pi/2] + [0] * 3
        ])
        limits_high = np.concatenate([
            obs_space.high,
            [np.pi/2] + [1] * 3
        ])

        self._n_obs_wo_ctrlr = len(limits_low)
        if cfg['task']['use_controller_features']:
            limits_low = np.concatenate([limits_low, np.repeat(self._controller_feature_extractor[0].low, self.n_low_level_controllers)])
            limits_high = np.concatenate([limits_high, np.repeat(self._controller_feature_extractor[0].high, self.n_low_level_controllers)])

        obs_space = Box(limits_low, limits_high, dtype=np.float32)

        return obs_space

    def step_expanded_mdp(self, all_actions, update_obs_cb):
        all_obs = self._compute_obs(all_actions)
        all_obs = update_obs_cb(all_obs, all_actions)
        all_rews = np.zeros((self.n_envs))
        assert np.sum(self._step_counts > self._max_steps) == 0, \
            "Cannot have episodes beyond max time"
        all_dones = np.zeros((self.n_envs), dtype=np.bool)
        all_infos = self._compute_infos(all_obs, all_actions, all_rews, all_dones)
        return all_obs, all_rews, all_dones, all_infos

    def _compute_obs(self, all_actions):
        """0:7 joint angles, 7 gripper, 8-15 ee pos+quat, 15:18 ee force, 18 pole angle, 19 handle angle, 20:23 handle pos"""
        all_obs = super()._compute_obs(all_actions)
        door_joints = np.zeros((self.n_envs, 2))
        handle_pos = np.zeros((self.n_envs, 3))
        for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
            door_name = self._door_names[self.get_door_config_name(env_idx)]
            door_ah = self._scene.ah_map[env_idx][door_name]
            door_asset = self._doors[self.get_door_config_name(env_idx)]
            door_joints[env_idx, :] = door_asset.get_joints(env_ptr, door_ah)
            handle_jnt = door_joints[env_idx, 1]
            assert self._target_handle_rotation > 0, "handle joint target must be greater than 0"
            block_force = np.abs(self._lock.get_rb_ct_forces(env_idx, self._lock_name)[0, 2])
            if handle_jnt > np.deg2rad(30) and block_force > 5:
                # unlock when handle is turned enough
                lock_ah = self._scene.ah_map[env_idx][self._lock_name]
                unlock_pose = self._unlock_poses[env_idx]
                self._lock.set_rb_transforms(env_ptr, lock_ah, [unlock_pose])

            handle_idx = door_asset.rb_names_map['door_handle']
            handle_transform = door_asset.get_rb_transforms(env_ptr, door_ah)[handle_idx]
            handle_tra = vec3_to_np(handle_transform.p)
            handle_pos[env_idx, :] = handle_tra

        all_obs = np.c_[all_obs, door_joints[:,0], handle_pos]
        if self._cfg['task']['use_controller_features']:
            all_obs = np.c_[all_obs, np.zeros((self.n_envs, self.n_low_level_controllers * self.n_ctrlr_features))]
            for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
                franka_ah = self._scene.ah_map[env_idx][self._franka_name]
                ee_tf = self._frankas[env_idx].get_desired_ee_transform(env_idx, self._franka_name)
                current_tra = vec3_to_np(ee_tf.p)
                current_R = quat_to_rot(ee_tf.r)
                current_force = self._frankas[env_idx].get_ee_ct_forces(env_ptr, franka_ah)

                ctrlr_features = self._controller_feature_extractor[env_idx] \
                                    .get_features(current_tra, current_R, current_force)

                all_obs[env_idx, self.n_obs_wo_ctrlr:] = ctrlr_features.flatten()

        return all_obs

    def _compute_joints_diff(self):
        joints = np.zeros((self.n_envs, 2))
        for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
            door_name = self._door_names[self.get_door_config_name(env_idx)]
            door_ah = self._scene.ah_map[env_idx][door_name]
            door_asset = self._doors[self.get_door_config_name(env_idx)]
            joints[env_idx, :] = door_asset.get_joints(env_ptr, door_ah)
        joints_diff = np.abs(joints - np.array([self._target_pole_rotation, self._target_handle_rotation]))
        return joints_diff

    def _compute_ee_dists_to_handle(self, all_obs=None):
        if all_obs is None:
            all_obs = self._compute_obs(None)
        ee_pos = all_obs[:, 8:11]
        handle_pos = all_obs[:, 19:22]
        dists = np.linalg.norm(ee_pos - handle_pos, axis=1)
        assert len(dists) == self.n_envs
        return dists

    def _compute_rews(self, all_obs, all_actions):
        ee_dists_to_handle = self._compute_ee_dists_to_handle(all_obs)
        approach_improvements = self._prev_ee_dists_to_handle - ee_dists_to_handle
        door_joints_diff = self._compute_joints_diff()
        angle_improvements = self._prev_joints_diff - door_joints_diff
        pole_improvements = angle_improvements[:, 0]
        handle_improvements = angle_improvements[:, 1]
        contact_force = np.linalg.norm(all_obs[:, 15:18], axis=1)
        rews = self._pole_rotation_weight * pole_improvements + \
               self._handle_rotation_weight * handle_improvements + \
               self._approach_weight * approach_improvements + \
               self._force_penalty_weight * contact_force + \
               self._alive_penalty

        self._prev_joints_diff = door_joints_diff
        self._prev_ee_dists_to_handle = ee_dists_to_handle

        self._debug_rews_dict['handle_rot'] += self._handle_rotation_weight * handle_improvements
        self._debug_rews_dict['pole_rot'] += self._pole_rotation_weight * pole_improvements
        self._debug_rews_dict['approach'] += self._approach_weight * approach_improvements
        self._debug_rews_dict['force_penalty'] += self._force_penalty_weight * contact_force

        task_dones = all_obs[:, 18] < self._target_pole_rotation
        rews[task_dones] += self._task_bonus
        self._debug_rews_dict['task_bonus'][task_dones] += self._task_bonus

        return rews

    def _compute_dones(self, all_obs, all_actions, all_rews):
        time_dones = self._step_counts >= self._max_steps
        # task_dones = self._prev_joints_diff[:, 0] < self._pole_rotation_th
        task_dones = all_obs[:, 18] < self._target_pole_rotation
        dones = np.bitwise_or(time_dones, task_dones)
        return dones

    def _compute_infos(self, all_obs, all_actions, all_rews, all_dones):
        task_dones = all_obs[:, 18] < self._target_pole_rotation
        all_infos = []
        for i in range(self.n_envs):
            info = {}
            info['is_success'] = 1 if task_dones[i] else 0

            # PPO2 code uses the episode key for some reason to save data.
            if 'rl' in self._cfg:
                if self._cfg['rl']['algo'] == 'ppo':
                    info['r'] = self.episode_rewards[i] + all_rews[i]
                    info['l'] = self.step_counts[i] + 1
                    info = {'episode': info}

            all_infos.append(info)
        self._prev_infos = all_infos
        return all_infos

    def _update_controllers(self):
        '''update non-static controllers (attractors to moving targets)'''
        attractors_cfg = self._cfg['action']['attractors']

        for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
            high_level_ctrlr = self._high_level_controllers[env_idx]

            door_name = self._door_names[self.get_door_config_name(env_idx)]
            door_ah = self._scene.ah_map[env_idx][door_name]
            door_asset = self._doors[self.get_door_config_name(env_idx)]
            handle_idx = door_asset.rb_names_map['door_handle']
            pole_idx = door_asset.rb_names_map['door_pole']

            handle_transform = door_asset.get_rb_transforms(env_ptr, door_ah)[handle_idx]
            handle_tra = vec3_to_np(handle_transform.p)

            pole_transform = door_asset.get_rb_transforms(env_ptr, door_ah)[pole_idx]
            pole_tra = vec3_to_np(pole_transform.p)
            pole_rot = quat_to_rot(pole_transform.r)

            handle_rot = quat_to_rot(handle_transform.r)
            handle_p = handle_rot @ np.array([-0.08, 0., 0.]).reshape((-1, 1))
            handle_p = handle_p.flatten() + handle_tra
            handle_r = quat_to_rot(handle_transform.r)

            if attractors_cfg['handle_err']:
                high_level_ctrlr.update_low_level_controller('handle_err', None, handle_p)
            if attractors_cfg['pole_rot']:
                high_level_ctrlr.update_low_level_controller('pole_rot', np.array([0, 0, 1]), -handle_r[:, 2])
            if attractors_cfg['handle_force']:
                handle_force = -50 * handle_r[:, 1]
                high_level_ctrlr.update_low_level_controller('handle_force', -handle_r[:, 1], handle_force)
            if attractors_cfg['door_force']:
                door_force = 50 * handle_r[:, 2]
                high_level_ctrlr.update_low_level_controller('door_force', handle_r[:, 2], door_force)
            if attractors_cfg['handle_curl']:
                high_level_ctrlr.update_low_level_controller('handle_curl', -handle_r[:, 1], handle_p)
            if attractors_cfg['pole_curl']:
                high_level_ctrlr.update_low_level_controller('pole_curl', pole_rot[:, 2], pole_tra)

    def _init_controllers(self):
        '''Initialize the high level and low level controllers.'''
        self._high_level_controllers = []
        self._controller_feature_extractor = []
        attractors_cfg = self._cfg['action']['attractors']
        dx = self._cfg['action']['dx']
        df = self._cfg['action']['df']
        dtheta = np.deg2rad(self._cfg['action']['dtheta'])
        dl = self._cfg['action']['dl']
        kI = self._cfg['action']['kI_f']

        for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
            low_level_ctrlr_dict = OrderedDict()
            low_level_ctrlr_dict['null'] = NullAttractorController()

            door_name = self._door_names[self.get_door_config_name(env_idx)]
            door_ah = self._scene.ah_map[env_idx][door_name]
            door_asset = self._doors[self.get_door_config_name(env_idx)]
            handle_idx = door_asset.rb_names_map['door_handle']
            pole_idx = door_asset.rb_names_map['door_pole']

            pole_transform = door_asset.get_rb_transforms(env_ptr, door_ah)[pole_idx]
            pole_tra = vec3_to_np(pole_transform.p)
            pole_rot = quat_to_rot(pole_transform.r)

            handle_transform = door_asset.get_rb_transforms(env_ptr, door_ah)[handle_idx]
            handle_tra = vec3_to_np(handle_transform.p)
            handle_rot = quat_to_rot(handle_transform.r)
            handle_p = handle_rot @ np.array([-0.08, 0., 0.]).reshape((-1, 1))
            handle_p = handle_p.flatten() + handle_tra
            handle_r = quat_to_rot(handle_transform.r)

            if attractors_cfg['close_gripper']:
                low_level_ctrlr_dict['close_gripper'] = FrankaGripperController('close')
            if attractors_cfg['open_gripper']:
                low_level_ctrlr_dict['open_gripper'] = FrankaGripperController('open')
            if attractors_cfg['handle_err']:
                low_level_ctrlr_dict['handle_err'] = ErrorAxisPositionAttractorController(handle_p, dx)
            if attractors_cfg['pole_rot']:
                low_level_ctrlr_dict['pole_rot'] = RotationAttractorController(np.array([0, 0, 1]), -handle_r[:, 2], dtheta)
            if attractors_cfg['handle_rot_final']:
                low_level_ctrlr_dict['handle_rot_final'] = RotationAttractorController(np.array([1, 0, 0]), handle_r[:, 1], dtheta)
            if attractors_cfg['handle_force']:
                handle_force = -20 * handle_r[:, 1]
                low_level_ctrlr_dict['handle_force'] = ForceAttractorController(-handle_r[:, 1], handle_force, df, kI=kI)
            if attractors_cfg['door_force']:
                door_force = 20 * handle_r[:, 2]
                low_level_ctrlr_dict['door_force'] = ForceAttractorController(handle_r[:, 2], door_force, df, kI=kI)
            if attractors_cfg['handle_curl']:
                low_level_ctrlr_dict['handle_curl'] = CurlPositionAttractorController(-handle_r[:, 1], handle_p, dl)
            if attractors_cfg['pole_curl']:
                low_level_ctrlr_dict['pole_curl'] = CurlPositionAttractorController(pole_rot[:, 2], pole_tra, dl)
            if attractors_cfg['handle_rot_init']:
                low_level_ctrlr_dict['handle_rot_init'] = RotationAttractorController(np.array([1, 0, 0]), handle_r[:, 0], dtheta)

            high_level_ctrlr = ComposedAttractorControllers(low_level_ctrlr_dict)
            self._high_level_controllers.append(high_level_ctrlr)
            self._controller_feature_extractor.append(ControllersListFeatureExtractor(
                high_level_ctrlr.low_level_controllers, is_2d=False
            ))

    def controller_combo_to_idx(self, controller_idxs):
        if self._cfg['action']['type'] == 'combo':
            return self._low_level_controller_combos_map[tuple(controller_idxs)]
        else:
            raise ValueError('Can only call this function when using combo action type!')

    @property
    def n_low_level_controllers(self):
        if self._cfg['action']['type'] not in ('combo', 'priority', 'discrete_one_controller'):
            raise ValueError('Can only access this property when using combo or priority action type!')
        return self._n_low_level_controllers

    @property
    def n_ctrlr_features(self):
        return self._controller_feature_extractor[0].dim

    @property
    def n_obs_wo_ctrlr(self):
        return self._n_obs_wo_ctrlr

    @property
    def last_infos(self):
        return self._prev_infos

    def _init_action_space(self, cfg):
        super()._init_action_space(cfg)
        self._action_type = cfg['action']['type']
        self._auto_clip_actions = cfg['action']['auto_clip_actions']
        self._raw_translation_goals = [gymapi.Vec3() for _ in range(self.n_envs)]
        self._raw_rotation_goals = [gymapi.Quat() for _ in range(self.n_envs)]
        self._current_controller_idxs = [[] for _ in range(self.n_envs)]

        self._current_vic_actions = np.zeros((self.n_envs, 8))
        self._current_vic_actions[:, -1] = 0 # close gripper
        self._current_vic_actions[:, 6] = self._cfg['action']['attractor_stiffness']

        self._raw_gripper_goal = [None, ] * self.n_envs
        self._prev_gripper_status = np.array(['close', ] * self.n_envs)

        if self._action_type == 'raw':
            limits = np.array(cfg['action']['raw']['limits'])
            limits[3:6] = np.deg2rad(limits[3:6])
            action_space = Box(-limits, limits, dtype=np.float32)
        elif self._action_type in ('combo', 'priority', 'discrete_one_controller'):
            self._init_controllers()
            self._n_low_level_controllers = len(self._high_level_controllers[0].low_level_controllers)
            if self._action_type == 'combo':
                self._low_level_controller_combos = list(permutations(
                        [0, 0] + list(range(self._n_low_level_controllers))
                    , 3))
                self._low_level_controller_combos_map = {combo : i for i, combo in enumerate(self._low_level_controller_combos)}
                action_space = Discrete(len(self._low_level_controller_combos))
            elif self._action_type == 'priority':
                action_space = Box(np.zeros(self._n_low_level_controllers), np.ones(self._n_low_level_controllers), dtype=np.float32)
            elif self._action_type == 'discrete_one_controller':
                action_space = Discrete(self._n_low_level_controllers)
            else:
                raise ValueError("Invalid action type.")
        else:
            raise ValueError('Unknown action type: {}'.format(self._action_type))

        return action_space

    def _apply_inter_actions(self, _, t_inter_step, n_inter_steps):
        if self._action_type in ('combo', 'priority', 'discrete_one_controller'):
            self._update_controllers()

        for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
            franka_ah = self._scene.ah_map[env_idx][self._franka_name]

            if t_inter_step % self._cfg['action']['n_interp_steps'] == 0:
                if self._action_type == 'raw':
                    dt = 1 / n_inter_steps
                    delta_position = self._raw_translation_goals[env_idx] * dt
                    delta_quat = slerp_quat(gymapi.Quat(), self._raw_rotation_goals[env_idx], dt)
                    gripper_action = self._raw_gripper_goal
                else:
                    high_level_ctrlr = self._high_level_controllers[env_idx]
                    dt = 1 / self._cfg['action']['n_interp_steps']
                    ee_tf = self._frankas[env_idx].get_ee_transform(env_ptr, self._franka_name)
                    current_tra = vec3_to_np(ee_tf.p)
                    current_R = quat_to_rot(ee_tf.r)
                    current_force = self._frankas[env_idx].get_ee_ct_forces(env_ptr, franka_ah)

                    delta_translation_target, delta_rotation_target = \
                        high_level_ctrlr.compute_controls(
                            current_tra, current_R, current_force, self._current_controller_idxs[env_idx])
                    gripper_action = high_level_ctrlr.compute_gripper_controls(self._current_controller_idxs[env_idx])

                    delta_position = delta_translation_target.p * dt
                    delta_quat = slerp_quat(gymapi.Quat(), delta_rotation_target.r, dt)

                if gripper_action is None:
                    gripper_action = self._prev_gripper_status[env_idx]

                self._current_vic_actions[env_idx, :3] = vec3_to_np(delta_position)
                self._current_vic_actions[env_idx, 3:6] = quat_to_rpy(delta_quat)

                current_gripper_width = self._frankas[env_idx].get_gripper_width(env_ptr, franka_ah)
                if gripper_action == 'open':
                    target_gripper_width = current_gripper_width + self._cfg['action']['dg']
                else:
                    target_gripper_width = current_gripper_width - self._cfg['action']['dg']
                self._current_vic_actions[env_idx, 7] = target_gripper_width

                self._prev_gripper_status[env_idx] = gripper_action

        super()._apply_actions(self._current_vic_actions)

    def _apply_actions(self, all_actions):
        for env_idx in range(self.n_envs):
            action = all_actions[env_idx]

            if self._action_type == 'raw':
                self._raw_translation_goals[env_idx] = gymapi.Vec3(*action[:3])
                self._raw_rotation_goals[env_idx] = rpy_to_quat(action[3:6])
                if action[6] > 0.333:
                    self._raw_gripper_goal = 'open'
                elif action[6] < -0.333:
                    self._raw_gripper_goal = 'close'
                else:
                    self._raw_gripper_goal = None
            elif self._action_type in ('combo', 'priority'):
                if self._action_type == 'combo':
                    controller_idxs = self._low_level_controller_combos[int(action)]
                elif self._action_type == 'priority':
                    controller_idxs = np.argsort(action)[:-4:-1]
                self._current_controller_idxs[env_idx] = controller_idxs
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
                raise ValueError("Invalid action type")

    def _reset(self, env_idxs):
        super()._reset(env_idxs)
        for env_idx in env_idxs:
            env_ptr = self._scene.env_ptrs[env_idx]
            door_asset = self._doors[self.get_door_config_name(env_idx)]
            door_name = self._door_names[self.get_door_config_name(env_idx)]
            lock_ah = self._scene.ah_map[env_idx][self._lock_name]
            door_ah = self._scene.ah_map[env_idx][door_name]

            if self._sample_door_pose:
                door_p = (np.random.rand(3) * 2 - 1) * self._door_pos_sample_range + self._mean_door_pos
                door_r = (np.random.rand() * 2 - 1) * self._door_rot_sample_range + self._mean_door_rot
                door_pose = gymapi.Transform(
                    p=np_to_vec3(door_p),
                    r=rpy_to_quat([0, door_r, 0])
                )
                door_asset.set_rb_transforms(env_ptr, door_ah, [door_pose])
                lock_pose, unlock_pos = self._calc_lock_unlock_pose(door_p, door_r)
                self._lock_poses[env_idx] = lock_pose
                self._lock.set_rb_transforms(env_ptr, lock_ah, [lock_pose])
                self._unlock_poses[env_idx] = unlock_pos
            else:
                lock_pose = self._lock_poses[env_idx]
                self._lock.set_rb_transforms(env_ptr, lock_ah, [lock_pose])

            door_asset.set_joints(env_ptr, self._scene.ah_map[env_idx][door_name], np.zeros(door_asset.n_dofs))
            for rew_key, rew_val in self._debug_rews_dict.items():
                self._log_debug_rews_dict[rew_key].append(rew_val[env_idx])
                rew_val[env_idx] = 0.0
            if self._sample_init_ee_pos:
                franka_init_jnts = self._frankas[env_idx].INIT_JOINTS.copy()
                franka_init_jnts[-4::] += (2 * np.random.rand(4) - 1) * np.abs(self._ee_pos_sample_range)
                franka_ah = self._scene.ah_map[env_idx][self._franka_name]
                self._frankas[env_idx].set_joints(env_ptr, franka_ah, franka_init_jnts)

        self._prev_joints_diff = self._compute_joints_diff()
        self._prev_ee_dists_to_handle = self._compute_ee_dists_to_handle()
