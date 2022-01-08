import numpy as np
from itertools import permutations
from collections import OrderedDict

from isaac_gym import gymapi
from isaac_gym_utils.assets import GymURDFAsset, GymBoxAsset, GymFranka
from isaac_gym_utils.math_utils import (rpy_to_quat, transform_to_np, np_to_vec3, vec3_to_np, 
                                        quat_to_rpy, quat_to_rot, slerp_quat, rot_to_quat)

from gym.spaces import Box, Discrete
from autolab_core import RigidTransform
from isaac_gym_utils.rl import GymFrankaVecEnv

from object_axes_ctrlrs.controllers.projected_axes_attractors import \
    PositionAttractorController, RotationAttractorController, \
    ForceAttractorController, NullAttractorController, \
    ComposedAttractorControllers, ErrorAxisPositionAttractorController, \
    PositionNoMovementAlongAxisController, ControllersListFeatureExtractor


class GymFrankaHexScrewVecEnv(GymFrankaVecEnv):

    def _fill_scene(self, cfg):
        self._table = GymBoxAsset(self._scene.gym, self._scene.sim, **cfg['table']['dims'],
                            shape_props=cfg['table']['shape_props'],
                            asset_options=cfg['table']['asset_options']
                            )
        self._table_name = 'table0'
        table_pose = gymapi.Transform(
            p=gymapi.Vec3(cfg['table']['dims']['width']/3, cfg['table']['dims']['height']/2, 0)
        )
        self._scene.add_asset(self._table_name, self._table, table_pose)

        self._actuation_mode = self._ACTUATION_MODE_MAP[cfg['franka']['action']['mode']]
        franka_pose = gymapi.Transform(
            p=gymapi.Vec3(0, cfg['table']['dims']['height'] + 0.01, 0),
            r=rpy_to_quat([np.pi/2, np.pi/2, -np.pi/2])
        )
        self._franka_name = 'franka0'
        self._hex_screw_base_name = 'hex_screw_base0'

        self._frankas = []
        self._hex_screw_bases = []
        self._hex_screw_scale_by_env_list = []
        original_custom_ee_offset = np.array(cfg['franka']['custom_ee_offset'])
        for env_idx in range(self.n_envs):
            n_suffix = env_idx % len(cfg['hex_screw_urdf_suffixes'])
            self._hex_screw_scale_by_env_list.append(cfg['hex_screw_attractor_scales'][n_suffix])

            if len(self._frankas) < n_suffix + 1:
                suffix = cfg['hex_screw_urdf_suffixes'][n_suffix]
                cfg['franka']['urdf'] = cfg['franka']['urdf_template'].format(suffix)
                cfg['hex_screw_base']['urdf'] = cfg['hex_screw_base']['urdf_template'].format(suffix)
                cfg['franka']['custom_ee_offset'] = (cfg['hex_screw_attractor_scales'][n_suffix] * \
                                                    original_custom_ee_offset).tolist()

                self._frankas.append(GymFranka(cfg['franka'], self._scene.gym, self._scene.sim,
                                    actuation_mode=self._actuation_mode))
                self._hex_screw_bases.append(GymURDFAsset(
                                    cfg['hex_screw_base']['urdf'], self._scene.gym, self._scene.sim, 
                                    shape_props=cfg['hex_screw_base']['shape_props'],
                                    rb_props=cfg['hex_screw_base']['rb_props'],
                                    dof_props=cfg['hex_screw_base']['dof_props'],
                                    asset_options=cfg['hex_screw_base']['asset_options'],
                                    assets_root=cfg['assets_root']
                                ))
            else:
                self._frankas.append(self._frankas[n_suffix])
                self._hex_screw_bases.append(self._hex_screw_bases[n_suffix])
        
            self._scene.add_asset(self._franka_name, self._frankas[-1], franka_pose, collision_filter=2, envs=[env_idx])
            self._scene.add_asset(self._hex_screw_base_name, self._hex_screw_bases[-1], gymapi.Transform(), collision_filter=4, envs=[env_idx])
        
        self._prev_infos = None
        self._max_steps = cfg['task']['max_steps']

    def _init_rews(self, cfg):
        self._prev_angles_to_target = None
        self._prev_dists_to_base = None
        self._alive_penalty = cfg['rews']['alive_penalty']
        self._task_bonus = cfg['rews']['task_bonus']
        self._angle_weight = cfg['rews']['angle_weight']
        self._approach_weight = cfg['rews']['approach_weight']
        self._vert_angle_penalty = cfg['rews']['vert_angle_penalty']
        self._force_penalty = cfg['rews']['force_penalty']
        self._task_angle_th = np.deg2rad(cfg['task']['task_angle_th'])
        self._task_angle_target_delta = np.deg2rad(cfg['task']['task_angle_target'])
        self._task_angle_targets = np.ones(self.n_envs) * self._task_angle_target_delta
        self._rot_force_th = cfg['task']['rot_force_th']
        self._screw_depth_th = cfg['task']['screw_depth_th']

    def _init_obs_space(self, cfg):
        obs_space = super()._init_obs_space(cfg)
        
        hex_screw_pose_low = [-5] * 3 + [-1] * 4
        hex_screw_pose_high = [5] * 3 + [1] * 4

        hex_screw_obs_low = hex_screw_pose_low
        hex_screw_obs_high = hex_screw_pose_high

        if self._cfg['task'].get('add_hex_screw_offset_to_obs', False):
            hex_screw_offset_low = [0.0]
            hex_screw_offset_high = [2.0]
            hex_screw_obs_low += hex_screw_offset_low
            hex_screw_obs_high += hex_screw_offset_high

        limits_low = np.concatenate([obs_space.low, hex_screw_obs_low])
        limits_high = np.concatenate([obs_space.high, hex_screw_obs_high])

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
        all_obs = super()._compute_obs(all_actions)
        box_pose_obs = np.zeros((self.n_envs, 7))
        for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
            ah = self._scene.ah_map[env_idx][self._hex_screw_base_name]
            screw_tf = self._hex_screw_bases[env_idx].get_rb_transforms(env_ptr, ah)[1]
            box_pose_obs[env_idx, :] = transform_to_np(screw_tf, format='wxyz')
        if self._cfg['task'].get('add_hex_screw_offset_to_obs', False):
            hex_screw_scale_obs = np.array(self._hex_screw_scale_by_env_list).reshape(-1, 1)
            all_obs = np.c_[all_obs, box_pose_obs, hex_screw_scale_obs]
        else:
            all_obs = np.c_[all_obs, box_pose_obs]
        
        # Add all controller features
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

    def _compute_screw_angles_to_target(self, env_idxs):
        angles_to_target = []
        for env_idx in env_idxs:
            env_ptr = self._scene.env_ptrs[env_idx]
            base_ah = self._scene.ah_map[env_idx][self._hex_screw_base_name]
            joints = self._hex_screw_bases[env_idx].get_joints(env_ptr, base_ah)
            angles_to_target.append(self._task_angle_targets[env_idx] - joints[0])

        return np.abs(angles_to_target)

    def _compute_ee_dists_to_base(self, env_idxs):
        dists = []
        for env_idx in env_idxs:
            env_ptr = self._scene.env_ptrs[env_idx]
            base_ah = self._scene.ah_map[env_idx][self._hex_screw_base_name]
            base_tf = self._hex_screw_bases[env_idx].get_rb_transforms(env_ptr, base_ah)[0]
            ee_tf = self._frankas[env_idx].get_ee_transform(env_ptr, self._franka_name)

            dist = np.linalg.norm(vec3_to_np(ee_tf.p - base_tf.p))
            dists.append(dist)
        dists = np.array(dists)
        return dists

    def _compute_wrench_vert_angles(self, env_idxs):
        angles = []
        for env_idx in env_idxs:
            env_ptr = self._scene.env_ptrs[env_idx]
            ee_tf = self._frankas[env_idx].get_ee_transform(env_ptr, self._franka_name)
            ee_z_axis = quat_to_rot(ee_tf.r)[:, 2]

            angle = np.abs(np.arccos(np.clip(-ee_z_axis[1], -1, 1)))
            angles.append(angle)

        return np.abs(angles)

    def _compute_rews(self, all_obs, all_actions):
        env_idxs = list(range(self.n_envs))
        cur_angles_to_target = self._compute_screw_angles_to_target(env_idxs)
        angle_improvements = self._prev_angles_to_target - cur_angles_to_target

        cur_dists_to_base = self._compute_ee_dists_to_base(env_idxs)
        approach_improvements = self._prev_dists_to_base - cur_dists_to_base

        angle_to_vert = self._compute_wrench_vert_angles(env_idxs)

        ct_forces = np.array([franka.get_rb_ct_forces(env_idx, self._franka_name) for env_idx, franka in enumerate(self._frankas)])
        ct_force_mags = np.sum(np.linalg.norm(ct_forces, axis=2), axis=1)

        rews = self._angle_weight * angle_improvements + \
               self._approach_weight * approach_improvements + \
               self._vert_angle_penalty * angle_to_vert + \
               self._force_penalty * ct_force_mags + \
               self._alive_penalty

        self._prev_angles_to_target = cur_angles_to_target
        self._prev_dists_to_base = cur_dists_to_base

        rews[self._prev_angles_to_target < self._task_angle_th] += self._task_bonus

        return rews

    def _compute_dones(self, all_obs, all_actions, all_rews):
        time_dones = self._step_counts >= self._max_steps
        task_dones = self._prev_angles_to_target < self._task_angle_th
        dones = time_dones | task_dones
        return dones

    def _compute_infos(self, all_obs, all_actions, all_rews, all_dones):
        task_dones = self._prev_angles_to_target < self._task_angle_th
        all_infos = []
        for i in range(self.n_envs):
            info = {}
            info['is_success'] = 1 if task_dones[i] else 0
            if 'rl' in self._cfg:
                if self._cfg['rl']['algo'] == 'ppo':
                    info['r'] = self.episode_rewards[i] + all_rews[i]
                    info['l'] = self.step_counts[i] + 1
                    info = {'episode': info}

            all_infos.append(info)
        self._prev_infos = all_infos
        return all_infos

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
        screw_normal_offset = np.array(self._cfg['action']['screw_normal_offset'])
        for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
            low_level_ctrlr_dict = OrderedDict()
            low_level_ctrlr_dict['null'] = NullAttractorController()

            base_ah = self._scene.ah_map[env_idx][self._hex_screw_base_name]
            screw_tf = self._hex_screw_bases[env_idx].get_rb_transforms(env_ptr, base_ah)[1]
            screw_tra = vec3_to_np(screw_tf.p)
            screw_R = quat_to_rot(screw_tf.r)
            screw_normal = screw_R[:, 1]

            if attractors_cfg['screw_normal']:
                low_level_ctrlr_dict['screw_normal_offset'] = PositionAttractorController(screw_normal, screw_tra + screw_normal_offset, dx)
            if attractors_cfg['screw_rot']:
                R = RigidTransform.y_axis_rotation(np.deg2rad(self._cfg['task']['ctrlr_angle_target']))
                low_level_ctrlr_dict['screw_rot_ccw'] = RotationAttractorController(np.array([1, 0, 0]), R @ screw_R[:, 0], dtheta)
                low_level_ctrlr_dict['screw_rot_cw'] = RotationAttractorController(np.array([1, 0, 0]), R.T @ screw_R[:, 0], dtheta)
                low_level_ctrlr_dict['screw_rot_z_to_ny'] = RotationAttractorController(np.array([0, 0, 1]), -screw_normal, dtheta)
            if attractors_cfg['screw_pos']:
                low_level_ctrlr_dict['screw_pos_offset'] = ErrorAxisPositionAttractorController(screw_tra + screw_normal_offset, dx)
            if attractors_cfg['screw_force']:
                screw_force_th = self._cfg['task'].get('max_attractor_force_th', 20)
                screw_force = -screw_force_th * screw_normal
                low_level_ctrlr_dict['screw_force'] = ForceAttractorController(screw_normal, screw_force, df, kI=kI)
            if attractors_cfg['screw_no_normal_movement']:
                low_level_ctrlr_dict['screw_no_normal_movement'] = PositionNoMovementAlongAxisController(screw_normal)

            high_level_ctrlr = ComposedAttractorControllers(low_level_ctrlr_dict)
            self._high_level_controllers.append(high_level_ctrlr)
            self._controller_feature_extractor.append(ControllersListFeatureExtractor(
                high_level_ctrlr.low_level_controllers, is_2d=False
            ))

    def controller_combo_to_idx(self, controller_idxs):
        if self._cfg['action']['type'] == 'combo':
            return self._low_level_controller_combos_map[tuple(controller_idxs)]
        else: raise ValueError('Can only call this function when using combo action type!') 
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
            raise ValueError('Unknown action type: {}'.format(self._action_type))
            
        return action_space        

    def _apply_inter_actions(self, _, t_inter_step, n_inter_steps):
        for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
            franka_ah = self._scene.ah_map[env_idx][self._franka_name]
            ee_tf = self._frankas[env_idx].get_ee_transform(env_ptr, self._franka_name)
            current_tra = vec3_to_np(ee_tf.p)

            if t_inter_step % self._cfg['action']['n_interp_steps'] == 0:
                if self._action_type == 'raw':
                    dt = 1 / n_inter_steps
                    delta_position = self._raw_translation_goals[env_idx] * dt
                    delta_quat = slerp_quat(gymapi.Quat(), self._raw_rotation_goals[env_idx], dt)
                else:
                    high_level_ctrlr = self._high_level_controllers[env_idx]
                    dt = 1 / self._cfg['action']['n_interp_steps']
                    current_R = quat_to_rot(ee_tf.r)
                    current_force = self._frankas[env_idx].get_ee_ct_forces(env_ptr, franka_ah)
                    delta_translation_target, delta_rotation_target = \
                        high_level_ctrlr.compute_controls(
                            current_tra, current_R, current_force, self._current_controller_idxs[env_idx])

                    delta_position = delta_translation_target.p * dt
                    delta_quat = slerp_quat(gymapi.Quat(), delta_rotation_target.r, dt)

                self._current_vic_actions[env_idx, :3] = vec3_to_np(delta_position)

                if self._cfg['action']['only_y_axis_rot']:
                    delta_R = quat_to_rot(delta_quat)
                    if not np.allclose(delta_R[1, :], [0, 1, 0]):
                        delta_R[:, 1] = [0, 1, 0]
                        delta_R[1, :] = [0, 1, 0]
                        delta_R[:, 0] /= np.linalg.norm(delta_R[:, 0])
                        delta_R[:, 2] /= np.linalg.norm(delta_R[:, 2])
                        delta_quat = rot_to_quat(delta_R)
                self._current_vic_actions[env_idx, 3:6] = quat_to_rpy(delta_quat)

            # lock screw if not enough force is applied
            base_ah = self._scene.ah_map[env_idx][self._hex_screw_base_name]
            screw_pos = self._hex_screw_bases[env_idx].get_rb_poses_as_np_array(env_ptr, base_ah)[1]
            base_y = self._hex_screw_bases[env_idx].get_rb_poses_as_np_array(env_ptr, base_ah)[1][1]
            screw_rb_idx = self._hex_screw_bases[env_idx].rb_names_map['screw']
            screw_ct_forces = self._hex_screw_bases[env_idx].get_rb_ct_forces(env_idx, self._hex_screw_base_name)[screw_rb_idx]
            screw_y_force = screw_ct_forces[1]

            nearby_th = 0.0012
            is_nearby_xz = abs(screw_pos[0] - current_tra[0]) <= nearby_th and abs(screw_pos[2] - current_tra[2]) <= nearby_th
            if screw_y_force > self._rot_force_th and np.abs(base_y - current_tra[1]) < self._screw_depth_th and is_nearby_xz:
                self._hex_screw_bases[env_idx].set_dof_props(env_ptr, base_ah, {'driveMode': ['DOF_MODE_NONE']})
            else:
                self._hex_screw_bases[env_idx].set_dof_props(env_ptr, base_ah, {
                    'driveMode': ['DOF_MODE_POS'],
                    'stiffness': [1e6],
                    'damping': [1],
                    'effort': [1e6]
                })
                current_joints = self._hex_screw_bases[env_idx].get_joints(env_ptr, base_ah)
                self._hex_screw_bases[env_idx].set_joints_targets(env_ptr, base_ah, current_joints)

        super()._apply_actions(self._current_vic_actions)

    def _apply_actions(self, all_actions):
        for env_idx in range(self.n_envs):
            action = all_actions[env_idx]
            if self._action_type == 'raw':
                self._raw_translation_goals[env_idx] = gymapi.Vec3(*action[:3])
                self._raw_rotation_goals[env_idx] = rpy_to_quat(action[3:])
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

    def _reset(self, env_idxs):
        super()._reset(env_idxs)

        for env_idx in env_idxs:
            env_ptr = self._scene.env_ptrs[env_idx]
            base_ah = self._scene.ah_map[env_idx][self._hex_screw_base_name]
            old_base_tf, old_screw_tf = self._hex_screw_bases[env_idx].get_rb_transforms(env_ptr, base_ah)
            delta_screw_tf = old_base_tf.inverse() * old_screw_tf
            base_tf = gymapi.Transform(
                p=np_to_vec3(np.array([
                    (np.random.rand()*2 - 1) * 0.1 + 0.5,
                    self._cfg['table']['dims']['height'] + 0.01,
                    (np.random.rand()*2 - 1) * 0.1]))
                )
            self._hex_screw_bases[env_idx].set_rb_transforms(env_ptr, base_ah, [base_tf])
            if self._cfg['task']['sample_init_screw_angle']:
                init_screw_angle = np.random.uniform(np.deg2rad(-15), np.deg2rad(15))
            else:
                init_screw_angle = 0
            self._hex_screw_bases[env_idx].set_joints(env_ptr, base_ah, np.array([init_screw_angle]))
            self._task_angle_targets[env_idx] = init_screw_angle + self._task_angle_target_delta
            if self._cfg['action']['type'] != 'raw':
                screw_tf = base_tf * delta_screw_tf
                screw_tra = vec3_to_np(screw_tf.p)
                screw_R = quat_to_rot(screw_tf.r)
                screw_normal_offset = np.array(self._cfg['action']['screw_normal_offset'])
                high_level_ctrlr = self._high_level_controllers[env_idx]
                high_level_ctrlr.low_level_controllers_dict['screw_normal_offset'].target = screw_tra + screw_normal_offset
                high_level_ctrlr.low_level_controllers_dict['screw_pos_offset'].target = screw_tra + screw_normal_offset
                self._high_level_controllers[env_idx].reset()

        if self._prev_angles_to_target is None:
            self._prev_angles_to_target = self._compute_screw_angles_to_target(np.arange(self.n_envs))
        else:
            self._prev_angles_to_target[env_idxs] = self._compute_screw_angles_to_target(env_idxs)
        if self._prev_dists_to_base is None:
            self._prev_dists_to_base = self._compute_ee_dists_to_base(np.arange(self.n_envs))
        else:
            self._prev_dists_to_base[env_idxs] = self._compute_ee_dists_to_base(env_idxs)
        

