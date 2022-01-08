from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np
import quaternion

from isaac_gym import gymapi
from isaac_gym_utils.math_utils import angle_axis_between_axes, rot_from_np_quat, \
                                        np_quat_to_quat, np_to_vec3, \
                                        angle_axis_to_rotation, rotation_to_angle_axis


class BaseAttractorController(ABC):

    def __init__(self, axis, target, mag, target_type=''):
        axis_norm = np.linalg.norm(axis)
        assert np.isclose(axis_norm, 1) or np.isclose(axis_norm, 0), 'Axis must be of unit or zero norm!'
        assert len(axis) == 3, 'Dimension of axis must match dim!'
        assert len(target) == 3, 'Dimension of target must match dim!'
        assert mag >= 0, 'Magnitude must be positive'
        assert isinstance(axis, np.ndarray)
        assert isinstance(target, np.ndarray)

        self._axis = axis
        self._target = target
        self._mag = mag

        self._proj_space = np.outer(self._axis, self._axis)
        self._null_space = np.eye(3) - self._proj_space

        self._target_type = target_type

    @property
    def target_type(self):
        return self._target_type

    @property
    def proj_space(self):
        return self._proj_space

    @property
    def null_space(self):
        return self._null_space

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target):
        self._target = target

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, axis):
        self._axis = axis
        self._proj_space = np.outer(self._axis, self._axis)
        self._null_space = np.eye(3) - self._proj_space

    @property
    def mag(self):
        return self._mag

    @mag.setter
    def mag(self, mag):
        self._mag = mag

    @abstractmethod
    def compute_err(self, current):
        pass


class PositionAttractorController(BaseAttractorController):

    def compute_err(self, current):
        ''' current is an np array of len 3
        '''
        err = self._target - current

        proj_err = self._proj_space @ err
        proj_err_norm = np.linalg.norm(proj_err)

        if np.isclose(proj_err_norm, 0):
            return np.zeros(3)

        return min(self._mag, proj_err_norm) * proj_err / proj_err_norm


class ErrorAxisPositionAttractorController(PositionAttractorController):

    def __init__(self, target, mag, *args, **kwargs):
        super().__init__(np.zeros(3), target, mag, *args, **kwargs)

    def compute_err(self, current):
        ''' current is an np array of len 3
        '''
        err = self._target - current
        err_mag = np.linalg.norm(err)

        if np.isclose(err_mag, 0):
            self._axis = np.zeros(3)
            return np.zeros(3)

        self._axis = err / err_mag
        self._proj_space = np.outer(self._axis, self._axis)
        self._null_space = np.eye(3) - self._proj_space

        return min(self._mag, err_mag) * self._axis


class PositionAxisAttractorController(PositionAttractorController):

    def __init__(self, axis, mag, *args, **kwargs):
        super().__init__(axis, np.zeros(3), mag, *args, **kwargs)

    def compute_err(self, _):
        ''' current is an np array of len 3
        '''
        return self._mag * self._axis


class PositionNoMovementAlongAxisController(PositionAttractorController):

    def __init__(self, axis, *args, **kwargs):
        super().__init__(axis, np.zeros(3), 0, *args, **kwargs)


class CurlPositionAttractorController(PositionAttractorController):

    def compute_err(self, current):
        ''' current is an np array of len 3
        '''
        diff = current - self._target
        diff_norm = np.linalg.norm(diff)

        if np.isclose(diff_norm, 0):
            return np.zeros(3)

        diff_unit = diff / diff_norm
        err_angax = angle_axis_between_axes(diff_unit, self._axis)
        err_theta = np.linalg.norm(err_angax)

        if np.isclose(err_theta, 0):
            return np.zeros(3)

        err_angax = min(err_theta, self._mag / diff_norm) * err_angax / err_theta
        R = angle_axis_to_rotation(err_angax)
        
        target = self._target + R @ diff
        delta = target - current

        delta_dist = np.linalg.norm(delta)
        if np.isclose(delta_dist, 0):
            return np.zeros(3)

        ctrl_axis = delta / delta_dist
        self._proj_space = np.outer(ctrl_axis, ctrl_axis)
        self._null_space = np.eye(3) - self._proj_space

        return delta


class ForceAttractorController(BaseAttractorController):

    def __init__(self, *args, kI=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._total_err_I = np.zeros(3)
        self._high_err_I = np.ones(1) * 1000
        self._low_err_I = -self._high_err_I
        self._kI = kI

    def reset_total_err(self):
        self._total_err_I = np.zeros(3)

    def compute_err(self, current):
        ''' current is an np array of len 3
        '''
        if self._kI == 0:
            err = self._target - current

            proj_err = self._proj_space @ err
            proj_err_norm = np.linalg.norm(proj_err)

            if np.isclose(proj_err_norm, 0):
                return np.zeros(3)

            return min(self._mag, proj_err_norm) * proj_err / proj_err_norm

        else:
            raw_err = self._target - current
            proj_err = self._proj_space @ raw_err

            err = self._mag * proj_err + self._kI * self._total_err_I
            self._total_err_I = np.clip(self._total_err_I + proj_err, self._low_err_I, self._high_err_I)

            return err


class RotationAttractorController(BaseAttractorController):

    def compute_err(self, current, nullspace=np.eye(3)):
        ''' current is a 3x3 rotation matrix

        returns projected rotation error in angle axis form
        '''
        assert current.shape == (3, 3)

        current_rot_axis = nullspace @ current @ self._axis

        err_angax = angle_axis_between_axes(current_rot_axis, nullspace @ self._target)
        err_theta = np.linalg.norm(err_angax)

        if np.isclose(err_theta, 0):
            return np.zeros(3)

        return min(self._mag, err_theta) * err_angax / err_theta


class NullAttractorController(BaseAttractorController):

    def __init__(self, *args, **kwargs):
        super().__init__(np.zeros(3), np.zeros(3), 0, *args, **kwargs)

    def compute_err(self, current):
        return np.zeros_like(current)


class FrankaGripperController(NullAttractorController):
    '''Gripper controllers. NONE means no controll.'''

    def __init__(self, action_mode=None):
        assert action_mode in [None, 'close', 'open'], "Action mode not defined"
        super().__init__()
        self._mode = action_mode

    @property
    def mode(self):
        return self._mode
    

class ComposedAttractorControllers:

    def __init__(self, attractors_dict, rot_nullspace_mode='adaptive'):
        assert isinstance(attractors_dict, OrderedDict), 'attractors_dict must be an ordered dict!'
        assert rot_nullspace_mode in ('adaptive', 'fixed', 'angax')
        self._attractors_dict = attractors_dict
        self._attractors = [attractor for attractor in self._attractors_dict.values()]
        self._rot_nullspace_mode = rot_nullspace_mode
        self._had_force_ctrlr = False

    @property
    def low_level_controllers(self):
        return self._attractors

    @property
    def low_level_controllers_dict(self):
        return self._attractors_dict

    def update_low_level_controller(self, key, new_axis, new_target):
        assert key in self._attractors_dict.keys(), "controller {} doesn't exist".format(key)
        assert (not isinstance(self._attractors_dict[key], FrankaGripperController)), "Gripper controllers can not be updated"
        if new_axis is not None:
            self._attractors_dict[key].axis = new_axis
        if new_target is not None:
            self._attractors_dict[key].target = new_target
    
    def reset(self):
        for attractor in self._attractors:
            if isinstance(attractor, ForceAttractorController):
                attractor.reset_total_err()

    def compute_err(self, current_tra, current_R, current_force, attractor_idxs):
        I = np.eye(3)
        err_tra, err_R = np.zeros(3), I
        n_tra_attractors, n_rot_attractors = 0, 0
        tra_axes, rot_axes = np.zeros((3, 3)), np.zeros((1, 3))

        has_force_ctrlr = False
        for attractor_idx in attractor_idxs:
            attractor = self._attractors[attractor_idx]

            if isinstance(attractor, PositionAttractorController):
                if n_tra_attractors < 3:
                    err = attractor.compute_err(current_tra)
                    err_tra += (I - np.linalg.pinv(tra_axes) @ tra_axes) @ err

                    tra_axes[n_tra_attractors] = attractor.axis
                    n_tra_attractors += 1

            elif isinstance(attractor, ForceAttractorController):
                if n_tra_attractors < 3:
                    if not self._had_force_ctrlr:
                        attractor.reset_total_err()
                    has_force_ctrlr = True

                    err = attractor.compute_err(current_force)
                    err_tra += (I - np.linalg.pinv(tra_axes) @ tra_axes) @ err

                    tra_axes[n_tra_attractors] = attractor.axis
                    n_tra_attractors += 1

            elif isinstance(attractor, RotationAttractorController):
                if n_rot_attractors == 0:
                    err = attractor.compute_err(current_R)
                    err_R = angle_axis_to_rotation(err)

                    if self._rot_nullspace_mode == 'fixed':
                        current_R = err_R @ current_R
                        rot_axes[0] = attractor.axis
                    elif self._rot_nullspace_mode == 'adaptive':
                        rot_axes[0] = current_R @ attractor.axis
                    elif self._rot_nullspace_mode == 'angax':
                        rot_axes[0] = np.zeros(3) if np.isclose(np.linalg.norm(err), 0) else (err / np.linalg.norm(err))
                    n_rot_attractors += 1

                elif n_rot_attractors == 1 and self._rot_nullspace_mode in ('fixed', 'adaptive'):
                    nullspace = (I - np.linalg.pinv(rot_axes) @ rot_axes)
                    err = attractor.compute_err(current_R, nullspace)
                    err_R = angle_axis_to_rotation(err) @ err_R

                    n_rot_attractors += 1

                elif n_rot_attractors == 1 and self._rot_nullspace_mode == 'angax':
                    nullspace = (I - np.linalg.pinv(rot_axes) @ rot_axes)
                    
                    err = attractor.compute_err(current_R)
                    err = nullspace @ err

                    err_R = angle_axis_to_rotation(err) @ err_R

                    n_rot_attractors += 1

            elif isinstance(attractor, NullAttractorController):
                continue
        
        self._had_force_ctrlr = has_force_ctrlr

        err_rot = rotation_to_angle_axis(err_R)
        err_theta = np.linalg.norm(err_rot)
        if err_theta > np.pi:
            err_rot = (err_theta - 2 * np.pi) * err_rot / err_theta

        return err_tra, quaternion.from_rotation_vector(err_rot)

    def compute_controls(self, *args, **kwargs):
        err_tra, err_quat = self.compute_err(*args, **kwargs)

        delta_translation = gymapi.Transform(p=np_to_vec3(err_tra))
        delta_rotation = gymapi.Transform(r=np_quat_to_quat(err_quat))

        return delta_translation, delta_rotation

    def compute_gripper_controls(self, attractor_idxs):
        for attractor_idx in attractor_idxs:
            attractor = self._attractors[attractor_idx]
            if isinstance(attractor, FrankaGripperController):
                return attractor.mode
        return None


class ControllersListFeatureExtractor:

    def __init__(self, attractors_list, is_2d=False):
        self._attractors_list = attractors_list
        self._is_2d = is_2d

        self._attractor_target_types = {}
        i = 0
        for attractor in self._attractors_list:
            if attractor.target_type not in self._attractor_target_types:
                self._attractor_target_types[attractor.target_type] = i 
                i += 1

        self._features_buffer = np.zeros((
            len(self._attractors_list),
            4 + len(self._attractor_target_types) + len(self._attractor_target_types) * (17 if self._is_2d else 27)
        ))

        self._low, self._high = np.zeros([self._features_buffer.shape[1]]), np.zeros([self._features_buffer.shape[1]])
        self._low[:4 + len(self._attractor_target_types)] = 0
        self._high[:4 + len(self._attractor_target_types)] = 1
        self._low[4 + len(self._attractor_target_types):] = -10
        self._high[4 + len(self._attractor_target_types):] = 10

        for i, attractor in enumerate(self._attractors_list):
            # attractor type
            if isinstance(attractor, NullAttractorController):
                self._features_buffer[i, :4] = [1, 0, 0, 0]
            elif isinstance(attractor, ForceAttractorController):
                self._features_buffer[i, :4] = [0, 1, 0, 0]
            elif isinstance(attractor, PositionAttractorController):
                self._features_buffer[i, :4] = [0, 0, 1, 0]
            elif isinstance(attractor, RotationAttractorController):
                self._features_buffer[i, :4] = [0, 0, 0, 1]
            else:
                raise ValueError('Unsupported attractor type')

            # scene element target type
            self._features_buffer[i, 4 + self._attractor_target_types[attractor.target_type]] = 1

        self._force_idxs = [
            4 + len(self._attractor_target_types),
            4 + len(self._attractor_target_types) + (6 if self._is_2d else 9)
        ]
        self._pos_idxs = [
            self._force_idxs[-1],
            self._force_idxs[-1] + (6 if self._is_2d else 9)
        ]
        self._rot_idxs = [
            self._pos_idxs[-1],
            self._pos_idxs[-1] + (5 if self._is_2d else 9)
        ]

    @property
    def dim(self):
        return self._features_buffer.shape[1]

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def latest_features(self):
        return self._features_buffer

    def get_features(self, current_tra, current_R, current_force):
        self._features_buffer[:, self._force_idxs[0]:] = 0

        for i, attractor in enumerate(self._attractors_list):
            if isinstance(attractor, ForceAttractorController):
                err = attractor.compute_err(current_force)
                self._features_buffer[i, self._force_idxs[0]:self._force_idxs[1]] = \
                    np.concatenate([
                        attractor.axis[[0, 2]] if self._is_2d else attractor.axis,
                        attractor.target[[0, 2]] if self._is_2d else attractor.target,
                        err[[0, 2]] if self._is_2d else err
                    ])
            elif isinstance(attractor, PositionAttractorController):
                err = attractor.compute_err(current_tra)
                self._features_buffer[i, self._pos_idxs[0]:self._pos_idxs[1]] = \
                    np.concatenate([
                        attractor.axis[[0, 2]] if self._is_2d else attractor.axis,
                        attractor.target[[0, 2]] if self._is_2d else attractor.target,
                        err[[0, 2]] if self._is_2d else err
                    ])
            elif isinstance(attractor, RotationAttractorController):
                err = attractor.compute_err(current_R)
                self._features_buffer[i, self._rot_idxs[0]:self._rot_idxs[1]] = \
                    np.concatenate([
                        attractor.axis[[0, 2]] if self._is_2d else attractor.axis,
                        attractor.target[[0, 2]] if self._is_2d else attractor.target,
                        [np.sign(err[1]) * np.linalg.norm(err)] if self._is_2d else err
                    ])

        return self._features_buffer
