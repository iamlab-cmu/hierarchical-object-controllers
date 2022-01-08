import numpy as np

from isaac_gym import gymapi
from isaac_gym_utils.ctrl_utils import ForcePositionController


class BoxWorldHighLevelController:

    def __init__(self, low_level_controllers, target_update_freq): 
        self.low_level_controllers = low_level_controllers
        self.target_update_freq = target_update_freq
    
    def reset(self):
        pass

    def _compute_low_level_controller_targets(self, controller_idxs):

        if len(controller_idxs) == 1:
            c_idx = controller_idxs[0]
            controller_0 = self.low_level_controllers[c_idx]
            target = controller_0.get_target()
            return target[:7]

        c0_idx, c1_idx, c2_idx = controller_idxs
        final_targets = np.zeros(10)
        args = ()

        controller_0 = self.low_level_controllers[c0_idx]
        c0_target = controller_0.get_target(*args)
        has_orientation_controller = False

        if controller_0.is_position_controller:
            assert np.sum(np.abs(c0_target[4:])) < 1e-6, "Invalid controller output"
        else:
            assert np.sum(np.abs(c0_target[:3])) < 1e-6, "Invalid controller output"
            has_orientation_controller = True
        final_targets += c0_target
        
        controller_1 = self.low_level_controllers[c1_idx]
        if controller_1.is_position_controller:
            c1_target = controller_1.get_target(*args)
            assert np.sum(np.abs(c1_target[4:])) < 1e-6, "Invalid controller output"
            c0_null = controller_0.null_space
            c1_target[:3] = c0_null @ c1_target[:3]
            final_targets += c1_target
        else:
            if not has_orientation_controller:
                c1_target = controller_1.get_target(*args)
                assert np.sum(np.abs(c1_target[:3])) < 1e-6, "Invalid controller output"
                final_targets += c1_target
                has_orientation_controller = True

        controller_2 = self.low_level_controllers[c2_idx]
        if controller_2.is_position_controller:
            c1_null = controller_1.null_space
            c1_null = c0_null @ c1_null
            c2_target = controller_2.get_target(*args)
            assert np.sum(np.abs(c2_target[4:])) < 1e-6, "Invalid controller output"
            c2_target[:3] = c1_null @ c2_target[:3]
            final_targets += c2_target
        else:
            if not has_orientation_controller:
                c2_target = controller_2.get_target(*args)
                assert np.sum(np.abs(c2_target[:3])) < 1e-6, "Invalid controller output"
                final_targets += c2_target
                has_orientation_controller = True

        return final_targets[:7]        

    def compute_controls(self, current_tra, current_R, block_force, controller_idxs):
        '''Compute delta transform controls
        
        controller_idxs: List of (max len 3) controllers to choose from
        '''
        delta_targets = self._compute_low_level_controller_targets(controller_idxs)
        pos, quat = delta_targets[:3], delta_targets[3:7]

        delta_translation = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]))
        # Quat(x, y, z, w) while numpy-quaternion represents it as (w, x, y, z)
        delta_rotation = gymapi.Transform(r=gymapi.Quat(quat[3], quat[0], quat[1], quat[2]))
            
        return delta_translation, delta_rotation
