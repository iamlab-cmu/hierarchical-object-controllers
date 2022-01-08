from isaac_gym_utils.rl.stable_baselines import StableBaselinesVecEnvAdapter
from object_axes_ctrlrs.exp.exp_utils import create_env_with_klass

from object_axes_ctrlrs.envs.block2d import GymBlock2DFittingVecEnv
from object_axes_ctrlrs.envs.block2d import GymBlock2DPushVecEnv
from object_axes_ctrlrs.envs.franka_door import GymFrankaDoorOpenVecEnv
from object_axes_ctrlrs.envs.franka_hex_screw import GymFrankaHexScrewVecEnv

# block 2d fitting
class GymBlock2DFittingVecEnvStableBaselines(GymBlock2DFittingVecEnv, StableBaselinesVecEnvAdapter):

    def __init__(self, *args, **kwargs):
        GymBlock2DFittingVecEnv.__init__(self, *args, **kwargs)


def create_block2d_fit_env(args, cfg, inter_step_cb, separate_policies=False, all_models=None):
    return create_env_with_klass(
        GymBlock2DFittingVecEnvStableBaselines,
        args, cfg, inter_step_cb, separate_policies=separate_policies, all_models=all_models)

# block 2d pushing
class GymBlock2DPushVecEnvStableBaselines(GymBlock2DPushVecEnv, StableBaselinesVecEnvAdapter):

    def __init__(self, *args, **kwargs):
        GymBlock2DPushVecEnv.__init__(self, *args, **kwargs)


def create_block2d_push_env(args, cfg, inter_step_cb, separate_policies=False, all_models=None):
    return create_env_with_klass(
        GymBlock2DPushVecEnvStableBaselines, args, cfg, inter_step_cb, separate_policies=separate_policies,
        all_models=all_models)


# franka door opening
class GymFrankaDoorOpenVecEnvStableBaselines(GymFrankaDoorOpenVecEnv, StableBaselinesVecEnvAdapter):

    def __init__(self, *args, **kwargs):
        GymFrankaDoorOpenVecEnv.__init__(self, *args, **kwargs)


def create_franka_door_open_env(args, cfg, inter_step_cb):
    return create_env_with_klass(
        GymFrankaDoorOpenVecEnvStableBaselines, args, cfg, inter_step_cb)

# fanka hex screw
class GymFrankaHexScrewVecEnvStableBaselines(GymFrankaHexScrewVecEnv, StableBaselinesVecEnvAdapter):

    def __init__(self, *args, **kwargs):
        GymFrankaHexScrewVecEnv.__init__(self, *args, **kwargs)


def create_franka_hex_screw_env(args, cfg, inter_step_cb):
    return create_env_with_klass(
        GymFrankaHexScrewVecEnvStableBaselines, args, cfg, inter_step_cb)