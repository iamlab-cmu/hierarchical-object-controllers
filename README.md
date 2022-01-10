# Learning to Compose Hierarchical Object-Centric Controllers for Robotic Manipulation


This code is associated with our [CoRL'20 paper]().  
**[[Paper]](https://arxiv.org/abs/2011.04627)**&nbsp;**[[Project Website]](https://sites.google.com/view/compositional-object-control/home)**


### Dependencies

Since our code was developed on a previous beta-version of [Nvidia Isaac Gym](https://developer.nvidia.com/isaac-gym), it may not run with the existing open source version. However, it maybe be possible to port over the environments to the new version. 
Our code further relies on an earlier version of [isaacgym-utils](https://github.com/iamlab-cmu/isaacgym-utils), although for it most of the API changes should be consistent.


### Training code

The training code is provided in the `scripts` folder, it contains examples to run the door env as well as the hex screw environment. Their corresponding config files can be found in the `cfg` folder. Most params in the config files should be correct but some may neeed a bit of tweaking. 

An example command to run the code would be 
```
python ./scripts/train_franka_door.py --seed 101 --cfg ./cfg/run_franka_door_open.yaml --logdir /tmp/franka_door --train 1
```

### Code

- Environments: All the environments used in the paper are defined in `object_axes_ctrls/envs`. 
- Task-Axes Controllers: The controllers and their composition is defined in `object_axes_ctrlrs/controllers/projected_axes_attractors.py`. Each controller is parameterized with respect to the environment and created in the coresponding environment class. 
- Expanded-MDP: The expanded MDP formulation is implemented in `envs/env_wrappers.py`.


### Assets 

The assets folder contains the URDF assets (and meshes etc.) for creating and rendering environments.

### Citation

In case you find our paper or code useful please consider citing:
```
@inproceedings{sharma2020learning,
  title={Learning to Compose Hierarchical Object-Centric Controllers for Robotic Manipulation},
  author={Sharma, Mohit and Liang, Jacky and Zhao, Jialiang and LaGrassa, Alex and Kroemer, Oliver},
  booktitle={arXiv preprint arXiv:2011.04627},
  year={2020}
}
```
