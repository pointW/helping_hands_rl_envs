import numpy as np

DEFAULT_CONFIG = {
  'robot' : 'ur5',
  'pos_candidate' : None,
  'perfect_grasp' : False,
  'perfect_place' : False,
  'workspace_check' : 'box',
  'in_hand_size' : 24,
  'in_hand_mode' : 'sub',
  'num_random_objects' : 0,
  'random_orientation' : False,
  'check_random_obj_valid' : False,
  'action_sequence' : 'pxyr',
  'simulate_grasp' : True,
  'workspace' : np.array([[0.30, 0.60], [-0.15, 0.15], [0, 1]]),
  'object_scale_range': (0.60, 0.70),
  'max_steps' : 10,
  'obs_size' : 128,
  'fast_mode' : True,
  'render' : False,
  'physics_mode' : 'fast',
  'reward_type' : 'sparse',
  'num_objects' : 1,
  'object_type' : 'cube',
  'hard_reset_freq': 1,
  'view_type': 'render', # render or camera
}
