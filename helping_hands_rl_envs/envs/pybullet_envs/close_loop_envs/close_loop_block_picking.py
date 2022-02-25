import pybullet as pb
import numpy as np

from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.envs.pybullet_envs.close_loop_envs.close_loop_env import CloseLoopEnv
from helping_hands_rl_envs.simulators.pybullet.utils import transformations
from helping_hands_rl_envs.planners.close_loop_block_picking_planner import CloseLoopBlockPickingPlanner

class CloseLoopBlockPickingEnv(CloseLoopEnv):
  def __init__(self, config):
    super().__init__(config)

  def reset(self):
    self.resetPybulletEnv()
    # self.robot.moveTo([self.workspace[0].mean(), self.workspace[1][0], 0.2], transformations.quaternion_from_euler(0, 0, 0))
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2],
                      transformations.quaternion_from_euler(0, 0, -np.pi / 2), dynamic=False)
    cube_pos = [0.4, -0.1, 0.025]
    cube_rot = transformations.quaternion_from_euler(0, 0, 0)
    self._generateShapes(constants.BRICK, 1, pos=[cube_pos], rot=[cube_rot])

    # for id in range(-1, 9):
    #   pb.changeVisualShape(self.robot.id, id, rgbaColor=[0, 0, 0, 0])
    # pb.changeVisualShape(self.robot.id, 11, rgbaColor=[0, 0, 0, 0])

    # step = 100
    # d_dis = 0.7-1.1
    # d_pitch = -89.999 - -40
    # d_x = 0.5 - 0.
    # import time
    # for i in range(step+1):
    #   pb.resetDebugVisualizerCamera(
    #     cameraDistance=1.1 + i*d_dis/step,
    #     cameraYaw=90,
    #     cameraPitch=-40 + i*d_pitch/step,
    #     cameraTargetPosition=[0+i*d_x/step, 0, 0])
    #   time.sleep(0.01)
    #
    import time
    time.sleep(0.5)

    total = 5000
    for i in range(total):
      d = 0.2/total * (i+1)
      ws_center = np.array([self.workspace[0].mean(), self.workspace[1].mean()])
      robot_xy = np.array([self.workspace[0].mean(), self.workspace[1].mean()+d])
      self.robot.moveTo([robot_xy[0], robot_xy[1], 0.2], transformations.quaternion_from_euler(0, 0, -np.pi / 2),
                        dynamic=False)

      cube_xy = np.array([cube_pos[0], cube_pos[1]+d])
      self.objects[0].resetPose([cube_xy[0], cube_xy[1], 0.025], transformations.quaternion_from_euler(0, 0, 0))
      pb.stepSimulation()

    for i in range(total):
      d = 0.2 - 0.2/total * (i+1)
      ws_center = np.array([self.workspace[0].mean(), self.workspace[1].mean()])
      robot_xy = np.array([self.workspace[0].mean(), self.workspace[1].mean()+d])
      self.robot.moveTo([robot_xy[0], robot_xy[1], 0.2], transformations.quaternion_from_euler(0, 0, -np.pi / 2),
                        dynamic=False)

      cube_xy = np.array([cube_pos[0], cube_pos[1]+d])
      self.objects[0].resetPose([cube_xy[0], cube_xy[1], 0.025], transformations.quaternion_from_euler(0, 0, 0))
      pb.stepSimulation()

    for i in range(total):
      r = 2*np.pi/4/total*(i+1)
      ws_center = np.array([self.workspace[0].mean(), self.workspace[1].mean()])
      robot_xy = np.array([self.workspace[0].mean(), self.workspace[1].mean()]) - ws_center
      robot_xy = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]]).dot(robot_xy) + ws_center
      self.robot.moveTo([robot_xy[0], robot_xy[1], 0.2], transformations.quaternion_from_euler(0, 0, -np.pi/2+r), dynamic=False)

      cube_xy = np.array(cube_pos[:2]) - ws_center
      cube_xy = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]]).dot(cube_xy) + ws_center
      self.objects[0].resetPose([cube_xy[0], cube_xy[1], 0.025], transformations.quaternion_from_euler(0, 0, r))
      pb.stepSimulation()

    for i in range(total):
      r = np.pi/2 - 2*np.pi/4/total*(i+1)
      ws_center = np.array([self.workspace[0].mean(), self.workspace[1].mean()])
      robot_xy = np.array([self.workspace[0].mean(), self.workspace[1].mean()]) - ws_center
      robot_xy = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]]).dot(robot_xy) + ws_center
      self.robot.moveTo([robot_xy[0], robot_xy[1], 0.2], transformations.quaternion_from_euler(0, 0, -np.pi/2+r), dynamic=False)

      cube_xy = np.array(cube_pos[:2]) - ws_center
      cube_xy = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]]).dot(cube_xy) + ws_center
      self.objects[0].resetPose([cube_xy[0], cube_xy[1], 0.025], transformations.quaternion_from_euler(0, 0, r))
      pb.stepSimulation()

    return self._getObservation()

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi/2 * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def _checkTermination(self):
    gripper_z = self.robot._getEndEffectorPosition()[-1]
    return self.robot.holding_obj == self.objects[-1] and gripper_z > 0.08

def createCloseLoopBlockPickingEnv(config):
  return CloseLoopBlockPickingEnv(config)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  workspace = np.asarray([[0.2, 0.8],
                          [-0.3, 0.3],
                          [0.01, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 2, 'action_sequence': 'pxyzr', 'num_objects': 1, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'object_init_space_check': 'point', 'physics_mode': 'fast', 'object_scale_range': (1, 1), 'hard_reset_freq': 1000}
  planner_config = {'random_orientation': False}
  env_config['seed'] = 1
  env = CloseLoopBlockPickingEnv(env_config)
  planner = CloseLoopBlockPickingPlanner(env, planner_config)
  s, in_hand, obs = env.reset()
  # while True:
  #   current_pos = env.robot._getEndEffectorPosition()
  #   current_rot = transformations.euler_from_quaternion(env.robot._getEndEffectorRotation())
  #
  #   block_pos = env.objects[0].getPosition()
  #   block_rot = transformations.euler_from_quaternion(env.objects[0].getRotation())
  #
  #   pos_diff = block_pos - current_pos
  #   rot_diff = np.array(block_rot) - current_rot
  #   pos_diff[pos_diff // 0.01 > 1] = 0.01
  #   pos_diff[pos_diff // -0.01 > 1] = -0.01
  #
  #   rot_diff[rot_diff // (np.pi/32) > 1] = np.pi/32
  #   rot_diff[rot_diff // (-np.pi/32) > 1] = -np.pi/32
  #
  #   action = [1, pos_diff[0], pos_diff[1], pos_diff[2], rot_diff[2]]
  #   obs, reward, done = env.step(action)

  while True:
    action = planner.getNextAction()
    obs, reward, done = env.step(action)

  # fig, axs = plt.subplots(8, 5, figsize=(25, 40))
  # for i in range(40):
  #   action = planner.getNextAction()
  #   obs, reward, done = env.step(action)
  #   axs[i//5, i%5].imshow(obs[2][0], vmax=0.3)
  # env.reset()
  # fig.show()