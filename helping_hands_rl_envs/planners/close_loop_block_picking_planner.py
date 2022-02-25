import numpy as np

from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.planners.close_loop_planner import CloseLoopPlanner
from helping_hands_rl_envs.simulators.pybullet.utils import transformations

class CloseLoopBlockPickingPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)

  def getNextAction(self):
    if not self.env._isHolding():
      block_pos = self.env.objects[0].getPosition()
      block_rot = list(transformations.euler_from_quaternion(self.env.objects[0].getRotation()))
      # block_rot[2] -= np.pi/2

      x, y, z, r = self.getActionByGoalPose(block_pos, block_rot)

      if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
        primitive = constants.PICK_PRIMATIVE
      else:
        primitive = constants.PLACE_PRIMATIVE

    else:
      x, y, z = 0, 0, self.dpos
      r = 0
      primitive = constants.PICK_PRIMATIVE
    return self.env._encodeAction(primitive, x, y, z, r)

  def getStepsLeft(self):
    return 100