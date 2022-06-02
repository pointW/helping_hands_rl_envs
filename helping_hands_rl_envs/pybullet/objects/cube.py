import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import helping_hands_rl_envs
from helping_hands_rl_envs.pybullet.objects.pybullet_object import PybulletObject
from helping_hands_rl_envs.pybullet.utils import constants
from helping_hands_rl_envs.pybullet.utils import transformations

class Cube(PybulletObject):
  def __init__(self, pos, rot, scale):
    root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    urdf_filepath = os.path.join(root_dir, constants.OBJECTS_PATH, 'cube.urdf')
    object_id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)
    pb.changeDynamics(object_id,
                      -1,
                      lateralFriction=1.0,
                      rollingFriction=0,
                      linearDamping=0.)
    super(Cube, self).__init__(constants.CUBE, object_id)

    self.original_size = 0.05
    self.size = 0.05 * scale

  def getHeight(self):
    return self.size

  def getRotation(self):
    pos, rot = self.getPose()
    return rot

  def getPose(self):
    pos, rot = pb.getBasePositionAndOrientation(self.object_id)
    T = transformations.quaternion_matrix(rot)
    t = 0
    while T[2, 2] < 0.5 and t < 4:
      T = T.dot(transformations.euler_matrix(np.pi/2, 0, 0))
      t += 1

    t = 0
    while T[2, 2] < 0.5 and t < 4:
      T = T.dot(transformations.euler_matrix(0, np.pi/2, 0))
      t += 1

    rot = transformations.quaternion_from_matrix(T)
    return list(pos), list(rot)
