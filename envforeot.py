import gym
from gym import spaces
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

class rmm(gym.Env):
    def __init__(self):
        super(rmm, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-2, high=2, shape=(1,), dtype=np.float32)
        self.gt_center = np.zeros((2, 112))
        self.gt_center[:, 0] = [0, 0]
        self.gt_orient = np.concatenate((
            np.tile(np.pi / 3, 112),
        ))
        self.gt_vel = np.vstack([
            (500 / 36) * np.cos(self.gt_orient),
            (500 / 36) * np.sin(self.gt_orient)
        ])
        self.gt_length = np.tile(np.array([[340 / 2], [80 / 2]]), (1, self.gt_vel.shape[1]))

        self.time_steps = self.gt_vel.shape[1]
        self.time_interval = 10

        self.gt_rotation = np.zeros((2, 2, self.time_steps))

        for t in range(self.time_steps):
            self.gt_rotation[:, :, t] = np.array([[np.cos(self.gt_orient[t]), -np.sin(self.gt_orient[t])],
                                                  [np.sin(self.gt_orient[t]), np.cos(self.gt_orient[t])]])
            if t > 0:
                self.gt_center[:, t] = self.gt_center[:, t - 1] + self.gt_vel[:, t] * self.time_interval

        self.gt = np.vstack([self.gt_center, self.gt_orient, self.gt_length, self.gt_vel])

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype='float32')

        self.Ar = np.array([[1, 0, 10, 0],
                            [0, 1, 0, 10],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype='float32')

        self.Ap = np.eye(3)

        self.Ch = np.array([[1 / 4, 0],
                            [0, 1 / 4]], dtype='float32')
        self.Cv = np.array([[200, 0],
                            [0, 8]], dtype='float32')
        self.Cwr = np.array([[100, 0, 0, 0],
                             [0, 100, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype='float32')
        self.Cwp = np.array([[0.05, 0, 0],
                             [0, 0.001, 0],
                             [0, 0, 0.001]], dtype='float32')

        self.lambda_val = 10
        self.r = np.array([[100],
                           [100],
                           [10],
                           [-17]], dtype='float32')
        self.p = np.array([[np.pi / 3],
                           [200],
                           [90]], dtype='float32')

        self.Cr = np.array([[900, 0, 0, 0],
                            [0, 900, 0, 0],
                            [0, 0, 16, 0],
                            [0, 0, 0, 16]], dtype='float32').T
        self.Cp = np.array([[0.2, 0, 0],
                            [0, 400, 0],
                            [0, 0, 400]], dtype='float32')



