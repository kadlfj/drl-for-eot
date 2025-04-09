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



    def step(self, action, t):
        nk = np.random.poisson(self.lambda_val)
        while nk == 0:
            nk = np.random.poisson(self.lambda_val)

        y = np.zeros((2, nk))
        h = np.zeros((nk, 2))
        for n in range(nk):
            h[n, :] = -1 + 2 * np.random.rand(1, 2)
            while np.linalg.norm(h[n, :]) > 1:
                h[n, :] = -1 + 2 * np.random.rand(1, 2)
            y[:, n] = self.gt[:2, t] + h[0, 0] * self.gt[3, t] * np.array(
                [np.cos(self.gt[2, t]), np.sin(self.gt[2, t])]) + \
                      h[0, 1] * self.gt[4, t] * np.array([-np.sin(self.gt[2, t]), np.cos(self.gt[2, t])]) + \
                      np.random.multivariate_normal([0, 0], self.Cv).T
        r, p, Cr, Cp, theta = measurement_update(y, self.H, self.r, self.p, self.Cr, self.Cp, self.Ch, self.Cv, action)
        ES1 = np.log(1 + abs(np.linalg.det(self.Cp) - np.linalg.det(Cp)))
        ES2 = np.log(1 + abs(np.linalg.det(self.Cr) - np.linalg.det(Cr)))
        ES = ES1 + 0.1 * ES2
        reward = -ES
        r, p, Cr, Cp = time_update(r, p, Cr, Cp, self.Ar, self.Ap, self.Cwr, self.Cwp)
        self.r = r
        self.p = p
        self.Cr = Cr
        self.Cp = Cp
        return theta, reward, {}, {}, {}

    def step1(self, action, t, j):
        global r, p, Cr, Cp
        if t == 0:
            r = self.r
            p = self.p
            Cr = self.Cr
            Cp = self.Cp
        nk = np.random.poisson(self.lambda_val)
        while nk == 0:
            nk = np.random.poisson(self.lambda_val)

        y = np.zeros((2, nk))
        h = np.zeros((nk, 2))
        for n in range(nk):
            h[n, :] = -1 + 2 * np.random.rand(1, 2)
            while np.linalg.norm(h[n, :]) > 1:
                h[n, :] = -1 + 2 * np.random.rand(1, 2)

            y[:, n] = self.gt[:2, t] + h[0, 0] * self.gt[3, t] * np.array(
                [np.cos(self.gt[2, t]), np.sin(self.gt[2, t])]) + \
                      h[0, 1] * self.gt[4, t] * np.array([-np.sin(self.gt[2, t]), np.cos(self.gt[2, t])]) + \
                      np.random.multivariate_normal([0, 0], self.Cv).T  # 量测方程

        r, p, Cr, Cp, theta = measurement_update(y, self.H, r, p, Cr, Cp, self.Ch, self.Cv, action)
        if j == 0:
            if t % 3 == 0:
                gt_plot = plot_extent(self.gt[:, t], line_style='-', color='k', line_width=1)
                est_plot = plot_extent1(r[:2], p, theta, line_style='-', color='r', line_width=1)
                plt.pause(0.1)

        r, p, Cr, Cp = time_update(r, p, Cr, Cp, self.Ar, self.Ap, self.Cwr, self.Cwp)
        return theta

    def reset(self):
        self.state = np.array([[0],
                               [0],
                               [10],
                               [-17]], dtype='float32')
        theta = np.pi/3
        return theta

def measurement_update(y, H, r, p, Cr, Cp, Ch, Cv, action):
    nk = y.shape[1]
    theta, l1, l2 = choose_action(action)
    p = np.vstack([theta, l1, l2])
    for i in range(nk):
        CI, CII, M, F, Ftilde = get_auxiliary_variables(p, Cp, Ch)
        yi = y[:, i]
        yibar = np.dot(H, r)
        Cry = np.dot(Cr, H.T)
        Cy = np.dot(np.dot(H, Cr), H.T) + CI + CII + Cv
        r = r + np.dot(np.dot(Cry, np.linalg.inv(Cy)), (yi.reshape(-1, 1) - yibar))
        Cr = Cr - np.dot(np.dot(Cry, np.linalg.inv(Cy)), Cry.T)
        Cr = (Cr + Cr.T) / 2
        Yi = np.dot(F, np.kron(yi.reshape(-1, 1) - yibar, yi.reshape(-1, 1) - yibar))
        Yibar = np.dot(F, np.reshape(Cy, (4, 1)))
        CpY = np.dot(Cp, M.T)
        CY = np.dot(np.dot(F, np.kron(Cy, Cy)), (F + Ftilde).T)
        Cp = Cp - np.dot(np.dot(CpY, np.linalg.inv(CY)), CpY.T)
        Cp = (Cp + Cp.T) / 2

    return r, p, Cr, Cp, theta

def get_auxiliary_variables(p, Cp, Ch):
    alpha = p[0][0]
    l1 = p[1][0]
    l2 = p[2][0]

    S = np.dot(np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]]), np.diag([l1, l2]))
    S1 = S[0, :]
    S2 = S[1, :]

    J1 = np.array([[-l1 * np.sin(alpha), np.cos(alpha), 0],
                   [-l2 * np.cos(alpha), 0, -np.sin(alpha)]])
    J2 = np.array([[l1 * np.cos(alpha), np.sin(alpha), 0.0],
                   [-l2 * np.sin(alpha), 0, np.cos(alpha)]])

    CI = np.dot(S, np.dot(Ch, S.T))

    CII = np.zeros((2, 2))
    CII[0, 0] = np.trace(np.dot(np.dot(Cp, J1.T), np.dot(Ch, J1)))
    CII[0, 1] = np.trace(np.dot(np.dot(Cp, J2.T), np.dot(Ch, J1)))
    CII[1, 0] = np.trace(np.dot(np.dot(Cp, J1.T), np.dot(Ch, J2)))
    CII[1, 1] = np.trace(np.dot(np.dot(Cp, J2.T), np.dot(Ch, J2)))

    M = np.vstack([
        2 * np.dot(S1, np.dot(Ch, J1)),
        2 * np.dot(S2, np.dot(Ch, J2)),
        np.dot(S1, np.dot(Ch, J2)) + np.dot(S2, np.dot(Ch, J1))
    ])

    F = np.array([[1, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 1, 0, 0]])
    Ftilde = np.array([[1, 0, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])
    return CI, CII, M, F, Ftilde

def time_update(r, p, Cr, Cp, Ar, Ap, Cwr, Cwp):
    r = np.dot(Ar, r)
    Cr = np.dot(np.dot(Ar, Cr), Ar.T) + Cwr
    Cp = np.dot(np.dot(Ap, Cp), Ap.T) + Cwp
    return r, p, Cr, Cp

def choose_action(action):
    theta = (action[0] + 2) * np.pi / 12 + np.pi / 6
    l1 = (action[0]+2)/4*(175-165)+165
    l2 = (action[0]+2)/4*(45-35)+45
    return theta, l1, l2

def d_gaussian_wasserstein(ellipse1, r, p, theta):
    m1 = ellipse1[:2]
    alpha1 = ellipse1[2]
    eigen_val1 = ellipse1[3:5]
    eigen_vec1 = np.array([[np.cos(alpha1), -np.sin(alpha1)], [np.sin(alpha1), np.cos(alpha1)]])

    sigma1 = np.dot(np.dot(eigen_vec1, np.diag(eigen_val1 ** 2)), eigen_vec1.T)
    sigma1 = (sigma1 + sigma1.T) / 2

    m2 = r[:2]
    alpha2 = theta
    eigen_val2 = p[1:3][:, 0]
    eigen_vec2 = np.array([[np.cos(alpha2), -np.sin(alpha2)], [np.sin(alpha2), np.cos(alpha2)]])

    sigma2 = np.dot(np.dot(eigen_vec2, np.diag(eigen_val2 ** 2)), eigen_vec2.T)
    sigma2 = (sigma2 + sigma2.T) / 2
    error_sq = np.linalg.norm(m1.reshape(-1, 1) - m2) ** 2 + np.trace(
        sigma1 + sigma2 - 2 * sqrtm((np.dot(np.dot(sqrtm(sigma1), sigma2), sqrtm(sigma1)))))

    distance = np.sqrt(error_sq)
    return distance

def plot_extent(ellipse,line_style, color, line_width):
    center = ellipse[:2]
    theta = ellipse[2]
    l = ellipse[3:5]
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    alpha = np.arange(0, 2 * np.pi + np.pi / 100, np.pi / 100)
    xunit = l[0] * np.cos(alpha)
    yunit = l[1] * np.sin(alpha)

    rotated = np.dot(R, np.vstack([xunit, yunit]))
    xpoints = rotated[0, :] + center[0]
    ypoints = rotated[1, :] + center[1]

    handle_extent = plt.plot(xpoints, ypoints, linestyle=line_style, color=color, linewidth=line_width)
    return handle_extent
def plot_extent1(r, p, theta, line_style, color, line_width):
    center = r[:2]
    theta = theta
    l = p[1:3]
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    alpha = np.arange(0, 2 * np.pi + np.pi / 100, np.pi / 100)
    xunit = l[0] * np.cos(alpha)
    yunit = l[1] * np.sin(alpha)

    rotated = np.dot(R, np.vstack([xunit, yunit]))
    xpoints = rotated[0, :] + center[0]
    ypoints = rotated[1, :] + center[1]

    handle_extent = plt.plot(xpoints, ypoints, linestyle=line_style, color=color, linewidth=line_width)
    return handle_extent