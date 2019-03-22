from forkan.rl import VAEStack
import numpy as np

from gym.envs.classic_control.pendulum_test import  PendulumTestEnv
import matplotlib.pyplot as plt
import seaborn as sns
#
# vae_name = 'pendvisualuniform-b22-lat5-lr0.001-2019-03-18T20/23'.replace('/', ':')

env = PendulumTestEnv()
venv = VAEStack(env, k=3)


thetas = []

dings = []

o = venv.reset()
d = False

while not d:
    th = env._get_theta()

    # when using env obs, we can plot to sanity check theta
    # plt.imshow(np.squeeze(o), cmap='Greys_r', label='theta {}'.format(th))
    # plt.title('{}'.format(th))
    # plt.show()

    thetas.append(th)
    dings.append(o)

    o, _, d, _ = venv.step(0)

dings = np.moveaxis(np.asarray(dings, dtype=np.float), 0, -1)
# dings = np.asarray(dings, dtype=np.float).reshape(6, 20)  ### WRONG!!! old bug, just for visualising
sns.set()

plt.plot(thetas, dings[0], label='mus')
plt.plot(thetas, np.sin(thetas), label='sin(th)')
plt.plot(thetas, np.cos(thetas), label='cos(th)')
plt.legend()
plt.show()

plt.plot(thetas, dings[1], label='mus')
plt.plot(thetas, np.sin(thetas), label='sin(th)')
plt.plot(thetas, np.cos(thetas), label='cos(th)')
plt.legend()
plt.show()
