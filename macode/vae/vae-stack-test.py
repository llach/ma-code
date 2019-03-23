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

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle("VAE Stack observation space", fontsize=16)

axes[0].plot(thetas, dings[0], label='mu 0')
axes[0].plot(thetas, dings[2], label='mu 2')
axes[0].plot(thetas, dings[4], label='mu 4')
axes[0].legend()

axes[1].plot(thetas, dings[1], label='mu 1')
axes[1].plot(thetas, dings[3], label='mu 3')
axes[1].plot(thetas, dings[5], label='mu 5')
axes[1].legend()

fig.tight_layout()
fig.subplots_adjust(top=0.88)

plt.show()
