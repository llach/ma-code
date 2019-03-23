from forkan.rl import VecVAEStack, VAEStack
import numpy as np
from baselines.common.cmd_util import make_vec_env

from gym.envs.classic_control.pendulum_test import PendulumTestEnv
import matplotlib.pyplot as plt
import seaborn as sns

vae_name = 'pendvisualuniform-b77.5-lat5-lr0.001-2019-03-21T00/13'.replace('/', ':')


env = make_vec_env('PendulumTest-v0', 'classic_control', 2, 0, flatten_dict_observations=False)
venv = VecVAEStack(env, k=3, load_from=vae_name)

thetas = np.linspace(0, 2*np.pi, 20)

dings1 = []
dings2 = []

o = venv.reset()
d = False

while not np.any(d):
    print(o)
    exit(0)
    # when using env obs, we can plot to sanity check theta
    # plt.imshow(np.squeeze(o), cmap='Greys_r', label='theta {}'.format(th))
    # plt.title('{}'.format(th))
    # plt.show()

    dings1.append(o[0])
    dings2.append(o[1])

    o, _, d, _ = venv.step([0, 0])

# print(dings1.shape)
dings1 = np.moveaxis(np.asarray(dings1, dtype=np.float), 0, -1)
dings2 = np.moveaxis(np.asarray(dings2, dtype=np.float), 0, -1)
print(dings1.shape)
# dings = np.asarray(dings, dtype=np.float).reshape(6, 20)  ### WRONG!!! old bug, just for visualising
sns.set()

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle("VAE Stack observation space", fontsize=16)

axes[0].plot(thetas, dings1[0], label='mu 0')
axes[0].plot(thetas, dings1[1], label='mu 1')
axes[0].plot(thetas, dings1[2], label='mu 2')
axes[0].plot(thetas, dings1[3], label='mu 3')
axes[0].plot(thetas, dings1[4], label='mu 4')
axes[0].legend()

axes[1].plot(thetas, dings2[5], label='mu 0')
axes[1].plot(thetas, dings2[1+5], label='mu 1')
axes[1].plot(thetas, dings2[2+5], label='mu 2')
axes[1].plot(thetas, dings2[3+5], label='mu 3')
axes[1].plot(thetas, dings2[4+5], label='mu 4')
axes[1].legend()


fig.tight_layout()
fig.subplots_adjust(top=0.88)

plt.show()
