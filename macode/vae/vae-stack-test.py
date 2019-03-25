from forkan.rl import VAEStack
import numpy as np

from gym.envs.classic_control.pendulum_test import PendulumTestEnv
import matplotlib.pyplot as plt
import seaborn as sns
#
vae_name = 'pendvisualuniform-b80.0-lat5-lr0.001-2019-03-21T00/20'.replace('/', ':')

env = PendulumTestEnv(steps=200)
venv = VAEStack(env, load_from=vae_name, k=3)


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

# dings = np.moveaxis(np.asarray(dings, dtype=np.float), 0, -1)
# dings = np.asarray(dings, dtype=np.float).reshape(6, 20)  ### WRONG!!! old bug, just for visualising
sns.set()
dings = np.asarray(dings)


fig, axes = plt.subplots(1, 1, figsize=(10, 8))
fig.suptitle("VAE Stack observation space", fontsize=16)

lats = 5
stack = 3

for jott in range(stack):
    for idx in range(lats):
        axes.plot(thetas, dings[..., idx+(jott*lats)], label='obs {}'.format(idx+(jott*lats)))
        # axes.scatter(thetas, dings[..., idx+(jott*lats)])
axes.legend()

fig.tight_layout()
fig.subplots_adjust(top=0.88)

plt.show()
