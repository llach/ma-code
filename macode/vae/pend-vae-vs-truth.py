import time
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from scipy import stats

from forkan.models import VAE

from baselines.common.cmd_util import make_env

env_id = 'Pendulum-v0'
env_type = 'classic_control'

env = make_env(env_id, env_type, vae_pend=True)

env.reset()

v = VAE(load_from='pend-optimal', network='pendulum')

t = 0
idx = 2

ths = []
zss = []

thds = []
zdots = []

old_z = 0
dt = 0.05

for _ in range(500):
    t += 1
    o, r, d, i = env.step(env.action_space.sample())

    th, thd = env.env.env.env.env.state
    th = th
    z = o[0][idx]
    zdot = np.clip((z - old_z) / 0.05, -8, 8)
    old_z = z

    zss.append(z)
    ths.append(th)

    zdots.append(zdot)
    thds.append(thd)
    if np.any(d):
        old_z = 0
        env.reset()

ths = np.asarray(ths, dtype=np.float)
zss = np.asarray(zss, dtype=np.float)

print('th [{}, {}] thd [{}, {}]'.format(np.min(ths), np.max(ths), np.min(thds), np.max(thds)))
print('z  [{}, {}] zdots [{}, {}]'.format(np.min(zss), np.max(zss), np.min(zdots), np.max(zdots)))

t_thsin, p_thsin = stats.ttest_ind(np.sin(ths), np.sin(zss))
print('sin(th) & sin(z) t: {}; p: {}'.format(t_thsin, p_thsin))

t_thdot, p_thdot = stats.ttest_ind(thds, zdots)
print('theta dot & z dot t: {}; p: {}'.format(t_thdot, p_thdot))

sns.distplot(zdots, label='zdot')
sns.distplot(thds, label='thdot')
plt.legend()
plt.show()

sns.distplot(ths, label='ths')
sns.distplot(zss, label='zs')
plt.legend()
plt.show()

sns.distplot(np.sin(ths), label='sin(ths)')
sns.distplot(np.sin(zss), label='sin(zs)')
plt.legend()
plt.show()

env.close()