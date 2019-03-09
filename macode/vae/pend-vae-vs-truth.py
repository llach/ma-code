import time
import numpy as np

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

for _ in range(500):
    t += 1
    o, r, d, i = env.step(env.action_space.sample())

    _, _, zs = v.encode(o)
    th =  env.env.env.env.state[0]
    z = zs[0][idx]
    print('{} env th {}'.format(t, th))
    print('{} vae th {}'.format(t, z))
    zss.append(z)
    ths.append(th)
    if np.any(d):
        env.reset()

ths = np.asarray(ths, dtype=np.float)
zss = np.asarray(zss, dtype=np.float)

print('th min {} max {}'.format(np.min(ths), np.max(ths)))
print('z  min {} max {}'.format(np.min(zss), np.max(zss)))

env.close()