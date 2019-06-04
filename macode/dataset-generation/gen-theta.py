import numpy as np
from gym.envs.classic_control.pendulum_test import PendulumTestEnv
from forkan.rl import VAEStack
from forkan import dataset_path

env_id = 'PendulumTest-v0'
env_type = 'classic_control'
load_from = 'pendvisualuniform-b77.5-lat5-lr0.001-2019-03-21T00:13'

steps = 2000
env = PendulumTestEnv(steps=steps)
venv = VAEStack(env, load_from=load_from, k=1)

venv.reset()
ths = []
obs = []

while True:
    ths.append(env._get_theta())

    o, r, d, i = venv.step(0)
    obs.append(o)

    if d:
        break

ths = np.asarray(ths, dtype=np.float32)
obs = np.asarray(obs, dtype=np.float32)

np.savez_compressed('{}/thetas-b77.5.npz'.format(dataset_path), thetas=ths, encodings=obs)
print('done')
