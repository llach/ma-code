import logging
import os

import numpy as np
import scipy

from baselines.run import main
from forkan import figure_path

log = logging.getLogger(__name__)
home = os.environ['HOME']

args = [
    '--env', 'BreakoutNoFrameskip-v4',
    '--alg', 'ppo2',
    '--num_timesteps', '10e7',
    '--num_env', '1',
    '--log_interval', '1',
    '--seed', '0',
    '--load_path', f'{home}/breakout-ppo/',
    '--play', 'True',
]

frames = [0, 588]

t = 0
model, env = main(args, just_return=True)
obs = env.reset()
d = False

log.info('generating frames')
while not d:

    fr = env.render(mode='rgb_array')

    if t in frames:
        scipy.misc.imsave(f'{figure_path}/frame{t}.png', fr)

    print(t)
    t += 1
    actions, _, _, _ = model.step(obs)

    obs, _, done, _ = env.step(actions)
    d = done.any() if isinstance(done, np.ndarray) else done
