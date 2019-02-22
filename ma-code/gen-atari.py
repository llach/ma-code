import logging
import argparse
import scipy.misc

import numpy as np

from forkan import dataset_path
from forkan.common.utils import create_dir
from baselines.common.cmd_util import make_env

log = logging.getLogger(__name__)

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--env', '-e', default='Pong')
args, unknown = parser.parse_known_args()


ENV = args.env + 'NoFrameskip-v4'
TOTAL_FRAMES = int(50e3)

log.info('running random agent on {} and storing {} frames'.format(ENV, TOTAL_FRAMES))

e = make_env(ENV, 'atari')

frames = np.zeros((TOTAL_FRAMES,)+e.observation_space.shape)

obs = e.reset()

log.info('generating frames')
for step in range(TOTAL_FRAMES):

    # normalize
    obs = obs / 255

    frames[step, ...] = obs

    a = e.action_space.sample()
    obs, reward, ds, info = e.step(a)

    if ds:
        obs = e.reset()

log.info('dumping file')
name = ENV.replace('NoFrameskip', '').lower().split('-')[0]
np.savez_compressed('{}/{}-normalized.npz'.format(dataset_path, name), data=frames)

log.info('storing some pngs')
create_dir('{}/{}/'.format(dataset_path, name))
for n, f in enumerate(frames[40:60, ...]):
    scipy.misc.imsave('{}/{}/frame{}.png'.format(dataset_path, name, n), np.squeeze(f))

log.info('done')
