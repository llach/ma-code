import logging
import os

import numpy as np
import scipy.misc
from tqdm import tqdm

from baselines.run import main
from forkan import dataset_path
from forkan.common.utils import create_dir

log = logging.getLogger(__name__)
TOTAL_FRAMES = 5e5
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

frames = np.zeros((int(TOTAL_FRAMES), 84, 84))

model, env = main(args, just_return=True)
obs = env.reset()

log.info('generating frames')
for step in tqdm(range(int(TOTAL_FRAMES))):
    actions, _, _, _ = model.step(obs)

    img = np.asarray(np.squeeze(obs[..., -1]) / 255, dtype=np.float32)


    frames[step, ...] = img

    obs, _, done, _ = env.step(actions)
    done = done.any() if isinstance(done, np.ndarray) else done

    if done:
        obs = env.reset()

log.info('dumping file')
name = args[1].replace('NoFrameskip', '').lower().split('-')[0]
np.savez_compressed('{}/{}-normalized.npz'.format(dataset_path, name), data=frames)

log.info('storing some example pngs for {}'.format(name))
create_dir('{}/{}/'.format(dataset_path, name))
for n, f in enumerate(frames[40:60, ...]):
    scipy.misc.imsave('{}/{}/frame{}.png'.format(dataset_path, name, n), np.squeeze(f))

log.info('done')
