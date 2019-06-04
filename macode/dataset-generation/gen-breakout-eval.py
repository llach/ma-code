import logging

import numpy as np
import scipy

from forkan import dataset_path
from forkan.common.utils import create_dir
from forkan.datasets import load_atari_normalized

logger = logging.getLogger(__name__)

logger.info('loading dataset ...')
data = load_atari_normalized('breakout-small')
logger.info('done loading')


np.random.seed(0)
idxs = [5, 6, 7, 305711, 244444]
rand_frames = data[idxs]

print('dumping file')
np.savez_compressed('{}/breakout-eval.npz'.format(dataset_path), data=rand_frames)

print('storing some pngs')

create_dir('{}/breakout-eval/'.format(dataset_path))

for n, f in enumerate(rand_frames[:, ...]):
    scipy.misc.imsave('{}/breakout-eval/frame{}.png'.format(dataset_path, n), np.squeeze(f))
print('done')