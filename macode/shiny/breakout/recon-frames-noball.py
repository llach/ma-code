import logging

import numpy as np
import scipy

from forkan import figure_path
from forkan.datasets import load_set
from forkan.models import VAE

logger = logging.getLogger(__name__)

logger.info('loading dataset ...')
data = load_set('breakout-eval')
logger.info('done loading')

mdir = '/Users/llach/.forkan/done/breakout/vae-atari/'

runs = [
    'breakout-b1.28-lat20-lr0.0001-2019-04-20T21/24'.replace('/', ':'),
]

v = VAE(load_from=f'{runs[0]}', network='atari')

rec_frames = np.squeeze(v.reconstruct(data))

i = 1
scipy.misc.imsave(f'{figure_path}/pro1.png', data[i,...])
scipy.misc.imsave(f'{figure_path}/rec1.png', rec_frames[i,...])

i = -2
scipy.misc.imsave(f'{figure_path}/pro2.png', data[i,...])
scipy.misc.imsave(f'{figure_path}/rec2.png', rec_frames[i,...])

