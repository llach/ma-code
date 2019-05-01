import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from forkan.datasets import load_set
from forkan.models import RetrainVAE

logger = logging.getLogger(__name__)

logger.info('loading dataset ...')
data = load_set('breakout-eval')
logger.info('done loading')

mdir = '/Users/llach/.forkan/done/breakout/ppo2-scratch/'

runs = [
    'breakout-nenv16-rlc1-k4-seed0-modelscratch-b1-2019-04-27T11/17'.replace('/', ':'),
    'breakout-nenv16-rlc10-k4-seed0-modelscratch-b1-2019-04-28T11/13'.replace('/', ':'),
]


v = RetrainVAE(f'{mdir}{runs[0]}/', (84, 84, 1), network='atari', latent_dim=20, beta=1, k=1, sess=tf.Session())
v.load()

rec_frames = np.squeeze(v.reconstruct(data))

fig, ax = plt.subplots(5, 2, figsize=(4, 10))

for i in range(5):
    ax[i, 0].imshow(data[i, ...], cmap="Greys")
    ax[i, 1].imshow(rec_frames[i, ...], cmap="Greys")

plt.show()
