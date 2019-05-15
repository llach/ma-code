import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from forkan.datasets import load_uniform_pendulum
from forkan.models import RetrainVAE

logger = logging.getLogger(__name__)

logger.info('loading dataset ...')
data = np.squeeze(load_uniform_pendulum()[:5])
logger.info('done loading')

mdir = '/Users/llach/.forkan/done/pendulum/ppo2-scratch/'

runs = [
    'pendulumvisual-nenv16-rlc1-k5-stop0.01-seed0-modelscratch-b1-2019-04-23T10/42'.replace('/', ':'),
    'pendulumvisual-nenv16-rlc10-k5-stop0.01-seed0-modelscratch-b1-2019-04-24T13/58'.replace('/', ':'),
    'pendulumvisual-nenv16-rlc30-k5-stop0.01-seed0-modelscratch-b1-2019-04-26T02/18'.replace('/', ':'),
]


v = RetrainVAE(f'{mdir}{runs[2]}/', (64, 64, 1), network='pendulum', latent_dim=5, beta=1, k=1, sess=tf.Session())
v.load()

rec_frames = np.squeeze(v.reconstruct(data))

fig, ax = plt.subplots(5, 2, figsize=(4, 10))

for i in range(5):
    ax[i, 0].imshow(data[i, ...], cmap="Greys")
    ax[i, 1].imshow(rec_frames[i, ...], cmap="Greys")

plt.show()
