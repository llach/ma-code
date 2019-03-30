import numpy as np
import tensorflow as tf

from forkan.models import VAE
from forkan.datasets import load_uniform_pendulum

latents = 5
lr = 1e-3
betas = 70+2.5*np.arange(5)

for beta in np.append([1, 22.5], betas):
    print('loading data ...')
    data = load_uniform_pendulum()
    print('starting training!')

    with tf.Session() as s:
        v = VAE(data.shape[1:], network='pendulum', name='pendvisualuniform',
                beta=beta, lr=lr, latent_dim=latents, sess=s)
        v.train(data, num_episodes=100, print_freq=200)

    tf.reset_default_graph()

    del data
    del v
