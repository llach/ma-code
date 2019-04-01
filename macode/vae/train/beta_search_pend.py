import numpy as np
import tensorflow as tf

from forkan.models import VAE
from forkan.datasets import load_uniform_pendulum

latents = 5
lr = 1e-3

for beta in np.round(np.linspace(85, 95, 15), 2):
    print('loading data ...')
    data = load_uniform_pendulum()
    print('starting training!')

    with tf.Session() as s:
        v = VAE(data.shape[1:], network='pendulum', name='pendvisualuniformONE',
                beta=beta, lr=lr, latent_dim=latents, session=s)
        v.train(data, batch_size=128, num_episodes=50, print_freq=200)

    tf.reset_default_graph()

    del data
    del v
