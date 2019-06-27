import tensorflow as tf

from forkan.datasets import load_set
from forkan.models import VAE

zetas = [1e5, 1e6, 1e7] # [2, 5, 30, 100, 1000, 1e4]

for zeta in zetas:
    data = load_set('breakout-normalized-small')

    v = VAE(data.shape[1:], network='atari', name='breakout', zeta=zeta, lr=1e-4, latent_dim=20,
            tensorboard=True)
    v.train(data, num_episodes=50, print_freq=100, batch_size=128)

    v.s.close()
    tf.reset_default_graph()
    del data
    del v
