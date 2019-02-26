import tensorflow as tf

from forkan.models import VAE
from forkan.datasets import load_atari_normalized

learning_rate = 1e-3
latents = 20

betas = [0.5, 1.0, 2.0, 5.0, 10.0, 22.5, 50.6]

game = 'boxing'

data = load_atari_normalized(game)

# paper used adam
for beta in betas:
    v = VAE(data.shape[1:], name=game, lr=learning_rate, beta=beta, latent_dim=latents)
    v.train(data, num_episodes=100, print_freq=20)

    tf.reset_default_graph()
    del data
    del v
