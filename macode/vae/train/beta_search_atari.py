import tensorflow as tf

from forkan.models import VAE
from forkan.datasets import load_atari_normalized

latents = 20
betas = [0.5, 1.0, 2.0, 5.0, 10.0, 22.5, 50.6]
game = 'boxing'

# paper used adam
for beta in betas:
    data = load_atari_normalized(game)

    v = VAE(data.shape[1:], network='atari', name=game, beta=beta, latent_dim=latents)
    v.train(data, num_episodes=100)

    del data
    del v
