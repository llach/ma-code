from forkan.datasets import load_atari_normalized
from forkan.models import VAE

betas = [1.0, 1.28, 2.0, 3.5, 5.0]

for beta in betas:
    data = load_atari_normalized('breakout')

    v = VAE(data.shape[1:], network='atari', name='breakout', beta=beta, lr=1e-4, latent_dim=20)
    v.train(data, num_episodes=100, print_freq=200, batch_size=128)

    del data
    del v
