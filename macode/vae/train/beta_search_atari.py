from forkan.models import VAE
from forkan.datasets import load_atari_normalized

lrs = [1e-3, 1e-4]
latents = 20
betas = [0.5, 2.0, 3.5, 5.0]
games = ['gopher', 'upndown', 'pong', 'breakout']

for game in games:
    for lr in lrs:
        for beta in betas:
            data = load_atari_normalized(game)

            v = VAE(data.shape[1:], network='atari', name=game, beta=beta, lr=lr, latent_dim=latents)
            v.train(data[:128], num_episodes=1)

            del data
            del v
