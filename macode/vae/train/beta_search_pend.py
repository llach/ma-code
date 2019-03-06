from forkan.models import VAE
from forkan.datasets import load_pendulum

latents = 5
lrs = [1e-3, 1e-4]
betas = [0.5, 1.5, 2.0, 3.5, 5.0]

for lr in lrs:
    for beta in betas:
        data = load_pendulum()

        v = VAE(data.shape[1:], network='pendulum', name='', beta=beta, lr=lr, latent_dim=latents)
        v.train(data, num_episodes=100)

        del data
        del v
