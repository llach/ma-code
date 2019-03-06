import numpy as np

from forkan.models import VAE
from forkan.datasets import load_pendulum

latents = 5
lrs = [1e-3, 1e-4]
betas = [0.5, 1.5, 2.0, 3.5, 5.0]

for lr in lrs:
    for beta in betas:
        print('loading data ...')
        data = load_pendulum()
        data = np.expand_dims(data, -1)
        print('starting training!')
        v = VAE(data.shape[1:], network='pendulum', name='', beta=beta, lr=lr, latent_dim=latents)
        v.train(data, num_episodes=100)

        del data
        del v
