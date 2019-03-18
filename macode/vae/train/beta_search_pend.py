import numpy as np

from forkan.models import VAE
from forkan.datasets import load_uniform_pendulum

latents = 5
lr = 1e-3
betas = [0.5, 1.0, 2, 4, 20, 22, 30, 40, 50]

for wu, bn in [(20, True), (None, False)]:
    for beta in betas:
        print('loading data ...')
        data = load_uniform_pendulum()
        print('starting training!')
        v = VAE(data.shape[1:], network='pendulum', name='pendvisualuniform', beta=beta, lr=lr, latent_dim=latents,
                warmup=wu, batch_norm=bn)
        v.train(data, num_episodes=100)

        del data
        del v
