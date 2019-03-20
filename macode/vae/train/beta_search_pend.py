import numpy as np

from forkan.models import VAE
from forkan.datasets import load_uniform_pendulum

latents = 5
lr = 1e-3
betas = 40+2.5*np.arange(17)

for beta in betas:
    print('loading data ...')
    data = load_uniform_pendulum()
    print('starting training!')
    v = VAE(data.shape[1:], network='pendulum', name='pendvisualuniform', beta=beta, lr=lr, latent_dim=latents)
    v.train(data, num_episodes=100)

    del data
    del v
