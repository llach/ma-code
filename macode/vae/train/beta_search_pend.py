from forkan.models import VAE
from forkan.datasets import load_pendulum

latents = 5
lr = 1e-3
betas = [6.0, 8.0, 10.0, 12.0, 20.0]

for beta in betas:
    print('loading data ...')
    data = load_pendulum()
    print('starting training!')
    v = VAE(data.shape[1:], network='pendulum', name='pend', beta=beta, lr=lr, latent_dim=latents)
    v.train(data, num_episodes=100)

    del data
    del v
