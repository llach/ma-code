from forkan.models import VAE
from forkan.datasets.dsprites import load_dsprites

latents = 10
opt = 'adam'

betas = [28.67, 40.96, 81.92]

for beta in betas:
    (data, _) = load_dsprites('translation', repetitions=10)

    v = VAE(data.shape[1:], name='trans', network='dsprites', beta=beta, latent_dim=latents, optimizer=opt)
    v.train(data, num_episodes=100)

    del data
    del v
