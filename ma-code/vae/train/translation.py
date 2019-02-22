from forkan.models import VAE
from forkan.datasets.dsprites import load_dsprites


(data, _) = load_dsprites('translation', repetitions=10)
v = VAE(data.shape[1:], name='dtrans', network='dsprites', beta=4.1, latent_dim=10, lr=1e-2)
v.train(data, num_episodes=50, print_freq=20)