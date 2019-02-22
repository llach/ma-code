from forkan.models import VAE
from forkan.datasets import load_atari_normalized

learning_rate = 1e-4
beta = 5.5
latents = 20

for name in ['pong', 'breakout', 'boxing', 'gopher', 'upndown']:
    data = load_atari_normalized(name)
    v = VAE(data.shape[1:], name='name', lr=learning_rate, beta=beta, latent_dim=latents)
    v.train(data, num_episodes=50, print_freq=20)

    del data
    del v
