import tensorflow as tf

from forkan.models import VAE
from forkan.datasets.dsprites import load_dsprites

learning_rate = 1e-2
latents = 5

betas = [0.5, 1.0, 2.0, 5.0, 10.0, 22.5, 50.6]

(data, _) = load_dsprites('translation', repetitions=10)

for beta in betas:
    v = VAE(data.shape[1:], name='trans', lr=learning_rate, beta=beta, latent_dim=latents)
    v.train(data, num_episodes=50, print_freq=-1)

    tf.reset_default_graph()
    del data
    del v
