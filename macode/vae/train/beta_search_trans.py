import tensorflow as tf

from forkan.models import VAE
from forkan.datasets.dsprites import load_dsprites

learning_rate = 1e-2
latents = 10
opt = tf.train.AdagradOptimizer

# betas = [28.67, 40.96, 81.92]
betas = [40.96]

for beta in betas:
    (data, _) = load_dsprites('translation', repetitions=10)

    v = VAE(data.shape[1:], name='trans', network='dsprites', lr=learning_rate, beta=beta,
            latent_dim=latents, optimizer=opt, tensorboard=False)
    v.train(data, num_episodes=50, print_freq=-1)

    tf.reset_default_graph()
    del data
    del v
