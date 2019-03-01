import matplotlib.pyplot as plt
import numpy as np

from forkan.models import VAE

name = 'trans-b28.67-lat10-lr0.001-2019-03-01T08:56'
v = VAE(load_from=name, network='dsprites')

idx = 5

np.random.seed(1)

latents = np.random.normal(0, 1, v.latent_dim)

for i, r in enumerate(np.linspace(-3, 3, 16)):
    latents[idx] = r

    img = v.decode(np.reshape(latents, [1, v.latent_dim]))

    plt.imshow(np.reshape(img, (64, 64)), cmap='Greys_r')
    plt.title('z_{}'.format(idx))
    plt.pause(0.1)
    plt.clf()
    i += 1

plt.show()
