import matplotlib.pyplot as plt
import numpy as np

from forkan.models import VAE


name = 'upndown-b3.5-lat20-lr0.001-2019-03-02T03:06'
v = VAE(load_from=name, network='atari')
shape = v.input_shape[:2]

idx = 5

np.random.seed(1)

latents = np.random.normal(0, 1, v.latent_dim)

for i, r in enumerate(np.linspace(-3, 3, 10)):
    latents[idx] = r

    img = v.decode(np.reshape(latents, [1, v.latent_dim]))

    plt.imshow(np.reshape(img, shape), cmap='Greys_r')
    plt.title('z_{}'.format(idx))
    plt.pause(0.1)
    plt.clf()
    i += 1

plt.show()
