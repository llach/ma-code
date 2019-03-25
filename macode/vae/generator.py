import matplotlib.pyplot as plt
import numpy as np

from forkan.models import VAE


name = 'pendvisualuniform-b80.0-lat5-lr0.001-2019-03-21T00/20'.replace('/', ':')
v = VAE(load_from=name, network='pendulum')
shape = v.input_shape[:2]

idx = 1

np.random.seed(1)

latents = np.random.normal(0, 1, v.latent_dim)

for i, r in enumerate(np.linspace(-3, 3, 20)):
    latents[idx] = r

    img = v.decode(np.reshape(latents, [1, v.latent_dim]))

    plt.imshow(np.reshape(img, shape), cmap='Greys_r')
    plt.title('z_{}'.format(idx))
    plt.pause(0.1)
    plt.clf()

plt.show()
