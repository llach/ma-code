import numpy as np
import cairocffi as cairo
from forkan.models import VAE
from scipy.stats import pearsonr as metric

FRAMES = 20

thetas = []
frames = np.zeros([FRAMES, 64, 64, 1])

w,  h = 64, 64
surf = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)


def _render_pendulum(th):
    cr = cairo.Context(surf)

    # draw background
    cr.set_source_rgb(1, 1, 1)
    cr.paint()

    # apply transforms
    cr.translate((w / 2), h / 2)
    cr.rotate(np.pi - th)

    # draw shapes that form the capsule
    cr.rectangle(-2.5, 0, 5, 27)
    cr.arc(0, 0, 2.5, 0, 2 * np.pi)
    cr.arc(0, (h / 2) - 4, 2.5, 0, 2 * np.pi)

    # draw color
    cr.set_source_rgb(.8, .3, .3)
    cr.fill()

    # center sphere
    cr.arc(0, 0, 1, 0, 2 * np.pi)
    cr.set_source_rgb(0, 0, 0)
    cr.fill()

    # reshape, delete fourth (alpha) channel, greyscale and normalise
    return np.expand_dims(np.dot(np.frombuffer(surf.get_data(), np.uint8).reshape([w, h, 4])[..., :3],
                                 [0.299, 0.587, 0.114]), -1) / 255


for i, theta in enumerate(np.linspace(0, 2*np.pi, FRAMES)):
    frame = _render_pendulum(theta)
    frames[i] = frame
    thetas.append(theta)


v = VAE(load_from='pendvisualuniform-b17.5-lat5-lr0.001-2019-03-15T13/42'.replace('/', ':'), network='pendulum')
nlat = v.latent_dim

thetas = np.asarray(thetas, dtype=np.float)

mus, logvars, zs = v.encode(frames)
zs = zs.reshape(nlat, FRAMES)
mus = mus.reshape(nlat, FRAMES)
sigmas = np.mean(np.exp(0.5 * logvars), axis=0)
print(zs.shape, thetas.shape)

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# plt.plot(zs[0], label='z0')
# plt.plot(zs[1], label='z1')
# plt.plot(zs[2], label='z2')
# plt.plot(zs[3], label='z3')
# plt.legend()
# plt.show()

plt.plot(thetas, zs[1], label='zs')
plt.plot(thetas, np.cos(thetas), label='sin(th)')
plt.legend()
plt.show()

# print(metric(zs[0], zs[4]))

# sns.distplot(thetas, label='thetas')
# plt.legend()
# plt.show()
#
# sns.distplot(np.sin(thetas), label='thetas-sin')
# plt.legend()
# plt.show()
#
# plt.scatter(thetas, zs[2], label='z')
# plt.legend()
# plt.show()
#
# sns.distplot(np.sin(zs[2]), label='zs-sin')
# plt.legend()
# plt.show()

# plt.scatter(thetas, zs[2], label='th')
# plt.legend()
# plt.show()
# plt.scatter(np.sin(thetas), zs[2], label='thsin')
# plt.legend()
# plt.show()
# plt.scatter(np.cos(thetas), zs[2], label='thcos')
# plt.legend()
# plt.show()

from scipy.stats import norm
#
# print('###### THETA ######')
# for i in range(nlat):
#     print(i, metric(thetas, zs[i]), sigmas[i])
#
# print('###### sin(THETA) ######')
# for i in range(nlat):
#     print(i, metric(np.sin(thetas), zs[i]), sigmas[i])
#
# print('###### cos(THETA) ######')
# for i in range(nlat):
#     print(i, metric(np.cos(thetas), zs[i]), sigmas[i])