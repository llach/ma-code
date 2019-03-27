import numpy as np
import cairocffi as cairo
from forkan.models import VAE
from scipy.stats import kendalltau as metric

import matplotlib.pyplot as plt
import seaborn as sns

FRAMES = 200

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


# b22
# np.sin(2*(thetas + np.pi/4))
# -np.cos(thetas + np.pi/4)

# load_from = 'pendvisualuniform-b20-lat5-lr0.001-2019-03-18T20/16'.replace('/', ':')
# idxes = [0, 1]
# load_from = 'pendvisualuniform-b22-lat5-lr0.001-2019-03-18T20/23'.replace('/', ':')
# idxes = [2, 3]
# load_from = 'pendvisualuniform-b80.0-lat5-lr0.001-2019-03-21T00/20'.replace('/', ':')
# idxes = [2]
load_from = 'pendvisualuniform-b1.0-lat5-lr0.001-2019-03-18T19/56'.replace('/', ':')
# load_from = 'pendvisualuniform-b77.5-lat5-lr0.001-2019-03-21T00/13'

load_from = load_from.replace('/', ':')
idxes = np.arange(5)

v = VAE(load_from=load_from, network='pendulum')
nlat = v.latent_dim
thetas = np.asarray(thetas, dtype=np.float)

idx = 1
show_recs = False

mus, logvars, zs = v.encode(frames)
print(zs.shape)
recons = v.full_decode(zs, frames)
print(recons.shape, frames.shape)

zs = np.moveaxis(zs, 0, -1)
mus = np.moveaxis(mus, 0, -1)
sigmas = np.moveaxis(np.exp(0.5 * logvars), 0, -1)
sigmasmean = np.mean(sigmas, axis=1)

print(zs.shape, thetas.shape)

if show_recs:
    for i in range(FRAMES):

        fig, axarr = plt.subplots(1, 2)
        fig.suptitle("Original vs. Reconstruction", fontsize=16)

        axarr[0].imshow(np.squeeze(frames[i,...]), cmap='Greys_r')
        axarr[0].set_title('input theta = {}'.format(thetas[i]))
        axarr[1].imshow(np.squeeze(recons[i,...]), cmap='Greys_r')
        axarr[1].set_title('x_hat mu = {}'.format(mus[idx][i]))

        # # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
        plt.setp(axarr[1].get_yticklabels(), visible=False)

        # Tight layout often produces nice results
        # but requires the title to be spaced accordingly
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)

        plt.show()

sns.set()

for idx in idxes:
    break
    plt.scatter(thetas, mus[idx], label='mus[{}]'.format(idx))
    # plt.plot(thetas, np.sin(thetas), label='sin(th)')
    plt.legend()
    plt.show()

    # plt.scatter(thetas, mus[idx], label='mus[{}]'.format(idx))
    # plt.plot(thetas, np.cos(thetas), label='cos(th)')
    # plt.legend()
    # plt.show()
    #
    # plt.scatter(thetas, sigmas[idx], label='sigmas')
    # plt.legend()
    # plt.show()

for idx in idxes:
    plt.plot(thetas, mus[idx], label='mus[{}]'.format(idx))
    plt.scatter(thetas, mus[idx])

# plt.plot(thetas, np.sin(thetas), label='sin')
# plt.plot(thetas, np.cos(thetas), label='cos')
plt.title(load_from)
plt.legend()
plt.show()

print('###### THETA ######')
for i in range(nlat):
    print(i, metric(thetas, mus[i]), sigmasmean[i])

print('###### SIN(THETA) ######')
for i in range(nlat):
    print(i, metric(np.sin(thetas), mus[i]), sigmasmean[i])

print('###### COS(THETA) ######')
for i in range(nlat):
    print(i, metric(np.cos(thetas), mus[i]), sigmasmean[i])


print(metric(mus[3], mus[4]))
