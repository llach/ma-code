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

def plot_latents(d, load_from, show_recs=False):

    global thetas

    load_from = load_from.replace('/', ':')
    idxes = np.arange(5)

    v = VAE(load_from=load_from, network='pendulum')
    nlat = v.latent_dim
    thetas = np.asarray(thetas, dtype=np.float)

    idx = 1
    show_recs = False

    mus, logvars = v.encode(frames)
    mus = np.moveaxis(mus, 0, -1)
    sigmas = np.moveaxis(np.exp(0.5 * logvars), 0, -1)
    sigmasmean = np.mean(sigmas, axis=1)

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
        plt.plot(thetas, mus[idx], label='mus[{}]'.format(idx))
        plt.scatter(thetas, mus[idx])

    plt.title(load_from)
    plt.legend()

    plt.savefig('{}/theta_traversal.png'.format(d))
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


import tensorflow as tf
from forkan import model_path
from forkan.common.utils import ls_dir

filter = ''
network = 'pendulum'

models_dir = '{}TFvae-{}/'.format(model_path, network)
dirs = ls_dir(models_dir)

for d in dirs:
    ds_name = d.split('/')[-1].split('-')[0]
    model_name = d.split('/')[-1]

    if filter is not None and not '':
        if filter not in model_name:
            print('skipping {}'.format(model_name))
            continue

    plot_latents(d, model_name)

    tf.reset_default_graph()
