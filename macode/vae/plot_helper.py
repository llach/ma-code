import csv

import cairocffi as cairo
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns;
from scipy.stats import kendalltau as metric

sns.set()


def bars(d, mean_sigma, plot_shape, type='sigma', thresh=0.8, title=None, axes=[]):
    num_zi = plot_shape[1]
    num_plots = plot_shape[0]

    # split zi into multiple bar plots
    ys = []
    xs = []
    pal = []

    for i in range(num_plots):
        xs.append(['z-{}'.format(j + (num_zi * i)) for j in range(num_zi)])
        ys.append(mean_sigma[i * num_zi:(i + 1) * num_zi])
        pal.append(['#90D7F3' if k > thresh else '#F78A8F' for k in ys[-1]])

    # create subplots
    t = 'z-i {}'.format(type)
    if title is not None:
        t += ' - {}'.format(title)
    plt.title(t)

    # show them heatmaps
    if num_plots == 1:
        sns.barplot(x=xs[0], y=ys[0], palette=pal[0], linewidth=0.5, label='mean-sigmas')
        plt.ylim(0, 1.05)
    elif len(axes) == num_plots:
        for i, ax in enumerate(axes):
            sns.barplot(x=xs[i], y=ys[i], palette=pal[i], linewidth=0.5, label='mean-sigmas', ax=ax)
            ax.set_ylim(0, 1.05)
    else:
        print('for multiple plots give as many subplots as indicated in plot_shape')


def plot_z_kl(d, split=True):
    model_name = d.split('/')[-1]

    z_idx = []
    z_kls = []

    with open('{}/progress.csv'.format(d)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                for n, ele in enumerate(row):
                    if 'z' in ele:
                        z_idx.append(n)
                        z_kls.append([])
                line_count += 1
            else:
                zk = [row[i] for i in z_idx]
                for n, zkl in enumerate(zk):
                    z_kls[n].append(zkl)

    print('plotting z_kls for {} ...'.format(model_name))
    if split:
        half = len(z_idx)//2

        plt.subplot(211)
        for n, zz in enumerate(z_kls[:half]):
            plt.plot(np.asarray(zz, dtype=np.float32), label='z{}-kl'.format(n))
        plt.legend()

        plt.subplot(212)
        for n, zz in enumerate(z_kls[half:]):
            plt.plot(np.asarray(zz, dtype=np.float32), label='z{}-kl'.format(n+half))

        pass
    else:
        for zz in z_kls:
            plt.plot(np.asarray(zz, dtype=np.float32))

    plt.legend()
    plt.title(model_name)

    plt.savefig('{}/kls.png'.format(d))
    plt.show()


def plot_losses(d):
    model_name = d.split('/')[-1]

    l_idx, kl_idx = 0, 0
    loss, kl = [], []

    with open('{}/progress.csv'.format(d)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                for n, ele in enumerate(row):
                    print(ele)
                    if ele == 'loss':
                        l_idx = n
                    elif ele == 'kl-loss':
                        kl_idx = n
                line_count += 1
            else:
                loss.append(row[l_idx])
                kl.append(row[kl_idx])

    print('plotting losses for {} ...'.format(model_name))
    plt.subplot(211)
    plt.plot(np.asarray(loss, dtype=np.float32), label='loss')
    plt.legend()

    plt.subplot(212)
    plt.plot(np.asarray(kl, dtype=np.float32), label='kl-loss')
    plt.legend()

    plt.title(model_name)

    plt.savefig('{}/losses.png'.format(d))
    plt.show()



def plot_latents(d, v, load_from, show_recs=False):
    FRAMES = 200

    thetas = []
    frames = np.zeros([FRAMES, 64, 64, 1])

    w, h = 64, 64
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

    for i, theta in enumerate(np.linspace(0, 2 * np.pi, FRAMES)):
        frame = _render_pendulum(theta)
        frames[i] = frame
        thetas.append(theta)

    idxes = np.arange(5)

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


