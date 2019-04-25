import csv
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns;
import tensorflow as tf

from forkan import model_path
from forkan.common.utils import ls_dir
from forkan.datasets import load_atari_normalized
from forkan.models import VAE

sns.set()

from macode.vae.plot_helper import bars

logger = logging.getLogger(__name__)

network = 'atari'
filter = ''.replace('/', ':')
plt_shape = [2, 10]


models_dir = '{}vae-{}/'.format(model_path, network)
dirs = ls_dir(models_dir)


def smooth(l):
    l = np.asarray(l, np.float32)
    N = 1000
    return np.convolve(l, np.ones((N,)) / N, mode='valid')
    # return medfilt(np.asarray(l, dtype=np.float32), 51)


logger.info('loading dataset ...')
data = load_atari_normalized('breakout-small')
logger.info('done loading')

for d in dirs:
    ds_name = d.split('/')[-1].split('-')[0]
    model_name = d.split('/')[-1]

    if filter is not None and not '':
        if filter not in model_name:
            logger.info('skipping {}'.format(model_name))
            continue

    if os.path.isfile(f'{d}/results.png') and os.path.isfile(f'{d}/z-kls.png') and os.path.isfile(f'{d}/theta_traversal.png'):
        logger.info('skipping {} | pngs exist'.format(model_name))
        continue

    if not os.path.isfile('{}/progress.csv'.format(d)):
        logger.info('skipping {} | file not found'.format(model_name))
        continue

    kls, recs, zidx, zs = [], [], [], []
    kli = reci = None

    with open('{}/progress.csv'.format(d)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                for n, ele in enumerate(row):
                    if 'rec-loss' in ele:
                        reci = n
                    elif 'kl-loss' in ele:
                        kli = n
                    elif '-kl' in ele:
                        zidx.append(n)

                for _ in zidx:
                    zs.append([])

                line_count += 1
            else:
                kls.append(row[kli])
                recs.append(row[reci])

                for n, idx in enumerate(zidx):
                    zs[n].append(row[idx])

                line_count += 1

    if kls is []:
        print('{} had no progress, skipping'.format(model_name))
        continue

    v = VAE(load_from=model_name, network=network)

    sns.reset_orig()
    fig, ax = plt.subplots(5, 2, figsize=(4, 10))

    np.random.seed(0)
    rand_frames = data[np.random.randint(0, data.shape[0]-1, size=(5,))]
    rec_frames = v.reconstruct(rand_frames)

    for i in range(5):
        ax[i, 0].imshow(rand_frames[i, ...], cmap="Greys")
        ax[i, 1].imshow(np.squeeze(rec_frames[i, ...]), cmap="Greys")

    for i, ax in enumerate(fig.axes):
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)

    plt.savefig('{}/recons.png'.format(d))
    plt.show()

    sns.set()
    plt.figure(0, figsize=(10, 10))
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((4, 4), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=2)
    ax4 = plt.subplot2grid((4, 4), (2, 0), colspan=2, rowspan=2)
    ax5 = plt.subplot2grid((4, 4), (2, 2), colspan=2, rowspan=2)

    mus, logvars = v.encode(data[:4000])[:2]
    sigmas = np.mean(np.exp(0.5 * logvars), axis=0)

    bars(d, sigmas, plt_shape, type='sigma', title=model_name, axes=[ax1, ax2])

    kls = smooth(kls[100:])
    recs = smooth(recs[100:])

    ax3.plot(np.asarray(kls, dtype=np.float32)+np.asarray(recs, dtype=np.float32), label='joint-loss')
    ax3.legend()

    ax4.plot(np.asarray(kls, dtype=np.float32), label='kl-loss')
    ax4.legend()

    ax5.plot(np.asarray(recs, dtype=np.float32), label='rec-loss')
    ax5.legend()

    plt.title(model_name)
    plt.savefig('{}/results.png'.format(d))
    plt.show()

    for n, zkl in enumerate(zs):
        zkl = smooth(zkl)
        plt.plot(zkl, label='z{}'.format(n))

    plt.title(f'latent kls - {model_name}')

    plt.savefig('{}/z-kls.png'.format(d))
    plt.show()

    tf.reset_default_graph()

logger.info('Done.')
