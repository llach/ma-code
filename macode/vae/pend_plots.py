import csv
import logging
import numpy as np
from forkan import model_path
from forkan.common.utils import ls_dir
from forkan.models import VAE
from forkan.datasets import load_uniform_pendulum

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from macode.vae.plot_helper import bars, plot_losses

logger = logging.getLogger(__name__)

network = 'pendulum'
# filter = 'pendvisualuniform-b50-lat5-lr0.001-WU20-2019-03-19T09:54'.replace('/', ':')
filter = 'pendvisualuniform'.replace('/', ':')
plt_shape = [1, 5]


models_dir = '{}vae-{}/'.format(model_path, network)
dirs = ls_dir(models_dir)

for d in dirs:
    ds_name = d.split('/')[-1].split('-')[0]
    model_name = d.split('/')[-1]

    if filter is not None and not '':
        if filter not in model_name:
            logger.info('skipping {}'.format(model_name))
            continue

    kls, recs = [], []
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
                line_count += 1
            else:
                kls.append(row[kli])
                recs.append(row[reci])

                line_count += 1

    if kls == []:
        print('{} had no progress, skipping'.format(model_name))
        continue

    kls = kls[1:]
    recs = recs[1:]

    plt.figure(figsize=(10, 10))
    plt.subplot(221)

    data = load_uniform_pendulum()

    v = VAE(load_from=model_name, network=network)

    mus, logvars = v.encode(data[:4000])[:2]
    sigmas = np.mean(np.exp(0.5 * logvars), axis=0)

    bars(d, sigmas, plt_shape, type='sigma', title=model_name)
    plt.legend()

    plt.subplot(222)
    plt.plot(np.asarray(kls, dtype=np.float32)+np.asarray(recs, dtype=np.float32), label='joint-loss')
    plt.legend()

    plt.subplot(223)
    plt.plot(np.asarray(kls, dtype=np.float32), label='kl-loss')
    plt.legend()

    plt.subplot(224)
    plt.plot(np.asarray(recs, dtype=np.float32), label='rec-loss')
    plt.legend()

    plt.savefig('{}/results.png'.format(d))
    plt.show()

    # if modes[1]:
    #     plot_losses(d)
    # plt.savefig('{}/{}.png'.format(d, type))
logger.info('Done.')
