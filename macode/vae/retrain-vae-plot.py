import csv
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns;
import tensorflow as tf

from forkan import model_path
from forkan.datasets import load_uniform_pendulum
from forkan.models import RetrainVAE

sns.set()

from macode.vae.plot_helper import bars, plot_latents
from scipy.signal import medfilt

logger = logging.getLogger(__name__)

network = 'pendulum'
filter = 'SEARCH'.replace('/', ':')
plt_shape = [1, 5]


d = f'{model_path}/retrain/retrain/'
ds_name = d.split('/')[-1].split('-')[0]
model_name = d.split('/')[-1]

if filter is not None and not '':
    if filter not in model_name:
        logger.info('skipping {}'.format(model_name))

if os.path.isfile(f'{d}/vae/results.png') and os.path.isfile(f'{d}/z-kls.png') and os.path.isfile(f'{d}/theta_traversal.png'):
    logger.info('skipping {} | pngs exist'.format(model_name))

if not os.path.isfile('{}/vae/progress.csv'.format(d)):
    logger.info('skipping {} | file not found'.format(model_name))

kls, recs, zidx, zs = [], [], [], []
kli = reci = None

with open('{}/vae/progress.csv'.format(d)) as csv_file:
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

if kls == []:
    print('{} had no progress, skipping'.format(model_name))


kls = medfilt(np.asarray(kls[100:], dtype=np.float32), 19)
recs = medfilt(np.asarray(recs[100:], dtype=np.float32), 19)

plt.figure(figsize=(10, 10))
plt.subplot(221)

data = load_uniform_pendulum()

v = RetrainVAE(d, (64, 64, 1), latent_dim=5, sess=tf.Session(), k=1, with_attrs=True)
v.s.run(tf.global_variables_initializer())
v.load()

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

for n, zkl in enumerate(zs):
    zkl = medfilt(np.asarray(zkl, dtype=np.float32), 19)
    plt.plot(zkl, label='z{}'.format(n))

plt.title('each latent\'s kl')
plt.legend()

plt.savefig('{}/z-kls.png'.format(d))
plt.show()

plot_latents(d, v, model_name)

tf.reset_default_graph()

logger.info('Done.')
