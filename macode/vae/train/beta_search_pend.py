import os
import csv
import logging

import numpy as np
import tensorflow as tf

from forkan import model_path
from forkan.models import VAE
from forkan.datasets import load_uniform_pendulum
from shutil import rmtree

latents = 5
lr = 1e-3

logger = logging.getLogger(__name__)
#np.round(np.linspace(65, 90, 150), 2)
for seed in [0]:
    for beta in np.asarray(np.arange(10)+76, dtype=float):

        np.random.seed(seed)
        tf.set_random_seed(seed)

        print('loading data ...')
        data = load_uniform_pendulum()
        print('starting training!')

        with tf.Session() as s:
            v = VAE(data.shape[1:], network='pendulum', name='pendvisualuniform',
                    beta=beta, lr=lr, latent_dim=latents, session=s)
            v.train(data, batch_size=128, num_episodes=50, print_freq=200)

        v.csv.flush()
        d = '{}/vae-pendulum/{}/'.format(model_path, v.savename)

        ds_name = d.split('/')[-1].split('-')[0]
        model_name = d.split('/')[-1]

        print(d)
        if not os.path.isfile('{}progress.csv'.format(d)):
            logger.info('skipping {} | file not found'.format(model_name))
            continue

        zidx, zs = [], []

        with open('{}progress.csv'.format(d)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    for n, ele in enumerate(row):
                        if '-kl' in ele:
                            zidx.append(n)

                    for _ in zidx:
                        zs.append([])

                    line_count += 1
                else:

                    for n, idx in enumerate(zidx):
                        zs[n].append(row[idx])

                    line_count += 1

        if zs[0] == []:
            print('{} had no progress, skipping'.format(model_name))
            continue

        for r in range(len(zidx)):
            zs[r] = zs[r][(line_count//2):]

        zs = np.asarray(zs, dtype=np.float32)
        zm = np.mean(zs, axis=-1)
        num_z = len(zm[np.where(zm > 0.2)])

        if num_z != 1:
            print(f'removing {model_path} bc zm = {num_z}')
            rmtree(d)

        tf.reset_default_graph()

        del data
        del v
