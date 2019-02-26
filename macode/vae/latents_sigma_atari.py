import logging
import numpy as np
import tensorflow as tf

from forkan import model_path
from forkan.common.utils import ls_dir
from forkan.models import VAE
from forkan.datasets import load_atari_normalized

from macode.vae.plot_helper import sigma_bars

logger = logging.getLogger(__name__)

network = 'atari'
filter = ''
plt_shape = [2, 10]

datasets = {}

models_dir = '{}vae-{}/'.format(model_path, network)
dirs = ls_dir(models_dir)

for d in dirs:
    ds_name = d.split('/')[-1].split('-')[0]
    model_name = d.split('/')[-1]

    if filter is not None and not '':
        if filter not in model_name:
            logger.info('skipping {}'.format(model_name))
            continue

    if ds_name not in datasets.keys():
        logger.info('loading {}'.format(ds_name))
        datasets.update({ds_name: load_atari_normalized(ds_name)})

    v = VAE(load_from=model_name)

    sigmas = np.exp(0.5 * v.encode(datasets[ds_name][:1024])[:,1,:])
    sigmas = np.mean(sigmas, axis=0)

    sigma_bars(sigmas, plt_shape, title=model_name)

    tf.reset_default_graph()

logger.info('Done.')
