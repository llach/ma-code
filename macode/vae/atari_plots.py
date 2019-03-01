import logging
import numpy as np

from forkan import model_path
from forkan.common.utils import ls_dir
from forkan.models import VAE
from forkan.datasets import load_atari_normalized

from macode.vae.plot_helper import bars, plot_z_kl, plot_losses

logger = logging.getLogger(__name__)

network = 'atari'
filter = 'boxing'
plt_shape = [2, 10]

# whether to plot sigma-bars, kl plots and losses
modes = [True, False, False]

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

    # sigma bars
    if modes[0]:
        if ds_name not in datasets.keys():
            logger.info('loading {}'.format(ds_name))
            datasets.update({ds_name: load_atari_normalized(ds_name)})

        v = VAE(load_from=model_name, network=network)

        mus, logvars = v.encode(datasets[ds_name][:1024])[:2]
        sigmas = np.mean(np.exp(0.5 * logvars), axis=0)
        mus = np.mean(mus, axis=0)

        bars(d, sigmas, plt_shape, type='sigma', title=model_name)
        bars(d, mus, plt_shape, type='mu', title=model_name)
    elif modes[1]:
        plot_z_kl(d, split=True)
    elif modes[2]:
        plot_losses(d)

logger.info('Done.')
