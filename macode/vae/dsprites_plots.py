import logging
import numpy as np
from forkan import model_path
from forkan.common.utils import ls_dir
from forkan.models import VAE
from forkan.datasets.dsprites import load_dsprites

from macode.vae.plot_helper import bars, plot_losses

logger = logging.getLogger(__name__)

network = 'dsprites'
filter = ''
plt_shape = [1, 10]

# whether to plot sigma-bars, kl plots and losses
modes = [True, False]

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
        (data, _) = load_dsprites('translation', repetitions=10)

        v = VAE(load_from=model_name)

        mus, logvars = v.encode(data[:1024])[:2]
        sigmas = np.mean(np.exp(0.5 * logvars), axis=0)

        bars(d, sigmas, plt_shape, type='sigma', title=model_name)
        bars(d, mus, plt_shape, type='mu', title=model_name)
    if modes[1]:
        plot_losses(d)

logger.info('Done.')
