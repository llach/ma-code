import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

from forkan.common.utils import read_keys, get_figure_size

logger = logging.getLogger(__name__)


home = os.environ['HOME']
models_dir = f'{home}/.forkan/done/classify-ball/'

def smooth(l, N=1000):
    l = np.asarray(l, np.float32)
    return np.convolve(l, np.ones((N,)) / N, mode='valid')
    # return medfilt(np.asarray(l, dtype=np.float32), 51)

fig, ax = plt.subplots(1, 1, figsize=get_figure_size())

data = read_keys(models_dir, 'RETRAIN-N128', ['nbatch', 'mae_train', 'mse_train', 'mae_test', 'mse_test'])

ret_mae_train = np.squeeze(data['mae_train'])
ret_mae_test = np.squeeze(data['mae_test'])
ret_mse_train = np.squeeze(data['mse_train'])
ret_mse_test = np.squeeze(data['mse_test'])
ret_nbatch = np.squeeze(data['nbatch'])

data = read_keys(models_dir, 'VAE-N128', ['nbatch', 'mae_train', 'mse_train', 'mae_test', 'mse_test'])

vae_mae_train = np.squeeze(data['mae_train'])
vae_mae_test = np.squeeze(data['mae_test'])
vae_mse_train = np.squeeze(data['mse_train'])
vae_mse_test = np.squeeze(data['mse_test'])
vae_nbatch = np.squeeze(data['nbatch'])

ax.plot(smooth(vae_mse_test, 10), label='VAE')
ax.plot(smooth(ret_mse_test, 10), label='SCRATCH')

ax.set_ylabel('Mean Squared Error')
ax.set_xlabel('Number of Updates')

ax.legend()

fig.tight_layout()

plt.savefig(f'{home}/classify-ball.png')
plt.show()

logger.info('Done.')
