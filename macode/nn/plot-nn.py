import csv
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

from forkan.common.utils import ls_dir

logger = logging.getLogger(__name__)


def read_keys(_dir, _filter, column_names):

    data = {}
    for cn in column_names:
        data.update({cn: []})

    dirs = ls_dir(_dir)
    for d in dirs:
        model_name = d.split('/')[-1]
        run_data = {}
        for cn in column_names:
            run_data.update({cn: []})

        if _filter is not None and not '':
            if _filter not in model_name:
                logger.info(f'skipping {model_name} | no match with {_filter}')
                continue

        if not os.path.isfile('{}/progress.csv'.format(d)):
            logger.info('skipping {} | file not found'.format(model_name))
            continue

        dic = csv.DictReader(open(f'{d}/progress.csv'))
        for cn in column_names:
            assert cn in dic.fieldnames, f'{cn} not in {dic.fieldnames}'

        for row in dic:
            for cn in column_names:
                run_data[cn].append(row[cn])

        for cn in column_names:
            data[cn].append(run_data[cn])

    for cn in column_names:
        data[cn] = np.asarray(data[cn], dtype=np.float32)
    return data


home = os.environ['HOME']
models_dir = f'{home}/.forkan/done/classify-ball/'

def smooth(l, N=1000):
    l = np.asarray(l, np.float32)
    return np.convolve(l, np.ones((N,)) / N, mode='valid')
    # return medfilt(np.asarray(l, dtype=np.float32), 51)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

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

ax[0, 0].set_title('TRAIN')

ax[0, 0].plot(smooth(vae_mae_train, 10), label='VAE')
ax[0, 0].plot(smooth(ret_mae_train, 10), label='SCRATCH')

ax[0, 0].set_ylabel('Mean Absolute Error')
ax[0, 0].set_xlabel('Number of Updates')

ax[0, 0].legend()
# ax[0, 0].set_ylim(bottom=0.12, top=0.26)


ax[0, 1].plot(smooth(vae_mse_train, 10), label='VAE')
ax[0, 1].plot(smooth(ret_mse_train, 10), label='SCRATCH')

ax[0, 1].set_ylabel('Mean Squared Error')
ax[0, 1].set_xlabel('Number of Updates')

# ax[0, 1].set_ylim(bottom=0.04, top=0.1)

ax[0, 1].legend()


ax[1, 0].set_title('TEST')

ax[1, 0].plot(smooth(vae_mae_test, 10), label='VAE')
ax[1, 0].plot(smooth(ret_mae_test, 10), label='SCRATCH')

ax[1, 0].set_ylabel('Mean Absolute Error')
ax[1, 0].set_xlabel('Number of Updates')

ax[1, 0].legend()
# ax[1, 0].set_ylim(bottom=0.12, top=0.26)

ax[1, 1].plot(smooth(vae_mse_test, 10), label='VAE')
ax[1, 1].plot(smooth(ret_mse_test, 10), label='SCRATCH')

ax[1, 1].set_ylabel('Mean Squared Error')
ax[1, 1].set_xlabel('Number of Updates')

ax[1, 1].legend()
# ax[1, 1].set_ylim(bottom=0.0, top=0.1)

fig.suptitle('MAE & MSE for joint & static models')
fig.tight_layout()

plt.savefig(f'{home}/ball.png')
plt.show()

logger.info('Done.')
