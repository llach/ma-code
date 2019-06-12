import csv
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns;

from forkan.common.utils import ls_dir

sns.set()

logger = logging.getLogger(__name__)


def read_keys(_dir, _filter, column_names):

    data = {}
    for cn in column_names:
        data.update({cn: []})

    dirs = ls_dir(_dir)
    print(len(dirs), dirs)
    assert len(dirs) > 0, f'filter {_filter} not found in {_dir}'
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
models_dir = f'{home}/.forkan/done/breakout/ppo2-fixed-vae'

vae_data = read_keys(f'{home}/.forkan/done/breakout/ppo2-fixed-vae', 'b1', ['mean_reward', 'nupdates'])

vae_xs = vae_data['nupdates'][0]
vae_ys = vae_data['mean_reward']

plt.plot(vae_xs, vae_ys[0], label='vae')

sc_models_dir = f'{home}/.forkan/done/breakout/ppo2-scratch'

data = read_keys(sc_models_dir, 'rlc10000.0-', ['mean_reward', 'nupdates'])

xs = data['nupdates'][0]
ys = data['mean_reward']

plt.plot(xs, np.nanmedian(ys, axis=0), label='kappa=1e4')

plt.title('fixed VAE vs. training from scratch')
plt.ylabel('Median Reward')
plt.xlabel('Number of Updates')

plt.legend()
plt.show()

logger.info('Done.')


