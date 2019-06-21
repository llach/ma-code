import csv
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from forkan.common.utils import ls_dir, setup_plotting, get_figure_size

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


ylims = setup_plotting()

home = os.environ['HOME']
models_dir = f'{home}/.forkan/done/pendulum/ppo2-retrain-raw-stop'

fig, ax = plt.subplots(1, 1, figsize=get_figure_size())

for fi, name in [('b1', 'ent'), ('b81', 'one lat'), ('b85', 'two lat')]:
    data = read_keys(models_dir, fi, ['mean_reward', 'nupdates'])

    xs = data['nupdates'][0]
    ys = data['mean_reward']

    ax.plot(xs, np.nanmedian(ys, axis=0), label=name)
    ax.fill_between(xs, np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.33)


plt.ylim(**ylims)

ax.set_ylabel('Median Reward')
ax.set_xlabel('Number of Updates')
ax.legend(loc='center right')

fig.tight_layout()

plt.savefig(f'{models_dir}/ret-raw-stop.pdf')
plt.show()

logger.info('Done.')
