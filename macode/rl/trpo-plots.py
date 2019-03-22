import csv
import logging
import numpy as np

from forkan import model_path
from forkan.common.utils import ls_dir

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import matplotlib as mlp

# print(mlp.rcParams)

log = logging.getLogger(__name__)

models_dir = '{}trpo/'.format(model_path)
dirs = ls_dir(models_dir)

filter = '16'
for d in dirs:
    ds_name = d.split('/')[-1].split('-')[0]
    model_name = d.split('/')[-1]

    if filter is not None and not '':
        if filter not in model_name:
            log.info('skipping {}'.format(model_name))
            continue

    log.info('plotting {}'.format(model_name))

    steps, rew, entr, ev, vall = [], [], [], [], []
    stepi = rewi = entri = evi = valli = None

    with open('{}/progress.csv'.format(d)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                for n, ele in enumerate(row):
                    if 'reward' in ele:
                        rewi = n
                    elif 'entropy' in ele:
                        entri = n
                    elif 'timesteps' in ele:
                        stepi = n
                    elif ele == 'explained_variance':
                        evi = n
                    elif ele == 'value_loss':
                        valli = n
                line_count += 1
            else:
                if np.asarray(row[rewi], dtype=np.float) == -np.infty:
                    pass
                else:
                    rew.append(row[rewi])
                    entr.append(row[entri])
                    ev.append(row[evi])
                    vall.append(row[valli])
                    steps.append(row[stepi])

                line_count += 1

    if rew == []:
        log.info('{} had no progress, skipping'.format(model_name))
        continue

    steps = np.asarray(steps, dtype=np.float32)

    second_line = steps[-1] > 19e5
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('{}'.format(model_name), fontsize=14)

    # sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

    print('plotting losses for {} ...'.format(model_name))
    axes[0, 0].axvline(1e6, linestyle='--', color='c')
    if second_line: axes[0, 0].axvline(2e6, linestyle='--', color='c')
    axes[0, 0].plot(steps, np.asarray(rew, dtype=np.float32), label='mean reward')
    # axes[0, 0].legend()
    axes[0, 0].set_title('reward')

    axes[0, 1].axvline(1e6, linestyle='--', color='c')
    if second_line: axes[0, 1].axvline(2e6, linestyle='--', color='c')
    axes[0, 1].plot(steps, np.asarray(entr, dtype=np.float32), label='policy entropy')
    # axes[0, 1].legend()
    axes[0, 1].set_title('policy entropy')

    axes[1, 0].axvline(1e6, linestyle='--', color='c')
    if second_line: axes[1, 0].axvline(2e6, linestyle='--', color='c')
    axes[1, 0].plot(steps, np.asarray(ev, dtype=np.float32), label='explained variance')
    axes[1, 0].set_title('explained variance')

    axes[1, 1].axvline(1e6, linestyle='--', color='c')
    if second_line: axes[1, 1].axvline(2e6, linestyle='--', color='c')
    axes[1, 1].plot(steps, np.asarray(vall, dtype=np.float32), label='value loss')
    axes[1, 1].set_title('value loss')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.88)

    plt.savefig('{}/plot.png'.format(d))
    plt.show()

log.info('Done.')
