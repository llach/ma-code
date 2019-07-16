import csv
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from forkan.common.utils import read_keys, setup_plotting, get_figure_size


logger = logging.getLogger(__name__)

ylims, tick_setup = setup_plotting()

home = os.environ['HOME']
models_dir = f'{home}/.forkan/done/pendulum/ppo2-beta-exp'
col = ['#8cd17d', '#b6992d', '#f1ce63', '#499894', '#86bcb6']

for rlc in [1, 30]:
    fig, ax = plt.subplots(1, 1, figsize=get_figure_size())
    i = 0
    for beta in [5, 20, 30, 60, 80]:

        data = read_keys(models_dir, [f'rlc{rlc}-', f'b{beta}-'], ['mean_reward', 'total_timesteps'])

        xs = data['total_timesteps'][0]
        ys = data['mean_reward']

        ax.plot(xs, np.nanmedian(ys, axis=0), label=f'$\\beta={beta}$', color=col[i])
        ax.fill_between(xs, np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.33, color=col[i])
        i+=1
    plt.ylim(**ylims)

    ax.set_ylabel('Median Reward')
    ax.set_xlabel('Steps')
    plt.xticks(tick_setup[0], tick_setup[1])
    ax.legend(loc='upper left')

    fig.tight_layout()

    plt.savefig(f'{home}/.forkan/done/pendulum/figures/rew-rlc{rlc}.pdf')
    plt.show()

logger.info('Done.')
