import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from forkan.common.utils import read_keys, setup_plotting, get_figure_size


logger = logging.getLogger(__name__)

ylims, tick_setup = setup_plotting('break-baseline')
fig, ax = plt.subplots(1, 1, figsize=get_figure_size())

home = os.environ['HOME']
models_dir = f'{home}/.forkan/done/breakout/baselines/'

for fi, name in [('', '')]:
    data = read_keys(models_dir, fi, ['mean_reward', 'total_timesteps'])

    xs = data['total_timesteps'][0]
    ys = data['mean_reward']

    plt.plot(xs, np.nanmedian(ys, axis=0), label=name)
    plt.fill_between(xs, np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.33)

plt.ylim(**ylims)

ax.set_ylabel('Median Reward')
ax.set_xlabel('Steps')
plt.xticks(tick_setup[0], tick_setup[1])
# ax.legend(loc='center right')

fig.tight_layout()

plt.savefig(f'{models_dir}/breakout-baseline.pdf')
plt.show()

logger.info('Done.')

