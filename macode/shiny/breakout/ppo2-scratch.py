import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from forkan.common.utils import read_keys, setup_plotting, get_figure_size


logger = logging.getLogger(__name__)

ylims, tick_setup = setup_plotting('break-baseline')
fig, ax = plt.subplots(1, 1, figsize=get_figure_size())

home = os.environ['HOME']
models_dir = f'{home}/.forkan/done/breakout/ppo2-scratch-clean'

col = ['#d4a6e8', '#ccb974', '#64b5cd']
i = 0
for fi, name in [('rlc1-', '$\kappa=1$'), ('rlc10-', '$\kappa=10$'), ('rlc20-', '$\kappa=20$')]:
    data = read_keys(models_dir, fi, ['mean_reward', 'total_timesteps'])

    xs = data['total_timesteps'][0]
    ys = data['mean_reward']

    plt.plot(xs, np.nanmedian(ys, axis=0), label=name, color=col[i])
    plt.fill_between(xs, np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.33, color=col[i])
    i += 1


plt.ylim(bottom=-1, top=50)

plt.ylim(**ylims)

ax.set_ylabel('Median Reward')
ax.set_xlabel('Steps')
plt.xticks(tick_setup[0], tick_setup[1])
ax.legend(loc='upper left')

fig.tight_layout()

plt.savefig(f'{home}/.forkan/done/breakout/figures/kappa-nostop.pdf')
plt.show()

logger.info('Done.')


