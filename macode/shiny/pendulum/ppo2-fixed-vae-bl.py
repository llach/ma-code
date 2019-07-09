import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from forkan.common.utils import read_keys, setup_plotting, get_figure_size


logger = logging.getLogger(__name__)

ylims, tick_setup = setup_plotting()
fig, ax = plt.subplots(1, 1, figsize=get_figure_size())

home = os.environ['HOME']
models_dir = f'{home}/.forkan/done/pendulum/ppo2-gt'

for fi, name in [('', 'with $\omega_t$')]:
    data = read_keys(models_dir, fi, ['mean_reward', 'total_timesteps'])

    xs = data['total_timesteps'][0]
    ys = data['mean_reward']

    ax.plot(xs, np.nanmedian(ys, axis=0), label=name)
    ax.fill_between(xs, np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.33)

models_dir = f'{home}/.forkan/done/pendulum/ppo2-gt-theta'
for fi, name in [('', 'without $\omega_t$')]:
    data = read_keys(models_dir, fi, ['mean_reward', 'total_timesteps'])

    xs = data['total_timesteps'][0]
    ys = data['mean_reward']

    ax.plot(xs, np.nanmedian(ys, axis=0), label=name)
    ax.fill_between(xs, np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.33)

models_dir = f'{home}/.forkan/done/pendulum/ppo2-fixed-vae'

for fi, name in [('b1', '$VAE^{ent}$'), ('b81', '$VAE^{one}$'), ('b85', '$VAE^{two}$')]:
    data = read_keys(models_dir, fi, ['mean_reward', 'total_timesteps'])

    xs = data['total_timesteps'][0]
    ys = data['mean_reward']

    ax.plot(xs, np.nanmedian(ys, axis=0), label=name)
    ax.fill_between(xs, np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.33)

plt.ylim(**ylims)

ax.set_ylabel('Median Reward')
ax.set_xlabel('Steps')
plt.xticks(tick_setup[0], tick_setup[1])
ax.legend(loc='lower right')

fig.tight_layout()

plt.savefig(f'{home}/.forkan/done/pendulum/ppo2-fixed-vae/pend-fixed-bl.pdf')
plt.show()

logger.info('Done.')