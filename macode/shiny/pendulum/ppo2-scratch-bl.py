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

models_dir = f'{home}/.forkan/done/pendulum/ppo2-scratch-pend-clean'

# for fi, name in [('rlc1-k5-seed0', 'kappa=1'), ('rlc10-k5-seed0', 'kappa=10'), ('rlc30-k5-seed0', 'kappa=30')]:
#     data = read_keys(models_dir, fi, ['mean_reward', 'nupdates'])
#
#     xs = data['nupdates'][0]
#     ys = data['mean_reward']
#
#     plt.plot(xs, np.nanmedian(ys, axis=0), label=name)
#     plt.fill_between(xs, np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.33)
#
#
# plt.ylim(bottom=-1300, top=-100)
#
# plt.title('Training from scratch with different kappa')
# plt.ylabel('Median Reward')
# plt.xlabel('Number of Updates')
#
# plt.legend()
#
# plt.savefig(f'{home}/.forkan/done/ppo2-scratch/kappa-nostop.pdf')
# plt.show()

# logger.info('second now --------------------------')

for fi, name in [('rlc1-k5-seed0', '$\\kappa=1$'), ('rlc10-k5-seed0', '$\\kappa=10$'), ('rlc30-k5-seed0', '$\\kappa=30$')]:
    fi = fi.replace('seed0', 'stop')
    data = read_keys(models_dir, fi, ['mean_reward', 'total_timesteps'])

    xs = data['total_timesteps'][0]
    ys = data['mean_reward']

    plt.plot(xs, np.nanmedian(ys, axis=0), label=name)
    plt.fill_between(xs, np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.33)

plt.ylim(**ylims)

ax.set_ylabel('Median Reward')
ax.set_xlabel('Steps')
plt.xticks(tick_setup[0], tick_setup[1])
ax.legend(loc='lower right')

fig.tight_layout()

plt.savefig(f'{home}/.forkan/done/pendulum/ppo2-scratch-pend-clean/scratch-bl.pdf')
plt.show()

logger.info('Done.')


