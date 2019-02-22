import argparse
import logging

from forkan.rl import make, DQN


# mark env as solved once rewards were higher than 195 50 episodes in a row
def solved_callback(rewards):
    if len(rewards) < 50:
        return False

    for r in rewards[-50:]:
        if r < 195:
            return False

    return True


# algorithm parameters
dqn_conf = {
    'name': 'cart-dqn',
    'total_timesteps': 1e5,
    'lr': 5e-4,
    'gamma': .95,
    'entropy_coef': 0.01,
    'v_loss_coef': 0.2,
    'gradient_clipping': None,
    'reward_clipping': None,
    'use_tensorboard': True,
    'clean_tensorboard_runs': True,
    'clean_previous_weights': True,
    'print_freq': 20,
    'solved_callback': solved_callback,
}

# environment parameters
env_conf = {
    'id': 'CartPole-v0',
}

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--run', '-r', action='store_true')
args = parser.parse_args()

log = logging.getLogger(__name__)

# remove keys from config so that the correct environment will be created
if args.run:
    dqn_conf['clean_previous_weights'] = False

e = make(**env_conf)
alg = DQN(e, **dqn_conf)

if args.run:
    log.info('Running dqn on {}'.format(env_conf['id']))
    alg.run()
else:
    log.info('Learning with dqn on {}'.format(env_conf['id']))
    alg.learn()
