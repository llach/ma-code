import argparse
import logging

from forkan.rl import make, DQN

dqn_conf = {
    'name': 'dqb-breakout',
    'network_type': 'nature-cnn',
    'buffer_size': 1e6,
    'total_timesteps': 5e7,
    'training_start': 5e4,
    'target_update_freq': 1e4,
    'exploration_fraction': 0.1,
    'gamma': .99,
    'batch_size': 32,
    'prioritized_replay': True,
    'double_q': True,
    'dueling': True,
    'clean_tensorboard_runs': True,
    'clean_previous_weights': True,
}

# environment parameters
env_conf = {
    'id': 'Breakout-v0',
    'frameskip': 4,
    'num_frames': 4,
    'obs_type': 'image',
    'target_shape': (110, 84),
    'crop_ranges': [(17, 103), (4, 80)]
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
