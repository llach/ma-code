import argparse
import logging

from forkan.rl import make, A2C

# algorithm parameters
a2c_conf = {
    'name': 'a2c-pong',
    'policy_type': 'mnih-2013',
    'total_timesteps': 1e6,
    'lr': 5e-4,
    'gamma': .95,
    'entropy_coef': 0.01,
    'v_loss_coef': 0.2,
    'gradient_clipping': None,
    'reward_clipping': None,
    'use_tensorboard': True,
    'clean_tensorboard_runs': True,
    'clean_previous_weights': True,
    'print_freq': 10,
}

# environment parameters
env_conf = {
    'id': 'Pong-v0',
    'frameskip': 4,
    'num_frames': 4,
    'num_envs': 16,
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
    env_conf.pop('num_envs')
    a2c_conf['clean_previous_weights'] = False

e = make(**env_conf)
alg = A2C(e, **a2c_conf)

if args.run:
    log.info('Running a2c on {}'.format(env_conf['id']))
    alg.run()
else:
    log.info('Learning with a2c on {}'.format(env_conf['id']))
    alg.learn()
