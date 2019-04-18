import tensorflow as tf

from baselines.common.tf_util import get_session
from baselines.run import main

args = [
    '--env', 'BreakoutNoFrameskip-v4',
    '--alg', 'ppo2',
    '--num_timesteps', '10e7',
    '--num_env', '1',
    '--log_interval', '1',
    '--seed', '0',
    '--load_path', '/Users/llach/breakout-nenv16-rlc1.0-seed0-modeld-2019-04-17T20:39/best/',
    '--play', 'True',
]

main(args)

get_session().close()
tf.reset_default_graph()

print('done')