import tensorflow as tf

from baselines.common.tf_util import get_session
from baselines.run import main

args = [
    '--env', 'BreakoutNoFrameskip-v4',
    '--alg', 'ppo2',
    '--num_timesteps', '10e6',
    '--num_env', '16',
    '--log_interval', '1',
    '--seed', '0',
    '--tensorboard', 'True',
]

main(args)

get_session().close()
tf.reset_default_graph()

print('done')