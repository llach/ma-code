import tensorflow as tf

from baselines.common.tf_util import get_session
from baselines.run import main

args = [
    '--env', 'BreakoutNoFrameskip-v4',
    '--alg', 'ppo2',
    '--num_env', '16',
    '--log_interval', '1',
    '--tensorboard', 'True',
    '--seed', '0',
]

main(args)

get_session().close()
tf.reset_default_graph()

print('done')
