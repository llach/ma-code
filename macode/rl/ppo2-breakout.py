import tensorflow as tf

from baselines.common.tf_util import get_session
from baselines.run import main

args = [
    '--env', 'BreakoutNoFrameskip-v4',
    '--alg', 'ppo2',
    '--num_env', '16',
    '--nminibatches', '16',
    '--noptepochs', '6',
    '--nsteps', '1024',
    '--log_interval', '1',
    '--seed', '0',
    '--tensorboard', 'True',
]

main(args)

get_session().close()
tf.reset_default_graph()

print('done')