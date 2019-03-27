from keras.backend import clear_session
from baselines.run import main
from baselines.common.tf_util import get_session
import tensorflow as tf


args = [
    '--num_timesteps', '2e6',
    '--env', 'Pendulum-v0',
    '--alg', 'a2c',
    '--network', 'mlp',
    '--seed', '1',
    '--num_env', str(6),
    '--tensorboard', 'True',
]
main(args)

print('done')
s = get_session()
s.close()
clear_session()
tf.reset_default_graph()
print('clear')