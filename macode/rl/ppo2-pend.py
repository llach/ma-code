from keras.backend import clear_session
from baselines.common.tf_util import get_session
import tensorflow as tf
from baselines.run import main


args = [
    '--env', 'PendulumTheta-v0',
    '--num_timesteps', '10e6',
    '--alg', 'ppo2',
    '--network', 'mlp',
    '--nminibatches', '16',
    '--noptepochs', '8',
    '--num_env', '8',
    '--log_interval', '2',
    '--seed', '1',
    '--tensorboard', 'True',
]

main(args)
s = get_session()
s.close()
clear_session()
tf.reset_default_graph()
pass