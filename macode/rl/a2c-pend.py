from keras.backend import clear_session
from baselines.run import main
from baselines.common.tf_util import get_session
import tensorflow as tf


for n, zahl in enumerate(['10']*4):
    args = [
        '--num_timesteps', '10e6',
        '--env', 'PendulumThetaStack-v0',
        '--alg', 'a2c',
        '--network', 'mlp',
        '--seed', str(n),
        '--num_env', zahl,
    ]
    main(args)

    print('done')
    s = get_session()
    s.close()
    clear_session()
    tf.reset_default_graph()
    print('clear')