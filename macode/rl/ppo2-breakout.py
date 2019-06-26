import tensorflow as tf

from baselines.common.tf_util import get_session
from baselines.run import main

for seed in [0, 1, 2]:
    args = [
        '--env', 'BreakoutNoFrameskip-v4',
        '--num_timesteps', '1e7',
        '--alg', 'ppo2',
        '--num_env', '16',
        '--nminibatches', '32',
        '--noptepochs', '10',
        '--log_interval', '1',
        '--tensorboard', 'True',
        '--seed', str(seed),
    ]

    main(args)

    get_session().close()
    tf.reset_default_graph()

print('done')
