from keras.backend import clear_session
from baselines.common.cmd_util import make_vec_env
from baselines.common.tf_util import get_session
import tensorflow as tf
from baselines.run import main

def build_pend_env(args, **kwargs):
    alg = args.alg
    seed = args.seed

    flatten_dict_observations = alg not in {'her'}
    return make_vec_env(args.env, 'classic_control', args.num_env or 1, seed, reward_scale=args.reward_scale,
                             flatten_dict_observations=flatten_dict_observations)


args = [
    '--env', 'PendulumTheta-v0',
    '--num_timesteps', '8e6',
    '--alg', 'ppo2',
    '--network', 'mlp',
    '--nminibatches', '32',
    '--noptepochs', '10',
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
print('done')