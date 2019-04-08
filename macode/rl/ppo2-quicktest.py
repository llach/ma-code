import tensorflow as tf
from keras.backend import clear_session

from baselines.common.cmd_util import make_vec_env
from baselines.common.tf_util import get_session
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.run import main

nsteps = 10
k = 3

vae_params = {
    'init_from': 'pendvisualuniform-b85.63-lat5-lr0.001-2019-04-06T02:14'.replace('/', ':'),
    'k': k,
    'latent_dim': 5,
    'with_kl': True
}

def build_pend_env(args, **kwargs):
    alg = args.alg
    seed = args.seed

    flatten_dict_observations = alg not in {'her'}
    env = make_vec_env(args.env, 'classic_control', args.num_env or 1, seed, reward_scale=args.reward_scale,
                       flatten_dict_observations=flatten_dict_observations)
    return VecFrameStack(env, k)

args = [
    '--env', 'PendulumTest-v0',
    '--num_timesteps', '8e6',
    '--alg', 'ppo2',
    '--network', 'mlp',
    '--nsteps', '10',
    '--nminibatches', '2',
    '--noptepochs', '3',
    '--nsteps', '10',
    '--num_env', '8',
    '--log_interval', '2',
    '--seed', '0',
    '--tensorboard', 'True',
    '--k', str(k),
]

main(args, build_fn=build_pend_env, vae_params=vae_params)
s = get_session()
s.close()
clear_session()
tf.reset_default_graph()
print('done')