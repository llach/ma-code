import tensorflow as tf

from baselines.common.cmd_util import make_vec_env
from baselines.common.tf_util import get_session
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.run import main

k = 5

def build_pend_env(args, **kwargs):
    alg = args.alg
    seed = args.seed

    flatten_dict_observations = alg not in {'her'}
    env = make_vec_env(args.env, 'classic_control', args.num_env or 1, seed, reward_scale=args.reward_scale,
                       flatten_dict_observations=flatten_dict_observations)
    return VecFrameStack(env, k)


for seed in [0, 1, 2]:
    vae_params = {
        'init_from': 'pendvisualuniform-b1-lat5-lr0.001-2019-04-08T22/04'.replace('/', ':'),
        'k': k,
        'latent_dim': 5,
        'with_attrs': True,
    }

    args = [
        '--env', 'PendulumVisual-v0',
        '--num_timesteps', '10e6',
        '--alg', 'ppo2',
        '--network', 'mlp',
        '--log_interval', '2',
        '--nminibatches', '32',
        '--noptepochs', '10',
        '--num_env', '16',
        '--seed', str(seed),
        '--tensorboard', 'True',
        '--k', str(k),
        '--log_weights', 'True',
    ]

    main(args, build_fn=build_pend_env, vae_params=None)
    s = get_session()
    s.close()
    tf.reset_default_graph()
    print('done')