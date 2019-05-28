import tensorflow as tf

from baselines.common.cmd_util import make_vec_env
from baselines.common.tf_util import get_session
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.run import main

k = 4

def build_pend_env(args, **kwargs):
    alg = args.alg
    seed = args.seed

    flatten_dict_observations = alg not in {'her'}
    env = make_vec_env(args.env, 'atari', args.num_env or 1, seed, reward_scale=args.reward_scale,
                       flatten_dict_observations=flatten_dict_observations)
    return VecFrameStack(env, k)


for seed in [0, 1, 2]:
    vae_params = {
        'init_from': 'breakout-b1.0-lat20-lr0.0001-2019-04-20T18/00'.replace('/', ':'),
        'k': k,
        'latent_dim': 20,
        'with_attrs': True,
    }

    args = [
        '--env', 'BreakoutNoFrameskip-v4',
        '--num_timesteps', '1e7',
        '--alg', 'ppo2',
        '--network', 'mlp',
        '--log_interval', '2',
        '--nminibatches', '32',
        '--noptepochs', '10',
        '--num_env', '16',
        '--v_net', 'atari',
        '--seed', str(seed),
        '--k', str(k),
        '--tensorboard', 'True',
        '--log_weights', 'True',
    ]

    main(args, build_fn=build_pend_env, vae_params=vae_params)
    s = get_session()
    s.close()
    tf.reset_default_graph()
    print('done')