from keras.backend import clear_session
from baselines.common.tf_util import get_session
import tensorflow as tf
from baselines.run import main
from baselines.common.cmd_util import make_vec_env
from forkan.rl import VecVAEStack


for nenvs in [4]:
    def build_pend_env(args, **kwargs):
        alg = args.alg
        seed = args.seed

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(args.env, 'classic_control', args.num_env or 1, seed, reward_scale=args.reward_scale,
                                 flatten_dict_observations=flatten_dict_observations)
        return VecVAEStack(env, k=3, load_from='pendvisualuniform-b77.5-lat5-lr0.001-2019-03-21T00/13'.replace('/', ':'))

    args = [
        '--env', 'PendulumVisual-v0',
        '--num_timesteps', '10e6',
        '--alg', 'a2c',
        '--network', 'mlp',
        '--num_env', str(nenvs),
        '--seed', '1',
        '--tensorboard', 'True',
    ]

    main(args, build_fn=build_pend_env)
    s = get_session()
    s.close()
    clear_session()
    tf.reset_default_graph()
    pass