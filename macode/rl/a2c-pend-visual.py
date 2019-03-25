from keras.backend import clear_session
from baselines.run import main
from baselines.common.tf_util import get_session
from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
import tensorflow as tf

def build_pend_env(args, **kwargs):
    alg = args.alg
    seed = args.seed

    flatten_dict_observations = alg not in {'her'}
    venv = make_vec_env(args.env, 'classic_control', args.num_env or 1, seed, reward_scale=args.reward_scale,
                        flatten_dict_observations=flatten_dict_observations)
    return VecFrameStack(venv, 4)


for zahl in [8, 12, 16]:
    args = [
        '--num_timesteps', '30e6',
        '--env', 'PendulumVisual-v0',
        '--alg', 'a2c',
        '--network', 'cnn',
        '--seed', '1',
        '--num_env', str(zahl),
    ]
    main(args, build_fn=build_pend_env)

    print('done')
    s = get_session()
    s.close()
    clear_session()
    tf.reset_default_graph()
    print('clear')