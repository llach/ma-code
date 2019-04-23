from baselines.common.cmd_util import make_vec_env
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


args = [
    '--env', 'PendulumVisual-v0',
    '--num_timesteps', '50e6',
    '--alg', 'ppo2',
    '--network', 'cnn_pend',
    '--nminibatches', '32',
    '--noptepochs', '10',
    '--num_env', '8',
    '--log_interval', '2',
    '--seed', '0',
    '--tensorboard', 'True',
]

main(args, build_fn=build_pend_env)
