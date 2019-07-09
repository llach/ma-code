import tensorflow as tf

from baselines.common.tf_util import get_session
from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.run import main

k = 5

for seed in [0, 1, 2]:
    for lat in [512, 5, 16, 64, 128]:
        def build_pend_env(args, **kwargs):
            alg = args.alg
            seed = args.seed

            flatten_dict_observations = alg not in {'her'}
            env = make_vec_env(args.env, 'classic_control', args.num_env or 1, seed, reward_scale=args.reward_scale,
                               flatten_dict_observations=flatten_dict_observations)
            return VecFrameStack(env, k)


        args = [
            '--env', 'PendulumVisual-v0',
            '--num_timesteps', '1e7',
            '--alg', 'ppo2',
            '--latents', str(lat),
            '--network', 'cnn_pend_shared',
            '--nminibatches', '32',
            '--noptepochs', '10',
            '--num_env', '16',
            '--log_interval', '2',
            '--seed', str(seed),
            '--tensorboard', 'True',
        ]

        main(args, build_fn=build_pend_env)
        s = get_session()
        s.close()
        tf.reset_default_graph()
