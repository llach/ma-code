import tensorflow as tf

from baselines.common.cmd_util import make_vec_env
from baselines.common.tf_util import get_session
from baselines.run import main
from forkan.rl import VecVAEStack

k = 5

vae_names = [
    'pendvisualuniform-b1-lat5-lr0.001-2019-04-08T22:04',
    'pendvisualuniform-b81.0-lat5-lr0.001-2019-04-04T15/08',
    'pendvisualuniform-b85.63-lat5-lr0.001-2019-04-06T02/14',
]

for seed in [1, 2, 3, 4, 5]:
    for vae_name in vae_names:
        def build_pend_env(args, **kwargs):
            alg = args.alg
            seed = args.seed

            flatten_dict_observations = alg not in {'her'}
            env = make_vec_env(args.env, 'classic_control', args.num_env or 1, seed, reward_scale=args.reward_scale,
                                     flatten_dict_observations=flatten_dict_observations)
            return VecVAEStack(env, k=k, load_from=vae_name.replace('/', ':'))

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
            '--k', str(k),
            '--tensorboard', 'True',
        ]

        main(args, build_fn=build_pend_env)
        s = get_session()
        s.close()
        tf.reset_default_graph()
        pass