import tensorflow as tf

from baselines.common.cmd_util import make_vec_env
from baselines.common.tf_util import get_session
from baselines.run import main
from forkan.rl import VecVAEStack

k = 4

vae_names = [
    'breakout-rlc10-2019-04-28T11-13',
]

for seed in [0]:
    for vae_name in vae_names:
        def build_pend_env(args, **kwargs):
            alg = args.alg
            seed = args.seed

            flatten_dict_observations = alg not in {'her'}
            env = make_vec_env(args.env, 'atari', args.num_env or 1, seed, reward_scale=args.reward_scale,
                                     flatten_dict_observations=flatten_dict_observations)
            return VecVAEStack(env, k=k, load_from=vae_name.replace('/', ':'), vae_network='atari', norm_fac=(1/255))

        args = [
            '--env', 'BreakoutNoFrameskip-v4',
            '--num_timesteps', '1e7',
            '--alg', 'ppo2',
            '--network', 'mlp',
            '--log_interval', '2',
            '--nminibatches', '32',
            '--noptepochs', '10',
            '--num_env', '16',
            '--seed', str(seed),
            '--k', str(k),
            '--tensorboard', 'True',
            '--early_stop', 'True',
        ]

        main(args, build_fn=build_pend_env)
        s = get_session()
        s.close()
        tf.reset_default_graph()
        pass
