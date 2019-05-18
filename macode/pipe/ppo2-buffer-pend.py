import tensorflow as tf

from baselines.common.cmd_util import make_vec_env
from baselines.common.tf_util import get_session
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.run import main

k = 5

vae_params = {
    'k': k,
    'latent_dim': 5,
    'beta': 1,
}

def build_pend_env(args, **kwargs):
    alg = args.alg
    seed = args.seed

    flatten_dict_observations = alg not in {'her'}
    env = make_vec_env(args.env, 'classic_control', args.num_env or 1, seed, reward_scale=args.reward_scale,
                       flatten_dict_observations=flatten_dict_observations)
    return VecFrameStack(env, k)

args = [
    '--env', 'PendulumVisual-v0',
    '--num_timesteps', '10e6',
    '--alg', 'ppo_buffer',
    '--network', 'mlp',
    '--log_interval', '2',
    '--nminibatches', '32',
    '--noptepochs', '10',
    '--num_env', '16',
    '--seed', '0',
    '--tensorboard', 'True',
    '--rl_coef', str(10),
    '--k', str(k),
    '--log_interval', '1',
]

main(args, build_fn=build_pend_env, vae_params=vae_params)
s = get_session()
s.close()
tf.reset_default_graph()
print('done')

for rl_coef in [30]:
    for es in ['True', 'False']:
        vae_params = {
            'k': k,
            'latent_dim': 5,
            'beta': 1,
        }

        def build_pend_env(args, **kwargs):
            alg = args.alg
            seed = args.seed

            flatten_dict_observations = alg not in {'her'}
            env = make_vec_env(args.env, 'classic_control', args.num_env or 1, seed, reward_scale=args.reward_scale,
                               flatten_dict_observations=flatten_dict_observations)
            return VecFrameStack(env, k)

        args = [
            '--env', 'PendulumVisual-v0',
            '--num_timesteps', '10e6',
            '--alg', 'ppo_buffer',
            '--network', 'mlp',
            '--log_interval', '2',
            '--nminibatches', '32',
            '--noptepochs', '10',
            '--num_env', '16',
            '--seed', '0',
            '--tensorboard', 'True',
            '--collect_until', '10e6',
            '--rl_coef', str(rl_coef),
            '--k', str(k),
            '--log_interval', '1',
            '--target_kl', '0.01',
            '--early_stop', 'True',
        ]

        main(args, build_fn=build_pend_env, vae_params=vae_params)
        s = get_session()
        s.close()
        tf.reset_default_graph()
        print('done')
