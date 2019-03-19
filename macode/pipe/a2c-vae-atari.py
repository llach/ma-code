from baselines.run import main
from baselines.common.cmd_util import make_vec_env
from forkan.rl import VecVAEStack

vae_name = 'gopher'

def build_pend_env(args, **kwargs):
    alg = args.alg
    seed = args.seed

    flatten_dict_observations = alg not in {'her'}
    env = make_vec_env(args.env, 'atari', args.num_env or 1, seed, reward_scale=args.reward_scale,
                             flatten_dict_observations=flatten_dict_observations)
    return VecVAEStack(env, k=3, load_from=vae_name)


args = [
    '--num_timesteps', '50e6',
    '--env', 'GopherNoFrameskip-v4',
    '--alg', 'a2c',
    '--num_env', '16',
    '--network', 'mlp',
]

main(args)
pass