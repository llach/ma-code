from baselines.run import main
from baselines.common.cmd_util import make_vec_env
from forkan.rl import VecVAEStack

vae_name = 'pend-optimal'

def build_pend_env(args, **kwargs):
    alg = args.alg
    seed = args.seed

    flatten_dict_observations = alg not in {'her'}
    env = make_vec_env('PendulumVisual-v0', 'classic_control', args.num_env or 1, seed, reward_scale=args.reward_scale,
                             flatten_dict_observations=flatten_dict_observations)
    return VecVAEStack(env, k=3, load_from=vae_name)

args = [
    '--num_timesteps', '10e6',
    '--alg', 'a2c',
    '--network', 'mlp',
    '--num_env', '16',
]

main(args, build_fn=build_pend_env)
pass