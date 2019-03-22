from baselines.run import main
from baselines.common.cmd_util import make_env
from forkan.rl import VAEStack

# vae_name = 'pendvisualuniform-b22-lat5-lr0.001-2019-03-18T20/23'.replace('/', ':')

def build_pend_env(args, **kwargs):
    seed = args.seed

    env = make_env(args.env, 'classic_control', seed=seed)
    return VAEStack(env, k=3)

args = [
    '--num_timesteps', '50e6',
    '--env', 'PendulumVisual-v0',
    '--alg', 'trpo_mpi',
    '--network', 'mlp',
]

main(args, build_fn=build_pend_env)
pass