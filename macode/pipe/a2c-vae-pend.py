from baselines.run import main
from baselines.common.cmd_util import make_dummy_vec_env

from forkan.rl import PendulumRenderEnv, PendulumVAEStackEnv


def build_pend_env(args, **kwargs):
    alg = args.alg
    seed = args.seed

    def wrap_fn(env):
        env = PendulumRenderEnv(env)
        env = PendulumVAEStackEnv(env, k=4)
        return env

    flatten_dict_observations = alg not in {'her'}
    return make_dummy_vec_env('Pendulum-v0', 'classic_control', args.num_env or 1, seed, reward_scale=args.reward_scale,
                             flatten_dict_observations=flatten_dict_observations, wrapper_fn=wrap_fn)


args = [
    '--num_timesteps', '10e6',
    '--env', 'PendulumVAEStack-v0',
    '--alg', 'a2c',
    '--network', 'mlp',
    '--num_env', '2',
    # '--vae', 'pend-optimal',
    # '--load_path', '/Users/llach/.forkan/models/a2c/pendulum-noVAE-nenv16-2019-03-07T22:11/weights_latest',
    # '--play'
]

main(args, build_fn=build_pend_env)
pass