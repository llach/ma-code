import tensorflow as tf

from baselines.common.cmd_util import make_vec_env
from baselines.common.tf_util import get_session
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.run import main

k = 4


for rlc in [1, 10]:
    vae_params = {
        'k': k,
        'latent_dim': 20,
        'beta': 1,
    }

    def build_pend_env(args, **kwargs):
        alg = args.alg
        seed = args.seed

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(args.env, 'atari', args.num_env or 1, seed, reward_scale=args.reward_scale,
                           flatten_dict_observations=flatten_dict_observations)
        return VecFrameStack(env, k, norm_frac=255)


    args = [
        '--env', 'BreakoutNoFrameskip-v4',
        '--num_timesteps', '1e7',
        '--alg', 'ppo2',
        '--network', 'mlp',
        '--log_interval', '2',
        '--nminibatches', '2',
        '--noptepochs', '1',
        '--nsteps', '32',
        '--num_env', '2',
        '--v_net', 'atari',
        '--seed', str(0),
        '--k', str(k),
        '--rl_coef', str(rlc),
        '--tensorboard', 'True',
        '--log_weights', 'True',
    ]

    main(args, build_fn=build_pend_env, vae_params=vae_params)
    s = get_session()
    s.close()
    tf.reset_default_graph()
print('done')
