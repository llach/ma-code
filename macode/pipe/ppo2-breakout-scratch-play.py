import numpy as np
from scipy.misc import imresize

from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.run import main
from gym.envs.classic_control import rendering

path = '/Users/llach/.forkan/done/breakout/ppo2-scratch/'

runs = [
    'breakout-nenv16-rlc1-k4-seed0-modelscratch-b1-2019-04-27T11/17'.replace('/', ':'),
    'breakout-nenv16-rlc10-k4-seed0-modelscratch-b1-2019-04-28T11/13'.replace('/', ':'),
]

k = 4

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
    return VecFrameStack(env, k)


args = [
    '--env', 'BreakoutNoFrameskip-v4',
    '--alg', 'ppo2',
    '--network', 'mlp',
    '--v_net', 'atari',
    '--seed', str(0),
    '--k', str(k),
    '--load_path', f'{path}{runs[0]}',
    '--play', 'True',
]

model, env = main(args, build_fn=build_pend_env, vae_params=vae_params, just_return=True)

obs = env.reset()
d = False

print('playing policy')

viewer = rendering.SimpleImageViewer()

t = 0

while t < 2000:
# while not d:

    (actions, xhat), x = model.step_xhat(obs)

    img = imresize(np.squeeze(x)[-1], (230, 160))
    rec = imresize(np.squeeze(xhat)[-1], (230, 160))

    shw = np.stack((np.concatenate((img, rec*255), axis=1),)*3, axis=-1)
    viewer.imshow(np.asarray(shw, dtype=np.uint8))

    obs, _, done, _ = env.step(actions)
    d = done.any() if isinstance(done, np.ndarray) else done
    t += 1
    if d:
        obs = env.reset()
