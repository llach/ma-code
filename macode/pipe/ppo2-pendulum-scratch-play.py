import numpy as np
from scipy.misc import imresize

from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.run import main
from gym.envs.classic_control import rendering

path = '/Users/llach/.forkan/done/pendulum/ppo2-scratch/'

runs = [
    'pendulumvisual-nenv16-rlc1-k5-stop0.01-seed0-modelscratch-b1-2019-04-23T10/42'.replace('/', ':'),
    'pendulumvisual-nenv16-rlc10-k5-stop0.01-seed0-modelscratch-b1-2019-04-24T13/58'.replace('/', ':'),
    'pendulumvisual-nenv16-rlc30-k5-stop0.01-seed0-modelscratch-b1-2019-04-26T02/18'.replace('/', ':'),
]

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
    '--alg', 'ppo2',
    '--network', 'mlp',
    '--v_net', 'pendulum',
    '--seed', str(0),
    '--k', str(k),
    '--load_path', f'{path}{runs[2]}',
    '--play', 'True',
]

model, env = main(args, build_fn=build_pend_env, vae_params=vae_params, just_return=True)

obs = env.reset()
d = False

print('playing policy')

viewer = rendering.SimpleImageViewer()

t = 0
import matplotlib.pyplot as plt

def smooth(l):
    l = np.squeeze(np.asarray(l, np.float32))
    N = 20
    return np.convolve(l, np.ones((N,)) / N, mode='valid')
    # return medfilt(np.asarray(l, dtype=np.float32), 51)


vals, recs, dkls, vls = [], [], [], []

num_ep = 0
while num_ep < 3:

    (actions, xhat, vf, dkl, recon, vl), x = model.step_xhat(obs)

    img = imresize(np.squeeze(x)[-1], (230, 160))
    rec = imresize(np.squeeze(xhat)[-1], (230, 160))

    vals.append(vf)
    recs.append(np.mean(recon))
    dkls.append(np.sum(dkl))
    vls.append(vl)

    print(len(recon), len(dkl))

    shw = np.stack((np.concatenate((img, rec), axis=1),)*3, axis=-1)
    viewer.imshow(np.asarray(shw, dtype=np.uint8))
    print((np.mean(recon)), np.mean(dkl))
    obs, _, done, _ = env.step(actions)
    d = done.any() if isinstance(done, np.ndarray) else done
    t += 1
    if d:
        obs = env.reset()

        for (p, n) in [(vals, f'vals{num_ep}'), (recs, f'recs{num_ep}'), (dkls, f'dkls{num_ep}'),  (vls, f'VAE loss {num_ep}')]:
            plt.plot(smooth(p))
            plt.title(n)
            plt.show()
            # p = []

        num_ep += 1
        print(f'beginning episode {num_ep} at {t}')