import numpy as np
from scipy.misc import imresize

from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.run import main
from gym.envs.classic_control import rendering

path = '/Users/llach/.forkan/done/breakout/ppo2-scratch/'

runs = [
    'breakout-nenv16-rlc0.5-k4-seed0-modelscratch-b1-2019-05-13T08:35'.replace('/', ':'),
    'breakout-nenv16-rlc1-k4-seed0-modelscratch-b1-2019-04-27T11/17'.replace('/', ':'),
    'breakout-nenv16-rlc10-k4-seed0-modelscratch-b1-2019-04-28T11/13'.replace('/', ':'),
]

k = 4

vae_params = {
    'k': k,
    'latent_dim': 20,
    'beta': 1,
    'with_attrs': True,
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
# exit(0)
obs = env.reset()
d = False

print('playing policy')

# viewer = rendering.SimpleImageViewer()

t = 0

import matplotlib.pyplot as plt

def smooth(l):
    l = np.squeeze(np.asarray(l, np.float32))
    N = 40
    return np.convolve(l, np.ones((N,)) / N, mode='valid')
    # return medfilt(np.asarray(l, dtype=np.float32), 51)

vals, recs, dkls, vls = [], [], [], []

num_ep = 0
m_rec = None

img_buffer = []

while num_ep < 3:

    (actions, xhat, vf, dkl, recon, vl), x = model.step_xhat(obs)

    recs.append(np.mean(recon))

    img = imresize(np.squeeze(x)[-1], (230, 160))
    rec = imresize(np.squeeze(xhat)[-1], (230, 160))

    vals.append(vf)
    dkls.append(np.sum(dkl))
    vls.append(vl)


    shw = np.stack((np.concatenate((img, rec), axis=1),)*3, axis=-1)
    # viewer.imshow(np.asarray(shw, dtype=np.uint8))
    img_buffer.append(shw)

    obs, _, done, _ = env.step(actions)
    d = done.any() if isinstance(done, np.ndarray) else done
    t += 1
    if d:
        obs = env.reset()
        m_rec = np.median(recs)

        # for (p, n) in [(vals, f'vals{num_ep}'), (dkls, f'dkls{num_ep}'),  (vls, f'VAE loss {num_ep}')]:
        #     p = p[5:]
        #     plt.plot(p)
        #     plt.title(n)
        #     plt.show()
        #     p = []

        plt.plot(smooth(recs))
        plt.title(f'reconstruction loss {num_ep}')
        plt.axhline(y=m_rec)
        plt.show()

        # img_buffer = []
        # recs = []

        num_ep += 1
        print(f'beginning episode {num_ep} at {t}')

recs = recs[5:]
# recs = smooth(recs)

img_buffer = np.asarray(img_buffer, dtype=np.uint8)[5:]
greater = img_buffer[recs > m_rec][-50:]

for i in range(greater.shape[0]):
    plt.imshow(greater[i])
    plt.show()
# viewer.close()
