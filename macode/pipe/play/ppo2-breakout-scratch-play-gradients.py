import numpy as np
from scipy.misc import imresize

from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.run import main
from baselines.common.tf_util import get_session
from gym.envs.classic_control import rendering

path = '/Users/llach/.forkan/done/breakout/ppo2-scratch/'
# path = '/Users/llach/.forkan/models/ppo2/'

runs = [
    'breakout-nenv16-rlc1-k4-seed0-modelscratch-b1-2019-05-28T16:40',
    'breakout-nenv16-rlc10-k4-seed0-modelscratch-b1-2019-05-29T02:24',
    'breakout-nenv16-rlc1000-k4-seed0-modelscratch-b1-2019-05-29T07:19',
    'breakout-nenv16-rlc10000-k4-seed0-modelscratch-b1-2019-05-28T21:30',
    'breakout-nenv16-rlc10000.0-k4-seed0-modelscratch-b1-2019-06-05T10:48',
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
    return VecFrameStack(env, k, norm_frac=255)


args = [
    '--env', 'BreakoutNoFrameskip-v4',
    '--alg', 'ppo2',
    '--network', 'mlp',
    '--v_net', 'atari',
    # '--seed', str(0),
    '--k', str(k),
    '--load_path', f'{path}{runs[-1]}',
    '--play', 'True',
]

model, env = main(args, build_fn=build_pend_env, vae_params=vae_params, just_return=True)
# exit(0)
obs = env.reset()
d = False

print('playing policy')

viewer = rendering.SimpleImageViewer()

t = 0
num_ep = 0

import tensorflow as tf
vf_grad_op = tf.gradients(model.train_model.vf, model.vae.X)

s = get_session()

while num_ep < 3:

    lob = np.expand_dims(np.squeeze(obs)[..., -1], -1)

    vf_grad = s.run(vf_grad_op, feed_dict={model.vae.X: np.expand_dims(np.moveaxis(obs, -1, 1), -1)})
    vf_grad = np.squeeze(np.asarray(vf_grad, dtype=np.float32))[-1]

    # scale gradients to [0,1]
    vf_grad += np.abs(np.min(vf_grad))
    vf_grad *= (1 / (np.abs(np.min(vf_grad)) + np.max(vf_grad)))

    lob = np.repeat(lob, 3, axis=-1)
    lob[..., -1] += np.where(vf_grad > np.mean(vf_grad)*1.1, vf_grad, np.zeros_like(vf_grad))

    viewer.imshow(imresize(np.asarray(lob*255, dtype=np.uint8), (460, 320, 3)))

    (actions, xhat, vf, dkl, recon, vl), x = model.step_xhat(obs)

    obs, _, done, _ = env.step(actions)
    d = done.any() if isinstance(done, np.ndarray) else done
    t += 1
    if d:
        num_ep += 1
        print(f'beginning episode {num_ep} at {t}')

viewer.close()