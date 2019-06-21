import numpy as np
from scipy.misc import imresize

from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.run import main
from baselines.common.tf_util import get_session
from gym.envs.classic_control import rendering

net_path = '/Users/llach/.forkan/done/classify-ball/RETRAIN-N128-ball_latents_VAE_breakout-b1.0-lat20-lr0.0001_POL_breakout-nenv16-rlc10000.0-k4-seed0-modelscratch-b1-2019-06-07T11:15/weights.h5'

path = '/Users/llach/.forkan/done/breakout/ppo2-scratch/'
# path = '/Users/llach/.forkan/models/ppo2/'

runs = [
    # 'breakout-nenv16-rlc1-k4-seed0-modelscratch-b1-2019-05-28T16:40',
    # 'breakout-nenv16-rlc10-k4-seed0-modelscratch-b1-2019-05-29T02:24',
    # 'breakout-nenv16-rlc1000-k4-seed0-modelscratch-b1-2019-05-29T07:19',
    # 'breakout-nenv16-rlc10000-k4-seed0-modelscratch-b1-2019-05-28T21:30',
    'breakout-nenv16-rlc10000.0-k4-seed0-modelscratch-b1-2019-06-05T10:48',
]

k = 4
mlp_neurons = 128

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

net = Sequential([
    Dense(mlp_neurons, activation='relu', input_shape=(20,)),
    Dense(mlp_neurons, activation='relu'),
    Dense(2, activation='sigmoid')])

net.compile(optimizer=tf.train.AdamOptimizer(0.01), loss='mse', metrics=['mae'])
net.load_weights(net_path)

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


def draw_predicted_balls(img, loc):

    for j in [-1, 0, 1]:
        for i in [-1, 0, 1]:
            x, y = np.clip(int((loc[0]*210)+j), 0, 209), np.clip(int((loc[1]*160)+i), 0, 159)
            img[x, y] = [0, 200, 200]

    return np.asarray(img, dtype=np.uint8)

args = [
    '--env', 'BreakoutNoFrameskip-v4',
    '--alg', 'ppo2',
    '--network', 'mlp',
    '--v_net', 'atari',
    '--seed', str(0),
    '--k', str(k),
    '--load_path', f'{path}{runs[-1]}',
    '--play', 'True',
]

model, env = main(args, build_fn=build_pend_env, vae_params=vae_params, just_return=True)
obs = env.reset()
d = False

print('playing policy')

viewer = rendering.SimpleImageViewer()

t = 0
num_ep = 0

while num_ep < 3:

    lob = np.expand_dims(np.squeeze(obs)[..., -1], -1)
    lobs = np.expand_dims(np.swapaxes(obs, -1, 0), 0)

    mus, logvars = model.vae.encode(lobs)
    mus = np.asarray(mus[-1], dtype=np.float32)
    locs = net.predict(mus)
    viewer.imshow(draw_predicted_balls(env.render(mode='rgb_array'), locs[0]))

    (actions, xhat, vf, dkl, recon, vl), x = model.step_xhat(obs)

    obs, _, done, _ = env.step(actions)
    d = done.any() if isinstance(done, np.ndarray) else done
    t += 1
    if d:
        num_ep += 1
        print(f'beginning episode {num_ep} at {t}')

viewer.close()