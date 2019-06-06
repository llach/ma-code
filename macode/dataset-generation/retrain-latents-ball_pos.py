import os
import numpy as np
import tensorflow as tf

from baselines.run import main

from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

from forkan.common.utils import ball_pos_from_rgb
from gym.envs.classic_control import rendering
from forkan import dataset_path
from forkan.models import VAE

home = os.environ['HOME']
policy_path =  f'{home}/.forkan/done/breakout/ppo2-scratch/'
pol = 'breakout-nenv16-rlc10000.0-k4-seed0-modelscratch-b1-2019-06-05T10:48'
policy_path += pol

vae_name = 'breakout-b1.0-lat20-lr0.0001-2019-04-20T18:00'

k = 4
lats = 20

vae_params = {
    'k': k,
    'latent_dim': lats,
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
    '--seed', str(0),
    '--k', str(k),
    '--load_path', policy_path,
    '--play', 'True',
]

v = VAE(load_from=vae_name, network='atari', with_opt=False, session=tf.Session())
model, env = main(args, build_fn=build_pend_env, vae_params=vae_params, just_return=True)
viewer = rendering.SimpleImageViewer()

obs = env.reset()
d = False

max_t = 25e3

t = 0
last_t = 0
max_t_ep = 2000
num_ep = 0

# buffers to be saved
org_buf = []
lat_buf = []
vae_lat_buf = []
pos_buf = []

# working buffers for each episode
raw_frames = []
preprocessed_frames = []

while True:
    img = env.render(mode='rgb_array')
    raw_frames.append(img)

    ball_pos, fimg = ball_pos_from_rgb(img)
    pos_buf.append(ball_pos)

    obs_slice = obs[0, ..., -1]
    preprocessed_frames.append(obs_slice)

    # viewer.imshow(np.concatenate((img, fimg), axis=1))

    action, _, _, _, mus, _ = model.step_code(obs)
    obs, _, done, _ = env.step(action)

    lat_buf.append(np.asarray(mus[-1], dtype=np.float32).copy())

    t += 1

    if (t-last_t)>=max_t_ep or done:
        last_t = t
        print(f'episode {num_ep} done at {t}; passing buffer into vae ...')

        mu_t, logv_t, z_t = v.encode_and_sample(np.asarray(preprocessed_frames, dtype=np.float32))
        preprocessed_frames = []

        org_buf.append(raw_frames.copy())
        raw_frames = []

        vae_lat_buf.append(mu_t.copy())

        if t > max_t:
            print(f'{t} > {max_t}. done collecting data samples.')
            break

        print('done, going on')

        env.reset()
        num_ep += 1
        done = False

vae_lat_buf = np.asarray(np.concatenate(vae_lat_buf, axis=0), dtype=np.float32)
org_buf = np.asarray(np.concatenate(org_buf, axis=0), dtype=np.float32)
lat_buf = np.asarray(np.concatenate(lat_buf, axis=0), dtype=np.float32)
pos_buf = np.asarray(pos_buf, dtype=np.float32)

print(f'sample size: {(org_buf.nbytes + lat_buf.nbytes + vae_lat_buf.nbytes + pos_buf.nbytes)/org_buf.shape[0]/1000/1000}MB')
print(f'dataset size: {(org_buf.nbytes + lat_buf.nbytes + vae_lat_buf.nbytes + pos_buf.nbytes)/1000/1000}MB')

print(org_buf.shape, lat_buf.shape, vae_lat_buf.shape, pos_buf.shape)

dataset_name = 'ball_latents_VAE_' + vae_name.split('-2019')[0] + '_POL_' + pol.split('-2019')[0] + '.npz'
print(f'saving dataset {dataset_name} ...')

np.savez_compressed(f'{dataset_path}/{dataset_name}',
                    originals=org_buf, latents=lat_buf, vae_latents=vae_lat_buf, ball_positions=pos_buf)

viewer.close()

print('dataset successfully generated!')