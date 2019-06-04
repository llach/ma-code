import os
import numpy as np
import tensorflow as tf

from baselines.run import main

from forkan.common.utils import ball_pos_from_rgb
from gym.envs.classic_control import rendering
from forkan.models import RetrainVAE
from forkan import dataset_path

home = os.environ['HOME']
policy_path = f'{home}/.forkan/done/stock-models/breakout-ppo/'
vae_path = f'{home}/.forkan/done/breakout/ppo2-scratch/'
vae_name = 'breakout-nenv16-rlc10000-k4-seed0-modelscratch-b1-2019-05-28T21:30/'
vae_path += vae_name

k = 4

args = [
    '--env', 'BreakoutNoFrameskip-v4',
    '--alg', 'ppo2',
    '--num_env', '1',
    '--seed', str(0),
    '--k', str(k),
    '--load_path', policy_path,
    '--play', 'True',
]

model, env = main(args, just_return=True)
v = RetrainVAE(vae_path, (84, 84, 1), latent_dim=20, sess=tf.Session(), k=1, with_attrs=True, network='atari')
v.s.run(tf.global_variables_initializer())
v.load()

viewer = rendering.SimpleImageViewer()

obs = env.reset()
d = False

max_t = 2e4

t = 0
last_t = 0
max_t_ep = 2000
num_ep = 0

# buffers to be saved
org_buf = []
rec_buf = []
lat_buf = []
pos_buf = []

# working buffers for each episode
raw_frames = []
preprocessed_frames = []

while True:

    obs_slice = obs[0, ..., -1]

    img = env.render(mode='rgb_array')
    ball_pos, fimg = ball_pos_from_rgb(img)

    raw_frames.append(img)
    pos_buf.append(ball_pos)
    preprocessed_frames.append(obs_slice/255)

    # viewer.imshow(np.concatenate((img, fimg), axis=1))

    action, _, _, _ = model.step(obs)
    obs, _, done, _ = env.step(action)

    t += 1

    if (t-last_t)>=max_t_ep or done:
        last_t = t
        print(f'episode {num_ep} done at {t}; passing buffer into vae ...')

        mu_t, logv_t = v.encode(np.asarray(preprocessed_frames, dtype=np.float32))
        rec_buf.append(np.squeeze(v.decode(mu_t, logv_t)))

        org_buf.append(raw_frames.copy())
        lat_buf.append(mu_t.copy())

        raw_frames = []
        preprocessed_frames = []

        if t > max_t:
            print(f'{t} > {max_t}. done collecting data samples.')
            break

        print('done, going on')

        env.reset()
        num_ep += 1
        done = False

org_buf = np.asarray(np.concatenate(org_buf, axis=0), dtype=np.float32)
rec_buf = np.asarray(np.concatenate(rec_buf, axis=0), dtype=np.float32)
lat_buf = np.asarray(np.concatenate(lat_buf, axis=0), dtype=np.float32)
pos_buf = np.asarray(pos_buf, dtype=np.float32)

print(f'sample size: {(org_buf.nbytes + rec_buf.nbytes + lat_buf.nbytes + pos_buf.nbytes)/org_buf.shape[0]/1000/1000}MB')
print(f'dataset size: {(org_buf.nbytes + rec_buf.nbytes + lat_buf.nbytes + pos_buf.nbytes)/1000/1000}MB')

print(org_buf.shape, rec_buf.shape, lat_buf.shape, pos_buf.shape)

dataset_name = 'ball_latents_' + vae_name.split('-2019')[0] + '.npz'
print(f'saving dataset {dataset_name} ...')

np.savez_compressed(f'{dataset_path}/{dataset_name}',
                    originals=org_buf, reconstructions=rec_buf, latents=lat_buf, ball_positions=pos_buf)

viewer.close()

print('dataset successfully generated!')