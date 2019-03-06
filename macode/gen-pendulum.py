import time
import numpy as np
from gym import make

import scipy.misc

from forkan import dataset_path
from forkan.common.utils import create_dir

FRAMES = 100000

env = make('Pendulum-v0')
o = env.reset()


# Start total timer
tstart = time.time()


def transform_pendulum_obs(obs):
    obs = obs / 255 # normalise
    obs = obs[121:256 + 121, 121:256 + 121, :] # cut out interesting area
    obs = np.dot(obs[..., :3], [0.299, 0.587, 0.114]) # inverted greyscale
    return obs


frames = np.zeros([FRAMES, 256, 256])

for i in range(FRAMES):
    action = env.action_space.sample()
    o, reward, done, info = env.step(action)

    arr = env.render(mode='rgb_array')

    # Calculate the fps (frame per second)
    nseconds = time.time() - tstart
    fps = int((i+1) / nseconds)

    frames[i] = transform_pendulum_obs(arr)

    if done:
        o = env.reset()

    print(i, fps)


print('dumping file')
np.savez_compressed('{}/pendulum-random-normalized-cut.npz'.format(dataset_path), data=frames)

print('storing some pngs')
create_dir('{}/pendulum/'.format(dataset_path))
for n, f in enumerate(frames[40:60, ...]):
    print(f)
    scipy.misc.imsave('{}/pendulum/frame{}.png'.format(dataset_path, n), np.squeeze(f))
print('done')
