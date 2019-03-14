import time
import numpy as np
from gym import make

import scipy.misc

from forkan import dataset_path
from forkan.common.utils import create_dir

FRAMES = 30000

env = make('PendulumVisual-v0')
o = env.reset()


# Start total timer
tstart = time.time()


frames = np.zeros([FRAMES, 64, 64, 1])

for i in range(FRAMES):
    action = env.action_space.sample()
    o, reward, done, info = env.step(action)

    # Calculate the fps (frame per second)
    nseconds = time.time() - tstart
    fps = int((i+1) / nseconds)

    frames[i] = o

    if done:
        o = env.reset()

    print(i, fps)


print('dumping file')
np.savez_compressed('{}/pendulum-visual-random-normalized-cut.npz'.format(dataset_path), data=frames)

print('storing some pngs')
create_dir('{}/pendulum-visual/'.format(dataset_path))
for n, f in enumerate(frames[40:60, ...]):
    scipy.misc.imsave('{}/pendulum-visual/frame{}.png'.format(dataset_path, n), np.squeeze(f))
print('done')
