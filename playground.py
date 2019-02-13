from forkan.rl import make
import numpy as np
import matplotlib.pyplot as plt

# environment parameters
env_conf = {
    'id': 'Breakout-v0',
    'frameskip': 4,
    'num_frames': 4,
    'obs_type': 'image',
    'target_shape': (200, 160),
}

e = make(**env_conf)

obs = e.reset()

print(obs.shape)

for _ in range(20):
    obs = e.step(1)
    obs = obs[0]
    for i in range(obs.shape[-1]):

        plt.imshow(obs[..., i], cmap='Greys_r')
        plt.show()

