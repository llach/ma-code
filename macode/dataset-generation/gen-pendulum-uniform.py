import numpy as np

import scipy.misc

from forkan import dataset_path
from forkan.common.utils import create_dir

import cairocffi as cairo


theta_res = 2000
reps = 15

frames = np.zeros([theta_res*reps, 64, 64, 1])

w , h = 64, 64
surf = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)

def _render_pendulum(th):
    cr = cairo.Context(surf)

    # draw background
    cr.set_source_rgb(1, 1, 1)
    cr.paint()

    # apply transforms
    cr.translate((w / 2), h / 2)
    cr.rotate(np.pi - th)

    # draw shapes that form the capsule
    cr.rectangle(-2.5, 0, 5, 27)
    cr.arc(0, 0, 2.5, 0, 2 * np.pi)
    cr.arc(0, (h / 2) - 4, 2.5, 0, 2 * np.pi)

    # draw color
    cr.set_source_rgb(.8, .3, .3)
    cr.fill()

    # center sphere
    cr.arc(0, 0, 1, 0, 2 * np.pi)
    cr.set_source_rgb(0, 0, 0)
    cr.fill()

    # reshape, delete fourth (alpha) channel, greyscale and normalise
    return np.expand_dims(np.dot(np.frombuffer(surf.get_data(), np.uint8).reshape([w, h, 4])[..., :3],
                                 [0.299, 0.587, 0.114]), -1) / 255

i = 0
for _ in range(reps):
    for theta in np.linspace(0, 2*np.pi, theta_res):
        frame = _render_pendulum(theta)
        frames[i] = frame
        i += 1


print('dumping file')
np.savez_compressed('{}/pendulum-visual-uniform.npz'.format(dataset_path), data=frames)

print('storing some pngs')
create_dir('{}/pendulum-uniform/'.format(dataset_path))
for n, f in enumerate(frames[40:60, ...]):
    scipy.misc.imsave('{}/pendulum-uniform/frame{}.png'.format(dataset_path, n), np.squeeze(f))
print('done')
