import logging
import numpy as np
from forkan.models import VAE

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import cairocffi as cairo
import tensorflow as tf

logger = logging.getLogger(__name__)

network = 'pendulum'
load_from = 'pendvisualuniform-b80.0-lat5-lr0.001-2019-04-04T15/03'.replace('/', ':')
plt_shape = [1, 5]

v = VAE(load_from=load_from, network=network)

FRAMES = 200

thetas = []
frames = np.zeros([FRAMES, 64, 64, 1])

w, h = 64, 64
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


for i, theta in enumerate(np.linspace(0, 2 * np.pi, FRAMES)):
    frame = _render_pendulum(theta)
    frames[i] = frame
    thetas.append(theta)
thetas = np.asarray(thetas, dtype=np.float)


en_ints = [tf.placeholder(tf.float32, shape=(None, 64, 64, 1)) for _ in range(2)]
mus_t = v.stack_encoder(en_ints)
combined_out = tf.concat(mus_t, axis=1)

with v.s as s:
    s.run(tf.global_variables_initializer())
    v._load()
    feed = {}
    for t in en_ints:
        feed.update({t: frames})
    mus = np.asarray(s.run(mus_t, feed_dict=feed), dtype=np.float32)
mus = np.moveaxis(mus, -1, 1)

for i in range(mus.shape[0]):
    idxes = np.arange(5)
    for idx in idxes:
        plt.plot(thetas, mus[i, idx], label=f'enc-{i} mu-{idx}')
        plt.scatter(thetas, mus[i, idx])

    plt.title(load_from)
    plt.legend()
    plt.show()



