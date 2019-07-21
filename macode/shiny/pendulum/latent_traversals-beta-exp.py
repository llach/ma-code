import os
import logging

import cairocffi as cairo
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from forkan.common.utils import ls_dir, setup_plotting, get_figure_size
from forkan.models import RetrainVAE

sns.set()

logger = logging.getLogger(__name__)

network = 'pendulum'
plt_shape = [1, 5]

ylims = setup_plotting()

home = os.environ['HOME']
models_dir = f'{home}/.forkan/done/pendulum/ppo2-beta-exp'
dirs = ls_dir(models_dir)


for rlc in [1, 30]:
    for beta in [5, 20, 30, 60, 80]:
        for d in dirs:
            model_name = d.split('/')[-1]

            if not all(sub in model_name for sub in [f'rlc{rlc}-', f'b{beta}-']):
                logger.info('skipping {}'.format(model_name))
                continue

            v = RetrainVAE(rlpath=f'{models_dir}/{model_name}', latent_dim=5, input_shape=(64, 64, 1), sess=tf.Session(), k=1, with_attrs=True)
            v.s.run(tf.global_variables_initializer())
            v.load()

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

            idxes = np.arange(5)

            nlat = v.latent_dim
            thetas = np.asarray(thetas, dtype=np.float)

            idx = 1
            show_recs = False

            mus, logvars = v.encode(frames)
            mus = np.squeeze(np.asarray(mus, dtype=np.float32))
            logvars = np.squeeze(np.asarray(logvars, dtype=np.float32))
            print(mus.shape, logvars.shape)

            mus = np.moveaxis(mus, 0, -1)
            sigmas = np.moveaxis(np.exp(0.5 * logvars), 0, -1)
            sigmasmean = np.mean(sigmas, axis=1)

            fig, ax = plt.subplots(1, 1, figsize=get_figure_size())

            for idx in idxes:
                ax.plot(thetas, mus[idx], label=f'$\mu_{idx}$'.format(idx))
                ax.scatter(thetas, mus[idx], linewidths=0.05, marker='+')

            ax.set_ylabel('$\mu_i$')
            ax.set_xlabel('$\\alpha$')
            # ax.legend(loc='upper right')

            pi = np.pi
            plt.xticks([0, pi/2, pi, 3*pi/2, 2*pi], ['0', '$\\frac{\pi}{2}$', '$\pi$', '$\\frac{3\pi}{2}$', '$2\pi$'])
            fig.tight_layout()
            plt.legend()

            plt.savefig(f'{home}/.forkan/done/pendulum/figures/trav-rlc{rlc}-b{beta}.pdf')
            plt.show()

        tf.reset_default_graph()

logger.info('Done.')
