import csv
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def sigma_bars(d, mean_sigma, plot_shape, thresh=0.8, title=None):
    num_zi = plot_shape[1]
    num_plots = plot_shape[0]

    # split zi into multiple bar plots
    ys = []
    xs = []
    pal = []

    for i in range(num_plots):
        xs.append(['z-{}'.format(j + (num_zi * i)) for j in range(num_zi)])
        ys.append(mean_sigma[i * num_zi:(i + 1) * num_zi])
        pal.append(['#90D7F3' if k > thresh else '#F78A8F' for k in ys[-1]])

    # create subplots
    f, axes = plt.subplots(num_plots, 1, figsize=(9, (6.5 * num_plots)))
    t = 'z-i sigmas'
    if title is not None:
        t += ' - {}'.format(title)
    plt.title(t)

    # show them heatmaps
    if num_plots == 1:
        sns.barplot(x=xs[0], y=ys[0], ax=axes, palette=pal[0], linewidth=0.5)
    else:
        for r, ax in enumerate(axes):
            sns.barplot(x=xs[r], y=ys[r], ax=ax, palette=pal[r], linewidth=0.5)

    plt.savefig('{}/sigmas.png'.format(d))
    plt.show()


def plot_z_kl(d, split=True):
    model_name = d.split('/')[-1]

    z_idx = []
    z_kls = []

    with open('{}/progress.csv'.format(d)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                for n, ele in enumerate(row):
                    if 'z' in ele:
                        z_idx.append(n)
                        z_kls.append([])
                line_count += 1
            else:
                zk = [row[i] for i in z_idx]
                for n, zkl in enumerate(zk):
                    z_kls[n].append(zkl)

    print('plotting z_kls for {} ...'.format(model_name))
    if split:
        half = len(z_idx)//2

        plt.subplot(211)
        for n, zz in enumerate(z_kls[:half]):
            plt.plot(np.asarray(zz, dtype=np.float32), label='z{}-kl'.format(n))
        plt.legend()

        plt.subplot(212)
        for n, zz in enumerate(z_kls[half:]):
            plt.plot(np.asarray(zz, dtype=np.float32), label='z{}-kl'.format(n+half))

        pass
    else:
        for zz in z_kls:
            plt.plot(np.asarray(zz, dtype=np.float32))

    plt.legend()
    plt.title(model_name)

    plt.savefig('{}/kls.png'.format(d))
    plt.show()


def plot_losses(d):
    model_name = d.split('/')[-1]

    l_idx, kl_idx = 0, 0
    loss, kl = [], []

    with open('{}/progress.csv'.format(d)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                for n, ele in enumerate(row):
                    print(ele)
                    if ele == 'loss':
                        l_idx = n
                    elif ele == 'kl-loss':
                        kl_idx = n
                line_count += 1
            else:
                loss.append(row[l_idx])
                kl.append(row[kl_idx])

    print('plotting losses for {} ...'.format(model_name))
    plt.subplot(211)
    plt.plot(np.asarray(loss, dtype=np.float32), label='loss')
    plt.legend()

    plt.subplot(212)
    plt.plot(np.asarray(kl, dtype=np.float32), label='kl-loss')
    plt.legend()

    plt.title(model_name)

    plt.savefig('{}/losses.png'.format(d))
    plt.show()

