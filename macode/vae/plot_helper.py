import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def sigma_bars(mean_sigma, plot_shape, thresh=0.8, title=None):
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

    plt.show()