import csv
import logging
import numpy as np

from forkan import model_path
from forkan.common.utils import ls_dir

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


log = logging.getLogger(__name__)

models_dir = '{}a2c/'.format(model_path)
dirs = ls_dir(models_dir)

filter = 'pendulum-noVAE-nenv24-2019-03-10T17:27'
# filter = ''
for d in dirs:
    ds_name = d.split('/')[-1].split('-')[0]
    model_name = d.split('/')[-1]

    if filter is not None and not '':
        if filter not in model_name:
            log.info('skipping {}'.format(model_name))
            continue

    log.info('plotting {}'.format(model_name))

    rew, entr, ev, vall = [], [], [], []
    rewi = entri = evi = valli = None

    with open('{}/progress.csv'.format(d)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                for n, ele in enumerate(row):
                    if 'reward' in ele:
                        rewi = n
                    elif 'entropy' in ele:
                        entri = n
                    elif ele == 'explained_variance':
                        evi = n
                    elif ele == 'value_loss':
                        valli = n
                line_count += 1
            else:
                if np.asarray(row[rewi], dtype=np.float) == -np.infty:
                    pass
                else:
                    rew.append(row[rewi])
                    entr.append(row[entri])
                    ev.append(row[evi])
                    vall.append(row[valli])

                line_count += 1

    if rew == []:
        log.info('{} had no progress, skipping'.format(model_name))
        continue

    plt.figure(figsize=(10, 10))

    print('plotting losses for {} ...'.format(model_name))
    plt.subplot(221)
    plt.plot(np.asarray(rew, dtype=np.float32), label='mean reward')
    plt.legend()

    plt.subplot(222)
    plt.plot(np.asarray(entr, dtype=np.float32), label='policy entropy')
    plt.legend()

    plt.subplot(223)
    plt.plot(np.asarray(ev, dtype=np.float32), label='explained variance')
    plt.legend()

    plt.subplot(224)
    plt.plot(np.asarray(vall, dtype=np.float32), label='value loss')
    plt.legend()

    plt.title(model_name)

    plt.savefig('{}/plot.png'.format(d))
    plt.show()

log.info('Done.')
