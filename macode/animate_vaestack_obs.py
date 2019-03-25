
import numpy as np
from gym.envs.classic_control.pendulum_test import PendulumTestEnv
from forkan.rl import VAEStack

env_id = 'PendulumTest-v0'
env_type = 'classic_control'
load_from = 'pendvisualuniform-b77.5-lat5-lr0.001-2019-03-21T00:13'

steps = 200
env = PendulumTestEnv(steps=steps)
venv = VAEStack(env, load_from=load_from, k=3)

venv.reset()
ths = []
mus = []

import pyqtgraph as pg
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

plt = pg.plot()
plt.setWindowTitle('PendulumTest - VAEStack')
plt.addLegend()

nobs = venv.observation_space.shape[0]

lines = []
for ob in range(nobs):
    lines.append(plt.plot(name='obs{}'.format(ob)))

offset = 0
LINECOLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'w']

for _ in range(5000):
    ths.append(offset+env._get_theta())

    o, r, d, i = venv.step(0)
    mus.append(o)

    for ob in range(nobs):
        lines[ob].setData(ths, np.asarray(mus)[...,ob], clear=True, pen=pg.mkPen(cosmetic=True, width=2.5))#, color=LINECOLORS[ob%len(LINECOLORS)-1]))

    pg.QtGui.QApplication.processEvents()

    if len(ths) > steps:
        ths = ths[-steps:]
        mus = mus[-steps:]

    env.render()
    if d:
        offset += 2*np.pi
        venv.reset()

