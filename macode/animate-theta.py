
import numpy as np
from gym.envs.classic_control.pendulum_theta import PendulumThetaEnv

env_id = 'PendulumTheta-v0'
env_type = 'classic_control'
load_from = 'pendvisualuniform-b77.5-lat5-lr0.001-2019-03-21T00:13'

steps = 200
env = PendulumThetaEnv()
ths = []
mus = []

import pyqtgraph as pg
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

plt = pg.plot()
plt.setWindowTitle('PendulumTest')
plt.addLegend()

nobs = env.observation_space.shape[0]
o = env.reset()
lines = []
for ob in range(nobs):
    lines.append(plt.plot(name='obs{}'.format(ob)))

offset = 0
LINECOLORS = ['r', 'g', 'b', 'c', 'm', 'y']
t = 0
for _ in range(5000):
    ths.append(t)
    t += 1

    o, r, d, i = env.step([env.action_space.sample()])
    mus.append(o)

    for ob in range(nobs):
        lines[ob].setData(ths, np.asarray(mus)[...,ob], clear=True, pen=pg.mkPen(cosmetic=True, width=2.5, color=LINECOLORS[ob%len(LINECOLORS)-1]))

    pg.QtGui.QApplication.processEvents()

    if len(ths) > steps:
        ths = ths[-steps:]
        mus = mus[-steps:]

    env.render()
    if t % steps == 0:
        offset += 2*np.pi
        env.reset()

