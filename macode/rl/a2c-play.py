from baselines.run import main

"""
obs space: theta and delta theta series:
pendulumthetastack-noVAE-nenv2-2019-03-11T13:30
k = 20

ground truth:
pendulum-noVAE-nenv16-2019-03-07T22:11

"""

args = [
    '--num_timesteps', '10e6',
    '--env', 'PendulumThetaStack-v0',
    '--alg', 'a2c',
    '--network', 'mlp',
    '--load_path', '/Users/llach/.forkan/models/a2c/pendulumthetastack-noVAE-nenv2-2019-03-11T13:30/weights_latest',
    '--play', 'True'
]

main(args)
