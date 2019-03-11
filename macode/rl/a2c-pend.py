from baselines.run import main

args = [
    '--num_timesteps', '10e6',
    '--env', 'PendulumThetaStack-v0',
    '--alg', 'a2c',
    '--network', 'mlp',
    '--num_env', '2',
]

main(args)
pass