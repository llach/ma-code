from baselines.run import main

args = [
    '--num_timesteps', '10e6',
    '--env', 'PendulumThetaStack-v0',
    '--alg', 'ddpg',
    '--network', 'mlp',
    '--num_env', '24',
]

main(args)
pass