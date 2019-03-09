from baselines.run import main

args = [
    '--num_timesteps', '10e6',
    '--env', 'PendulumTheta-v0',
    '--alg', 'a2c',
    '--network', 'mlp',
    '--num_env', '24',
]

main(args)
pass