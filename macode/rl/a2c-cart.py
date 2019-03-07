from baselines.run import main

args = [
    '--num_timesteps', '1e6',
    '--env', 'Pendulum-v0',
    '--alg', 'a2c',
    '--num_env', '16',
    '--network', 'mlp',
]

main(args)
pass