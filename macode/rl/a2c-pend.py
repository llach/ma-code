from baselines.run import main

args = [
    '--num_timesteps', '5e6',
    '--env', 'Pendulum-v0',
    '--alg', 'a2c',
    '--network', 'mlp',
    '--num_env', '16',
]

main(args)
pass