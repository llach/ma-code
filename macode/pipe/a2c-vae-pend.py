from baselines.run import main

args = [
    '--num_timesteps', '1e6',
    '--env', 'Pendulum-v0',
    '--alg', 'a2c',
    '--network', 'mlp',
    '--num_env', '1',
    '--vae', 'pend'
]

main(args)
pass