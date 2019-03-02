from baselines.run import main

args = [
    '--num_timesteps', '50e6',
    '--env', 'GopherNoFrameskip-v4',
    '--alg', 'a2c',
    '--num_env', '2',
    '--lr', '1e-3',
    '--network', 'mlp',
    '--vae', 'gopher',
]

main(args)
pass