from baselines.run import main

args = [
    '--num_timesteps', '50e6',
    '--env', 'GopherNoFrameskip-v4',
    '--alg', 'a2c',
    '--num_env', '16',
    '--network', 'cnn',
]

main(args)
pass