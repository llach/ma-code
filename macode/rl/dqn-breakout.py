from baselines.run import main

args = [
    '--num_timesteps', '50e6',
    '--env', 'BreakoutNoFrameskip-v4',
    '--alg', 'deepq',
]

main(args)