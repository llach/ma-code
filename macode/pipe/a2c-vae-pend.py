from baselines.run import main

args = [
    '--num_timesteps', '10e6',
    '--env', 'Pendulum-v0',
    '--alg', 'a2c',
    '--network', 'mlp',
    '--num_env', '16',
    '--vae', 'pend-optimal',
    # '--load_path', '/Users/llach/.forkan/models/a2c/pendulum-noVAE-nenv16-2019-03-07T22:11/weights_latest',
    # '--play'
]

main(args)
pass