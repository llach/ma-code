from baselines.run import main

args = [
    '--num_timesteps', '5e6',
    '--env', 'Pendulum-v0',
    '--alg', 'a2c',
    '--network', 'mlp',
    '--num_env', '1',
    '--vae', 'pend',
    # '--load_path', '/Users/llach/.forkan/models/a2c/pendulum-noVAE-nenv16-2019-03-07T21:56/weights_latest',
    # '--play'
]

main(args)
pass