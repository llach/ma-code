from baselines.run import main

args = [
    '--env', 'PendulumVisual-v0',
    '--num_timesteps', '50e6',
    '--alg', 'ppo2',
    '--network', 'cnn',
    '--nminibatches', '32',
    '--noptepochs', '10',
    '--num_env', '8',
    '--log_interval', '2',
    '--seed', '1',
    '--tensorboard', 'True',
]

main(args)
