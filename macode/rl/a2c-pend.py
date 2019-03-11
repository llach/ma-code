from baselines.run import main

for zahl in ['3', '4', '5']:
    args = [
        '--num_timesteps', '10e6',
        '--env', 'PendulumThetaStack-v0',
        '--alg', 'a2c',
        '--network', 'mlp',
        '--num_env', zahl,
    ]
    main(args)
