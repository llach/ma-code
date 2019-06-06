import tensorflow as tf

from tensorflow.keras.backend import get_session
from macode.nn.classify_ball_base import classify_ball

dataset_name = 'ball_latents_VAE_breakout-b1.0-lat20-lr0.0001_POL_breakout-nenv16-rlc10000.0-k4-seed0-modelscratch-b1'


for neurons in [1024, 512, 128, 32]:

    classify_ball(dataset_name, 'RETRAIN', mlp_neurons=neurons)
    get_session().close()
    tf.reset_default_graph()

    classify_ball(dataset_name, 'VAE', mlp_neurons=neurons)
    get_session().close()
    tf.reset_default_graph()
