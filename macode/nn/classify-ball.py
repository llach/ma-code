import tensorflow as tf

from tensorflow.keras.backend import get_session
from macode.nn.classify_ball_base import classify_ball

retrain_set = 'breakout-nenv16-rlc10000-k4-seed0-modelscratch-b1'
vae_set = 'breakout-b1.0-lat20-lr0.0001'


for neurons in [1024, 512, 128, 32]:

    classify_ball(retrain_set, mlp_neurons=neurons, epochs=1)
    get_session().close()
    tf.reset_default_graph()

    classify_ball(vae_set, mlp_neurons=neurons, epochs=1)
    get_session().close()
    tf.reset_default_graph()
