import os
import numpy as np
import tensorflow as tf
import datetime
import tensorflow.keras.backend as K

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import Callback

from forkan.common.utils import create_dir
from forkan.common.csv_logger import CSVLogger
from forkan.common.tf_utils import scalar_summary


from forkan import dataset_path, model_path


def classify_ball(ds_path, name_prefix, mlp_neurons=16, val_split=0.2, batch_size=128, epochs=100):

    K.set_session(tf.Session())

    dataset_prefix = 'ball_latents_'
    ds = np.load(f'{dataset_path}{ds_path}.npz')
    home = os.environ['HOME']

    orgs = ds['originals']
    poss = ds['ball_positions']

    dt = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M')
    model_name = f'{name_prefix}-N{mlp_neurons}-{ds_path}-{dt}'

    model_save_path = f'{model_path}classify-ball/{model_name}'
    create_dir(model_save_path)

    csv = CSVLogger(f'{model_save_path}/progress.csv', *['timestamp', 'nbatch', 'mae_train',
                                                         'mse_train', 'mae_test', 'mse_test'])

    if name_prefix == 'VAE':
        lats = ds['vae_latents']
    elif name_prefix == 'RETRAIN':
        lats = ds['latents']
    else:
        print(f'name {name_prefix} unknown!')
        print(0)


    model = Sequential([
    Dense(mlp_neurons, activation='relu', input_shape=(lats.shape[-1],)),
    Dense(mlp_neurons, activation='relu'),
    Dense(poss.shape[-1], activation='sigmoid')])

    model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss='mse', metrics=['mae'])

    sess = K.get_session()

    idxes = np.arange(lats.shape[0])
    np.random.shuffle(idxes)
    split_idx = int(lats.shape[0]*(1-val_split))

    def draw_predicted_balls(imgs, locations, real_loc):
        imgs = imgs.copy()

        for n, img in enumerate(imgs):
            for j in [-1, 0, 1]:
                for i in [-1, 0, 1]:
                    x, y = np.clip(int((locations[n, 0]*210)+j), 0, 209), np.clip(int((locations[n, 1]*160)+i), 0, 159)
                    img[x, y] = [0, 200, 200]

                    x, y = np.clip(int((real_loc[n, 0] * 210) + j), 0, 209), np.clip(int((real_loc[n, 1] * 160) + i),
                                                                                      0, 159)
                    img[x, y] = [200, 0, 200]

        return np.asarray(imgs, dtype=np.uint8)

    class TBCB(Callback):
        def __init__(self, m, ovo):
            self.mse_ph = tf.placeholder(tf.float32, (), name='mse-train')
            self.mae_ph = tf.placeholder(tf.float32, (), name='mae-train')

            self.val_mse_ph = tf.placeholder(tf.float32, (), name='mse-test')
            self.val_mae_ph = tf.placeholder(tf.float32, (), name='mae-test')

            self.im_ph = tf.placeholder(tf.uint8, (1, 210*3, 160*5, 3), name='pred-ball-pos-ph')

            tr_sum = []
            tr_sum.append(scalar_summary('mse-train', self.mse_ph, scope='train'))
            tr_sum.append(scalar_summary('mae-train', self.mae_ph, scope='train'))

            te_sum = []
            te_sum.append(scalar_summary('mse-test', self.val_mse_ph, scope='test'))
            te_sum.append(scalar_summary('mae-test', self.val_mae_ph, scope='test'))

            self.im_sum = tf.summary.image('pred-ball-pos', self.im_ph)
            self.mtr_sum = tf.summary.merge(tr_sum)
            self.mte_sum = tf.summary.merge(te_sum)

            self.fw = tf.summary.FileWriter(f'{home}/ball/{model_name}', graph=sess.graph)
            self.ovo = ovo
            self.step = 0
            self.m = m

        def on_batch_end(self, batch, logs={}):
            self.step += 1

            mse_t = logs['loss']
            mae_t = logs['mean_absolute_error']

            # this is usually only given on epoch end. may that resolution suffices?
            val_mse_t,  val_mae_t = self.m.evaluate(x=self.validation_data[0], y=self.validation_data[1])

            su, se = sess.run([self.mtr_sum, self.mte_sum], feed_dict={
                self.mse_ph: mse_t,
                self.mae_ph: mae_t,
                self.val_mse_ph: val_mse_t,
                self.val_mae_ph: val_mae_t,
            })

            csv.writeline(
                datetime.datetime.now().isoformat(),
                self.step,
                mae_t,
                mse_t,
                val_mae_t,
                val_mse_t,
            )

            self.fw.add_summary(su, self.step)
            self.fw.add_summary(se, self.step)

        def on_epoch_end(self, epoch, logs=None):
            test_idxes = np.random.choice(self.validation_data[0].shape[0]-1, 15, replace=False)
            predicted_locations = model.predict(self.validation_data[0][test_idxes])
            imgs = draw_predicted_balls(self.ovo[test_idxes], predicted_locations, self.validation_data[1][test_idxes])

            r1 = np.concatenate(imgs[0:5], axis=1)
            r2 = np.concatenate(imgs[5:10], axis=1)
            r3 = np.concatenate(imgs[10:15], axis=1)

            img_mat = np.concatenate([r1, r2, r3], axis=0)

            img_sum = sess.run(self.im_sum, feed_dict={self.im_ph: [img_mat]})

            self.fw.add_summary(img_sum, self.step)

    model.fit(lats[idxes][:split_idx], poss[idxes][:split_idx], epochs=epochs, batch_size=batch_size,
              validation_data=(lats[idxes][split_idx:], poss[idxes][split_idx:]), callbacks=[TBCB(model, orgs[idxes][split_idx:])])

    model.save_weights(f'{model_save_path}/weights.h5')
    csv.flush()
    del csv

    return None