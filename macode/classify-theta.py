import numpy as np
from forkan import dataset_path
from keras import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b


dataset_zip = np.load('{}/thetas-b77.5.npz'.format(dataset_path))

thetas = dataset_zip['thetas']
encodings = dataset_zip['encodings']

sidx = int(thetas.shape[0] * 0.75)

inp = Input((5,))
x = Dense(5, activation='relu')(inp)
x = Dense(5, activation='relu')(x)
out = Dense(1, activation='linear')(x)

model = Model(inp, out)
model.summary()

thetas, encodings = randomize(thetas, encodings)

x_train, x_test = encodings[:sidx], encodings[sidx:]
# y_train, y_test = np.stack((np.sin(thetas[:sidx]), np.cos(thetas[:sidx])), axis=1), np.stack((np.sin(thetas[sidx:]), np.cos(thetas[sidx:])), axis=1)
y_train, y_test = thetas[:sidx], thetas[sidx:]

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
hist = model.fit(x_train, y_train, batch_size=32, epochs=200, validation_data=(x_test, y_test))

plt.figure(figsize=(10, 7))

plt.plot(hist.history.get('loss'), label='loss')
plt.plot(hist.history.get('val_loss'), label='val_loss')
plt.legend()

plt.title('Encoding -> theta')

plt.show()
