from utils import molecule_atom_distance_vectors
import tensorflow as tf
from keras.layers import Dense

Xtrain, Ytrain, Xtest, Ytest = molecule_atom_distance_vectors()

N = len(Xtrain)

model = tf.keras.Sequential([
    Dense(units=200, input_shape=158, activation='relu'),
    Dense(untis=300, activation='relu'),
    Dense(units=10, activation='')
])

model.compile(tf.optimizers.Adam, loss=tf.losses.binary_crossentropy, metrics=['Accuracy'])

model.fit(Xtrain, Ytrain, batch_size=500, epochs=1)

model.evaluate(Xtest, Ytest)
