# LSTM for sequence classification in the IMDB dataset
import numpy as np
import tensorflow as tf
from keras.layers import *
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

features_brain_cn = np.load('processed_data/train_data/numpy_listener/features_brain_cn.npy.')
features_memory = np.load('processed_data/train_data/numpy_listener/features_memory.npy')
labels = np.load('processed_data/train_data/numpy_listener/labels.npy')
labels = zscore(labels)

train_brain_cn, test_brain_cn, train_memory, test_memory, train_labels, test_labels \
    = train_test_split(features_brain_cn, features_memory, labels)

brain_input = Input(shape=(300, 50, 1))
info_input = Input(shape=(2,))

brain = Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding="same")(brain_input)
brain = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(brain)
brain = Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid')(brain)
brain = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(brain)
brain = Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid')(brain)
brain = Flatten()(brain)
brain = Dense(84, activation='tanh')(brain)
brain = Dense(4)(brain)
brain_model = Model(inputs=brain_input, outputs=brain)

info = Dense(4)(info_input)
info_model = Model(inputs=info_input, outputs=info)

combined = Concatenate(axis=1)([brain_model.output, info_model.output])

total = Dense(2)(combined)
total = Dense(1, activation='linear')(total)

model = Model(inputs=[brain_input, info_input], outputs=total)

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
print(model.summary())

history = model.fit(x=[train_brain_cn, train_memory], y=train_labels, epochs=50, batch_size=1, validation_split=0.2,
          callbacks=[ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')], shuffle=True)
# # Final evaluation of the model
scores = model.evaluate([test_brain_cn, test_memory], test_labels, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# plot train and validation loss
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()

df = pd.DataFrame(history.history)
df.to_excel('result/model_info_v1.xlsx')

model.save('models/model_v1.h5')
