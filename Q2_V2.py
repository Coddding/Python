# LSTM for sequence classification in the IMDB dataset
import numpy as np
import tensorflow as tf
from keras.layers import *
from scipy.io import loadmat
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

import datetime

time_back = datetime.datetime.now().strftime("%H_%M_%S")

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

features_brain_cn = np.load('processed_data/train_data/numpy_listener/features_brain_cn.npy.')
features_memory = np.load('processed_data/train_data/numpy_listener/features_memory.npy')
features_speaker = np.load('processed_data/train_data/numpy_listener/features_speaker.npy')
labels = np.load('processed_data/train_data/numpy_listener/labels.npy')
# labels = zscore(labels)

train_brain_cn, test_brain_cn, train_memory,\
test_memory, train_speaker, test_speaker, train_labels, test_labels \
    = train_test_split(features_brain_cn, features_memory, features_speaker, labels, test_size=0.25)

brain_input = Input(shape=(300, 50, 1))
info_input = Input(shape=(2,))
speaker_input = Input(shape=(300, 50, 1))

brain = Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding="same")(brain_input)
brain = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(brain)
brain = Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid')(brain)
brain = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(brain)
brain = Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid')(brain)
brain = Flatten()(brain)
brain = Dense(84, activation='tanh')(brain)
brain = Dense(4)(brain)
brain_model = Model(inputs=brain_input, outputs=brain)

speaker = Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding="same")(brain_input)
speaker = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(speaker)
speaker = Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid')(speaker)
speaker = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(speaker)
speaker = Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid')(speaker)
speaker = Flatten()(speaker)
speaker = Dense(84, activation='tanh')(speaker)
speaker = Dense(4)(speaker)
speaker_model = Model(inputs=brain_input, outputs=speaker)

info = Dense(4)(info_input)
info_model = Model(inputs=info_input, outputs=info)

combined = Concatenate(axis=1)([brain_model.output,info_model.output, speaker_model.output])

total = Dense(2)(combined)
total = Dense(1, activation='linear')(total)

model = Model(inputs=[brain_input, info_input, speaker_input], outputs=total)

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
print(model.summary())

history = model.fit(x=[train_brain_cn, train_memory, train_speaker], y=train_labels, epochs=30, batch_size=1, validation_split=0.3,
          callbacks=[ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')], shuffle=True)
# # Final evaluation of the model
scores = model.evaluate([test_brain_cn, test_memory, test_speaker], test_labels, verbose=0)
print(f"evaluate: {scores[0]}, {scores[1]}")
df = pd.DataFrame(history.history)
df.to_excel(f'result/model_info_v2_acc_{int(scores[1] * 100)}_loss_{int(scores[0])}.xlsx')

model.save(f'models/model_v2_acc_{int(scores[1] * 100)}_loss_{int(scores[0])}.h5')

# plot train and validation loss
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()


