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

features = np.load('processed_data/train_data/numpy_listener/features.npy')
labels = np.load('processed_data/train_data/numpy_listener/labels.npy')

cn_and_rest = labels >= 2
features = features[cn_and_rest]
labels = labels[cn_and_rest]

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

xTrain, xTest, yTrain, yTest = train_test_split(features, labels, test_size=0.25, random_state=101)

# truncate and pad input sequences
# create the model
# yTest = to_categorical(yTest-1, 3, dtype='int16')
# yTrain = to_categorical(yTrain-1, 3, dtype='int16')
yTest = to_categorical(yTest-2, dtype='int16')
yTrain = to_categorical(yTrain-2, dtype='int16')

model = Sequential()
# model.add(Embedding(10000, 32))
model.add(LSTM(500, input_shape=(300, 50), activation='tanh'))
model.add(Dense(16))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
print(model.summary())

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')
history = model.fit(xTrain, yTrain, epochs=30, batch_size=1, validation_split=0.33, callbacks=[reduce_lr], shuffle=True)
# Final evaluation of the model
scores = model.evaluate(xTest, yTest, verbose=0)
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
df.to_excel('result/1.1_failed_1/model_info_1.1_failed_1.xlsx')

model.save('models/model_1.1_failed_1.h5')
