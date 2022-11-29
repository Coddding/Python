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
cn_labels = np.load('processed_data/train_data/numpy_listener/labels.npy')
S1 = 0
S2 = 1
S3 = 2
cn_and_rest = cn_labels == 2
features = features[cn_and_rest]
labels = [S1,
          S1,
          S2,
          S2,
          S2,
          S2,
          S2,
          S2,
          S2,
          S2,
          S2,
          S2,
          S2,
          S2,
          S3,
          S3,
          S3,
          S3,
          S3,
          S3,
          S3,
          S3,
          S3,
          S3,
          S1,
          S1,
          S3,
          S3,
          S1,
          S2,
          S1,
          S3,
          S1,
          S3,
          S2,
          S2,
          S3,
          S2,
          S1,
          S2,
          ]

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

xTrain, xTest, yTrain, yTest = train_test_split(features, labels, test_size=0.3)

# truncate and pad input sequences
# create the model
# yTest = to_categorical(yTest-1, 3, dtype='int16')
# yTrain = to_categorical(yTrain-1, 3, dtype='int16')
yTest = to_categorical(yTest, dtype='int16')
yTrain = to_categorical(yTrain, dtype='int16')

xTrain = xTrain.reshape(xTrain.shape[0], 300, 50, 1)
xTest = xTest.reshape(xTest.shape[0], 300, 50, 1)
# model.add(Embedding(10000, 32))

model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1),
                 activation='tanh', input_shape=(300, 50, 1), padding="same"))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
# model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
model.add(Flatten())
model.add(Dense(84, activation='tanh'))
model.add(Reshape((84, 1)))
model.add(LSTM(32))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(0.00001), metrics=['accuracy'])
print(model.summary())

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')
history = model.fit(xTrain, yTrain, epochs=3, batch_size=1, validation_split=0.2, callbacks=[reduce_lr], shuffle=True)
# Final evaluation of the model
scores = model.evaluate(xTest, yTest, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

# plot train and validation loss
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()

df = pd.DataFrame(history.history)
df.to_excel('result/1.2/model_info_1.2.xlsx')

model.save('models/model_1.2.h5')
