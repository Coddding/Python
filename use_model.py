# LSTM for sequence classification in the IMDB dataset
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import *
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model
from scipy.io import loadmat

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
model = keras.models.load_model('models/model_v2_acc_445_20_39_13.h5')

features_brain_cn = np.load('processed_data/train_data/numpy_listener/features_brain_cn.npy.')
features_memory = np.load('processed_data/train_data/numpy_listener/features_memory.npy')
features_speaker = np.load('processed_data/train_data/numpy_listener/features_speaker.npy')
labels = np.load('processed_data/train_data/numpy_listener/labels.npy')

# score = model.evaluate(x=[features_brain_cn, features_memory, features_speaker], y=labels)
# print(f"evaluate: {score[0]}, {score[1]}")

def get_cn_data(people):
    cn_index = [2,1,3,1,1,2,2,1,1]
    return loadmat(f'origin_data/test_data/Q1-Q2/C{people}/unknown{cn_index[people - 51]}.mat')['tc'].reshape(1, 300, 50, 1)

def get_speaker_data(people):
    speaker_index = [3,1,3,2,1,3,3,3,1]
    return loadmat(f'origin_data/train_data/speaker/ica_speaker_s{speaker_index[people-51]}.mat')['tc'].reshape(1, 300, 50, 1)

def get_memory_data(people):
    memory_data = [
        [9,9],
        [10,6],
        [9,8],
        [12,8],
        [9,7],
        [9,8],
        [8,5],
        [10,8],
        [9,4]
    ]
    return np.array(memory_data[people-51]).reshape((1,2))

def getTestResult(people):
    cn = get_cn_data(people)
    speaker = get_speaker_data(people)
    memory = get_memory_data(people)
    result = model.predict([cn, memory, speaker])
    return result[0][0]

result = []
peoples = list(range(51, 60))
for peo in peoples:
    result.append(getTestResult(peo))

df = pd.DataFrame(result)
df.to_excel('result/result_v2.xlsx')



