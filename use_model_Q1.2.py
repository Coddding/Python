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
model = keras.models.load_model('models/model_1.2.h5')


def get_cn_data(people, number):
    return loadmat(f'origin_data/test_data/Q1-Q2/C{people}/unknown{number}.mat')['tc'].reshape(1, 300, 50, 1)


def getTestResult(people, number):
    data = get_cn_data(people, number)
    result = model.predict(data)
    return list(result[0])

result = []
peoples = list(range(51, 60))
numbers = [2,1,3,1,1,2,2,1,1]
for peo, num in zip(peoples, numbers):
    result.append(getTestResult(peo, num))

df = pd.DataFrame(result)
df.to_excel('result/1.2/result_1.2.xlsx')



