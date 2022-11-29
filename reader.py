from scipy.io import loadmat
import pandas as pd
import numpy as np
import os

def read_file(file_name: str, data_folder: str = '') -> np.ndarray:
    d = loadmat(os.path.join(data_folder, file_name))
    tc: np.ndarray = d['tc']
    ic: np.ndarray = d['ic']
    if tc.shape == (300, 50):
        return np.matmul(tc, ic)
    else:
        raise Exception('read data file error!')

def get_data_folder(lang: int) -> str:
    return {
        1: 'origin_data/train_data/listener_MG',
        2: 'origin_data/train_data/listener_CN',
        3: 'origin_data/train_data/listener_Rest'
    }[lang]


result = np.zeros((40*3, 300, 57090)) # 150 numbers, 300 lines, 50 dims
labels = np.zeros((40*3, ), dtype='int16')
i = 0
for lang in [1, 2, 3]:
    data_folder = get_data_folder(lang)
    file_names = os.listdir(data_folder)

    for file_name in file_names:
        d = read_file(file_name, data_folder=data_folder)
        result[i] = d
        labels[i] = lang
        i += 1

np.save('processed_data/train_data/numpy_listener_origin/features.npy', result)
np.save('processed_data/train_data/numpy_listener_origin/labels.npy', labels)






