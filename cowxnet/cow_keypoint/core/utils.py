import numpy as np

def save_npy(save_path, array):
    np.save(save_path, array)

def load_npy(load_path):
    array = np.load(load_path)
    return array