import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from celluloid import Camera
from random import random

global grayscale
grayscale = None

sqrt_resolution = 24
resolution = (sqrt_resolution, sqrt_resolution)

grayscale_image_shape = (sqrt_resolution, sqrt_resolution, 1)
rgb_image_shape = (sqrt_resolution, sqrt_resolution, 3)

def main():
    global grayscale

    paths = get_image_paths('./images/')
    print(paths.head())
    img = proccess_path(paths[0])
    images = paths.agg(proccess_path)
    print(type(images))    
    

def get_boolean_input(prompt: str) -> bool:
    accepted_answers = ['yes', 'no', 'y', 'n', '1', '0']
    positive_answers = ['yes', 'y', '1']
    successful_input = False
    answer = None
    while not successful_input:
        print(prompt)
        print(f'Accepted answers: {str(accepted_answers)}')
        answer = input().lower()
        successful_input = accepted_answers.__contains__(answer)
    return positive_answers.__contains__(answer)

def get_image_paths(folder_path: str) -> pd.DataFrame:
    paths = []
    for folder_path, _, file_paths in os.walk(folder_path):
        for path in file_paths:
            paths.append(os.path.join(folder_path, path))

    output = pd.Series(data=paths)
    return output

def proccess_path(path: str) -> tf.Tensor:
    def decode_img(img) -> tf.Tensor:
        global channels
        img = tf.io.decode_jpeg(img, channels=3)
        return tf.image.resize(img, resolution)
    
    img = tf.io.read_file(path)
    img = decode_img(img)
    return img

def img_to_numpy(img: tf.Tensor):
    return img.numpy().astype('uint8')

def GetGaussianNoise(mean: tf.Tensor | np.ndarray | float, std: float, shape: tuple) -> tf.Tensor:
    mean = np.zeros(shape) + mean
    noise = np.random.normal(mean, std)
    noise = tf.convert_to_tensor(noise, float)
    return noise

def AddGaussianNoise(img: tf.Tensor, std: float) -> tf.Tensor:
    shape = img.shape
    output = GetGaussianNoise(img, std, shape)
    return output

#def GetTrainingData()

if __name__ == '__main__':
    main()