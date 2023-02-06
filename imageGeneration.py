import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from celluloid import Camera
from random import random


sqrt_resolution = 24
resolution = (sqrt_resolution, sqrt_resolution)
rgb_image_shape = (sqrt_resolution, sqrt_resolution, 3)

def main():
    paths = get_image_paths('./images/')
    images = paths.agg(proccess_path)
    diffusions_per_image = 50
    std = 255. / diffusions_per_image
    X, Y = generate_training_data(images, std, diffusions_per_image)
    model = generate_fit_model(X, Y)
    model.save('./nn.hdf5')
    

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

def GetGaussianNoise(mean: tf.Tensor | np.ndarray | float, std: float, shape: tuple, dtype: str = 'uint8') -> tf.Tensor:
    mean = np.zeros(shape) + mean
    noise = np.random.normal(mean, std)
    noise = noise.astype(dtype)
    noise = tf.convert_to_tensor(noise, float)
    return noise

def AddGaussianNoise(img: tf.Tensor, std: float) -> tf.Tensor:
    shape = img.shape
    output = GetGaussianNoise(img, std, shape)
    return output

def generate_training_data(images: pd.Series[np.ndarray], std: float, diffusion_count: int) -> tuple[np.ndarray, np.ndarray]:
    X = []
    Y = []
    for img in images:
        diffusions = []
        diffusions.append(img)
        for i in range(diffusion_count - 1):
            diffusions.append(AddGaussianNoise(diffusions[i], std))
            X.append(diffusions[i])
            Y.append(diffusions[i + 1])

    X = np.array(X, dtype=int)
    Y = np.array(Y, dtype=int)
    return (X, Y)

def generate_fit_model(X: np.ndarray[tf.Tensor], Y: np.ndarray[tf.Tensor]) -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Rescaling(1.0/255))
    model.add(tf.keras.layers.Conv2D(2, resolution, input_shape=rgb_image_shape))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(2, (24, 34)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Rescaling(255.0))

    model.compile(tf.keras.optimizers.Nadam, loss=tf.keras.losses.MeanSquaredError)

    model.fit(x=X, y=Y, batch_size=8, epochs=50, use_multiprocessing=True)
    return model

if __name__ == '__main__':
    main()