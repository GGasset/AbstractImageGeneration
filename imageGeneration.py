import os
from random import randint
from time import localtime
import datetime as date
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from celluloid import Camera


sqrt_resolution = 32
resolution = (sqrt_resolution, sqrt_resolution)
rgb_image_shape = (sqrt_resolution, sqrt_resolution, 3)
single_rgb_image_shape = (1, sqrt_resolution, sqrt_resolution, 3)
rgb_pixel_count = sqrt_resolution * sqrt_resolution * 3

global model_path
global is_nn_saved
global diffusions_per_image
global image_folders

def main():
    global diffusions_per_image
    diffusions_per_image = 50

    std = 255. / (diffusions_per_image * 6)

    global model_path, is_nn_saved
    model_path = './nn.hdf5'

    print('Getting paths...')
    paths = get_image_paths('./images/')

    while True:
        is_nn_saved = os.path.isfile(model_path)

        first_prompt = 'Do you want to continue with the saved nn?'
        if is_nn_saved:
            first_prompt += ' (else you will loose the saved network)'
        if is_nn_saved and get_boolean_input(first_prompt):
            if get_boolean_input('Do you want to show (an) image/s from the nn instead of training the nn?'):
                show_generated_images(paths, std, save_instead_of_displaying=get_boolean_input('Do you want to save the nn/\'s images as a .gif instead of showing them?'), img_count=get_input_int('How many images do you want to generate?'))
            else:
                train(paths, std, model_path)
        else:
            train(paths, std)

def show_generated_images(paths: list[str], std: float, save_instead_of_displaying: bool = True, img_count: int = 1):
    global model_path
    model = generate_model(model_path)
    fig = plt.figure()
    camera = Camera(fig)
    for i in range(img_count):
        generated = generate_image(model, paths, std)
        plt.imshow(generated)
        if save_instead_of_displaying:
            camera.snap()
        else:
            plt.show()
            plt.clf()
    if save_instead_of_displaying:
        animation = camera.animate()
        animation.save('./GeneratedImages.gif')

def train(paths: list[str], std: float, model_path: str = None) -> None:
    global diffusions_per_image
    
    to_datetime = get_current_datetime()
    if get_boolean_input('Do you want to train the network for a specific amount of time?'):
        to_datetime = get_input_future_datetime('For how much time do you want to train the network?')

    epoch_prompt = 'On how many epochs do you want to train the network?'
    if to_datetime:
        epoch_prompt += ' (for each iteration for the epochs until you pass the specified time)'
    trained_once = False
    epochs=get_input_int(epoch_prompt)

    print('Gathering images...')
    images = paths.agg(proccess_path)
    print('Generating training data...')
    X, Y = generate_training_data(images, std)
    print('Generating model...')
    model = generate_model(model_path)
    print('Fitting model..')
    while not trained_once or get_current_datetime() < to_datetime:
        model = fit_model(model, X, Y, epochs)
        trained_once = True
    print('Saving model...')
    model.save('./nn.hdf5')

def get_input_int(prompt: str, end='\n') -> int:
    while True:
        print(prompt, end=end)
        try:
            output = int(input())
            return output
        except:
            pass

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

def get_current_datetime() -> date.datetime:
    current_time = localtime()
    current_year = current_time[0]
    current_month = current_time[1]
    current_day = current_time[2]
    current_hour = current_time[3]
    current_minute = current_time[4]
    current_date = date.datetime(current_year, current_month, current_day, current_hour, current_minute)
    return current_date

def get_input_future_datetime(prompt: str, action: str = 'Train') -> date.datetime:
    print(prompt)
    days = 10E30
    while days < 0:
        days = get_input_int(f'\tFor how many days do you want to {action}', end='')
    hours = 1E30
    while hours >= 24 or hours < 0:
        hours = get_input_int(f'\tFow how many hours do you want to {action}')
    
    current_date = get_current_datetime()
    to_date = current_date + date.timedelta(days=days, hours=hours)
    return to_date

def get_image_paths(folder_path: str) -> pd.DataFrame:
    global image_folders
    image_folders = []
    paths = []
    for folder_path, _, file_paths in os.walk(folder_path):
        image_folders.append(os.path.basename(folder_path))
        for path in file_paths:
            paths.append(os.path.join(folder_path, path))

    output = pd.Series(data=paths)
    return output

def proccess_path(path: str) -> tf.Tensor:
    def decode_img(img) -> tf.Tensor:
        img = tf.io.decode_jpeg(img, channels=3)
        return tf.image.resize(img, resolution)
    
    img = tf.io.read_file(path)
    img = decode_img(img)
    return img

def tensor_to_numpy(img: tf.Tensor) -> np.ndarray:
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

def get_training_data_shape(images: pd.Series) -> tuple[int, int, int, int]:
    global diffusions_per_image
    return (len(images) * (diffusions_per_image - 1), sqrt_resolution, sqrt_resolution, 3)

def diffuse_img(img: tf.Tensor, std: float) -> list[tf.Tensor]:
    global diffusions_per_image

    diffusions = []
    diffusions.append(img)
    for i in range(diffusions_per_image - 1):
        diffusions.append(AddGaussianNoise(diffusions[i], std))
    return diffusions

def generate_training_data(images: pd.Series, std: float, dtype: str = 'uint8') -> tuple[np.ndarray, np.ndarray]:
    global diffusions_per_image

    data_shape = get_training_data_shape(images)
    X = np.ndarray(data_shape)
    Y = np.ndarray(data_shape)
    counter = 0
    for i, img in enumerate(images):
        diffusions = diffuse_img(img, std)
        for j in range(diffusions_per_image - 1):
            X[counter] = diffusions[j + 1]
            Y[counter] = diffusions[j]
            counter += 1

    X = X.astype(dtype)
    Y = Y.astype(dtype)
    return (X, Y)

def generate_model(path: str = None):
    print('Generating model...')
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Rescaling(1./255., input_shape=rgb_image_shape))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(rgb_pixel_count, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(300, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(350, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(400, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(rgb_pixel_count, activation='sigmoid'))
    model.add(tf.keras.layers.Reshape(rgb_image_shape))
    model.add(tf.keras.layers.Rescaling(255.))

    if path:
        print('Getting values from disk...')
        model.load_weights(path)

    model.compile(optimizer='Nadam', loss=tf.keras.losses.MeanSquaredError())

    print(model.summary())
    return model


def fit_model(model: tf.keras.Sequential, X: np.ndarray, Y: np.ndarray, epochs: int = 10) -> tf.keras.models.Sequential:
    model.fit(x=X, y=Y, batch_size=8, epochs=epochs, use_multiprocessing=True)
    return model

def generate_image(model: tf.keras.Sequential, paths: list[str], std: float, img: tf.Tensor | np.ndarray = None, counter: int = 0, dtype: str = 'uint8') -> np.ndarray:
    global diffusions_per_image

    if img == None:
        img = proccess_path(paths[randint(0, len(paths) - 1)])
        img = diffuse_img(img, std)[diffusions_per_image - 1]
        img = tf.convert_to_tensor(img.numpy().reshape(single_rgb_image_shape))
        counter += 1

    img = model.predict(img, use_multiprocessing=True)
    counter += 1
    
    is_final_output = counter >= diffusions_per_image
    if is_final_output:
        img = img.reshape(rgb_image_shape).astype(dtype)
        return img
    else:
        img = tf.convert_to_tensor(img)
        return generate_image(model, None, None, img=img, counter=counter)

def display_img(img: tf.Tensor | np.ndarray, title: str = ''):
    if  type(img) == tf.Tensor:
        img = tensor_to_numpy(img)
    img = img.reshape(rgb_image_shape)
    plt.title(title)
    plt.imshow(img)
    plt.show()
    plt.clf()

if __name__ == '__main__':
    main()