import os
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

def main():
    global diffusions_per_image
    diffusions_per_image = 50

    global model_path, is_nn_saved
    model_path = './nn.hdf5'

    epoch_prompt = 'On how many epochs do you want to train the network?'

    while True:
        is_nn_saved = os.path.isfile(model_path)

        first_prompt = 'Do you want to continue with the saved nn?'
        if is_nn_saved:
            first_prompt += ' (else you will loose the saved network)'
        if is_nn_saved and get_boolean_input(first_prompt):
            if get_boolean_input('Do you want to show (an) image/s from the nn instead of training the nn?'):
                show_generated_images(save_instead_of_displaying=get_boolean_input('Do you want to save the nn/\'s images as a .gif instead of showing them?'), img_count=get_input_int('How many images do you want to generate?'))
            else:
                train(model_path, epochs=get_input_int(epoch_prompt))
        else:
            train(epochs=get_input_int(epoch_prompt))

def show_generated_images(save_instead_of_displaying: bool = True, img_count: int = 1):
    global model_path
    model = generate_model(model_path)
    fig = plt.figure()
    camera = Camera(fig)
    for i in range(img_count):
        generated = generate_image(model)
        plt.imshow(generated)
        if save_instead_of_displaying:
            camera.snap()
        else:
            plt.show()
            plt.clf()
    if save_instead_of_displaying:
        animation = camera.animate()
        animation.save('./GeneratedImages.gif')

def train(model_path: str = None, epochs: int = 12) -> None:
    global diffusions_per_image

    print('Getting paths...')
    paths = get_image_paths('./images/')
    print('Gathering images...')
    images = paths.agg(proccess_path)
    print('Generating training data...')
    std = 255. / (diffusions_per_image * 6)
    X, Y = generate_training_data(images, std, diffusions_per_image)
    print('Generating model...')
    model = generate_model(model_path)
    print('Fitting model..')
    model = fit_model(model, X, Y, epochs)
    print('Saving model...')
    model.save('./nn.hdf5')

def get_input_int(prompt: str) -> int:
    while True:
        print(prompt)
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

def get_image_paths(folder_path: str) -> pd.DataFrame:
    paths = []
    for folder_path, _, file_paths in os.walk(folder_path):
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

def get_training_data_shape(images: pd.Series, diffusion_count: int) -> tuple[int, int, int, int]:
    return (len(images) * (diffusion_count - 1), sqrt_resolution, sqrt_resolution, 3)

def generate_training_data(images: pd.Series, std: float, diffusion_count: int, dtype: str = 'uint8') -> tuple[np.ndarray, np.ndarray]:
    data_shape = get_training_data_shape(images, diffusion_count)
    X = np.ndarray(data_shape)
    Y = np.ndarray(data_shape)
    counter = 0
    for i, img in enumerate(images):
        diffusions = []
        diffusions.append(img)
        for j in range(diffusion_count - 1):
            diffusions.append(AddGaussianNoise(diffusions[j], std))
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

def generate_image(model: tf.keras.Sequential, img: tf.Tensor | np.ndarray = None, counter: int = 0, dtype: str = 'uint8') -> np.ndarray:
    global diffusions_per_image

    if img == None:
        img = GetGaussianNoise(255. / 2, 255. / 6 / 2, single_rgb_image_shape)
        counter += 1

    img = model.predict(img, use_multiprocessing=True)
    counter += 1
    
    is_final_output = counter >= diffusions_per_image
    if is_final_output:
        img = img.reshape(rgb_image_shape).astype(dtype)
        return img
    else:
        img = tf.convert_to_tensor(img)
        return generate_image(model, img=img, counter=counter)

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