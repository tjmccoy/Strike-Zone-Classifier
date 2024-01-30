import os
import cv2
import imghdr
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

"""
Strike Zone Classifier

This script performs image classification using TensorFlow and Keras. It includes functions for data preprocessing,
model building, training, evaluation, and visualization. The script also removes images with invalid formats from
the dataset and showcases the training performance using TensorBoard.

Author: Tyler McCoy
"""

# Configuration and Constants
DATA_DIR = 'data'
IMAGE_EXTS = ['jpeg', 'jpg', 'bmp', 'png']
LOG_DIR = 'logs'
MODEL_DIR = 'models'
MODEL_FILE = 'better_umpire_model.keras'

# Avoid Out-of-Memory (OOM) errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)  # Limits memory growth per GPU


def remove_bad_images(data_directory, valid_extensions):
    """
        Remove images with invalid formats from the dataset.

        Parameters:
        - data_directory (str): The path to the dataset directory.
        - valid_extensions (list): List of valid image file extensions.

        Returns:
        None
    """
    for image_class in os.listdir(data_directory):
        for image in os.listdir(os.path.join(data_directory, image_class)):
            image_path = os.path.join(data_directory, image_class, image)
            try:
                # Check image format
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)

                # If the image format is not in the valid extensions, remove it
                if tip not in valid_extensions:
                    print(f'Image not in ext list {image_path}')
                    os.remove(image_path)
            except Exception as e:
                print(f'Issue with image {image_path}')


def load_data(directory):
    """
       Load image data from a directory using TensorFlow's image_dataset_from_directory.

       Parameters:
       - directory (str): The path to the dataset directory.

       Returns:
       tf.data.Dataset: Image dataset.
    """
    return tf.keras.utils.image_dataset_from_directory(directory)


def preprocess_data(data):
    """
       Preprocess image data by scaling pixel values to the range [0, 1].

       Parameters:
       - data (tf.data.Dataset): Image dataset.

       Returns:
       tf.data.Dataset: Preprocessed image dataset.
    """
    # Scale pixel values of images to the range [0, 1]
    data = data.map(lambda x, target_y: (x / 255, target_y))
    return data


def split_data(data, train_ratio=0.7, val_ratio=0.2):
    """
       Split image data into training, validation, and test sets.

       Parameters:
       - data (tf.data.Dataset): Image dataset.
       - train_ratio (float): Ratio of data used for training.
       - val_ratio (float): Ratio of data used for validation.

       Returns:
       (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset): Tuple of training, validation, and test datasets.
    """
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio) + 1
    test_size = int(len(data) * (1 - train_ratio - val_ratio)) + 1

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)

    return train, val, test


def build_model():
    """
        Build a convolutional neural network model for image classification.

        Returns:
        tf.keras.models.Sequential: Compiled model.
    """
    model = Sequential()
    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return model


def train_model(model, train_data, val_data, log_dir):
    """
        Train the image classification model.

        Parameters:
        - model (tf.keras.models.Sequential): Compiled model.
        - train_data (tf.data.Dataset): Training dataset.
        - val_data (tf.data.Dataset): Validation dataset.
        - log_dir (str): Directory for TensorBoard logs.

        Returns:
        tf.keras.callbacks.History: Training history.
    """
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    history = model.fit(train_data, epochs=20, validation_data=val_data, callbacks=[tensorboard_callback])
    return history


def evaluate_model(model, test_data):
    """
       Evaluate the image classification model on the test dataset.

       Parameters:
       - model (tf.keras.models.Sequential): Trained model.
       - test_data (tf.data.Dataset): Test dataset.

       Returns:
       (float, float, float): Precision, recall, and accuracy metrics.
    """
    precision_metric = Precision()
    recall_metric = Recall()
    accuracy_metric = BinaryAccuracy()

    for batch in test_data.as_numpy_iterator():
        x, y = batch
        yhat = model.predict(x)
        precision_metric.update_state(y, yhat)
        recall_metric.update_state(y, yhat)
        accuracy_metric.update_state(y, yhat)

    precision = precision_metric.result().numpy()
    recall = recall_metric.result().numpy()
    accuracy = accuracy_metric.result().numpy()

    return precision, recall, accuracy


def plot_training_performance(history):
    """
        Plot the training performance using Matplotlib.

        Parameters:
        - history (tf.keras.callbacks.History): Training history.

        Returns:
        None
    """
    fig = plt.figure()
    plt.plot(history.history['loss'], color='teal', label=' Training Loss')
    plt.plot(history.history['val_loss'], color='orange', label='Validation Loss')
    plt.suptitle('Loss', fontsize=20)
    plt.legend(loc='upper right')
    plt.show()

    fig2 = plt.figure()
    plt.plot(history.history['accuracy'], color='teal', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], color='orange', label='Validation Accuracy')
    plt.suptitle('Accuracy', fontsize=20)
    plt.legend(loc='upper left')
    plt.show()
