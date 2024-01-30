import os
from image_classification import remove_bad_images, load_data, preprocess_data, split_data, build_model, \
    train_model, evaluate_model, plot_training_performance

# Configuration and Constants
DATA_DIR = 'data'
IMAGE_EXTS = ['jpeg', 'jpg', 'bmp', 'png']
LOG_DIR = 'logs'
MODEL_DIR = 'models'
MODEL_FILE = 'better_umpire_model.keras'

"""
Main Script for Strike Zone Classifier

This script utilizes the image classification functions from the 'image_classification' module. It removes
images with invalid formats, loads and preprocesses the image data, builds and trains a convolutional neural
network model, evaluates the model, and saves it to a file.

Author: Tyler McCoy
"""


def main():
    """
    Main function for image classification.

    - Removes invalid images from the dataset.
    - Loads and preprocesses the image data.
    - Builds, trains, and evaluates a convolutional neural network model.
    - Saves the trained model to a file.

    Returns:
    None
    """
    remove_bad_images(DATA_DIR, IMAGE_EXTS)

    data = load_data(DATA_DIR)
    data = preprocess_data(data)

    train_data, val_data, test_data = split_data(data)

    model = build_model()

    log_directory = os.path.join(LOG_DIR, 'model_logs')
    history = train_model(model, train_data, val_data, log_directory)

    plot_training_performance(history)

    precision, recall, accuracy = evaluate_model(model, test_data)
    print(f'Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}')

    # Save the trained model to a file
    model.save(os.path.join(MODEL_DIR, MODEL_FILE))

    # Load the saved model (commented out for now)
    # new_model = load_model(os.path.join(MODEL_DIR, MODEL_FILE))


if __name__ == "__main__":
    main()
