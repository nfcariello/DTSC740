import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import datasets, layers, models


def simcnn():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()

    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=25,
                        validation_data=(test_images, test_labels))

    # Plot results
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1])
    plt.legend(loc='lower right')
    plt.show()


def resnet50():
    (training_images, training_labels), (validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()

    # Use functions from example to shape
    def preprocess_image_input(input_images):
        input_images = input_images.astype('float32')
        output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
        return output_ims

    def feature_extractor(inputs):
        feature_extractor_ = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                                                   include_top=False,
                                                                   weights='imagenet')(inputs)
        return feature_extractor_

    def classifier(inputs):
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
        return x

    def final_model(inputs):
        resize = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)

        resnet_feature_extractor = feature_extractor(resize)
        classification_output = classifier(resnet_feature_extractor)

        return classification_output

    inputs = tf.keras.layers.Input(shape=(32, 32, 3))

    classification_output = final_model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=classification_output)

    model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    train_X = preprocess_image_input(training_images)
    valid_X = preprocess_image_input(validation_images)

    history = model.fit(train_X, training_labels, epochs=25, validation_data=(valid_X, validation_labels),
                        batch_size=32)

    # Plot results
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.ylim([0.0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1])
    plt.legend(loc='lower right')
    plt.show()

def resnet50_custom():
    (training_images, training_labels), (validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()

    # Use functions from example to shape
    def preprocess_image_input(input_images):
        input_images = input_images.astype('float32')
        output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
        return output_ims

    def feature_extractor(inputs):
        feature_extractor_ = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                                                   include_top=False,
                                                                   weights='imagenet')(inputs)
        return feature_extractor_

    def classifier(inputs):
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
        return x

    def final_model(inputs):
        resize = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)

        resnet_feature_extractor = feature_extractor(resize)
        classification_output = classifier(resnet_feature_extractor)

        return classification_output

    inputs = tf.keras.layers.Input(shape=(32, 32, 3))

    classification_output = final_model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=classification_output)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    train_X = preprocess_image_input(training_images)
    valid_X = preprocess_image_input(validation_images)

    history = model.fit(train_X, training_labels, epochs=10, validation_data=(valid_X, validation_labels),
                        batch_size=64)

    # Plot results
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.ylim([0.0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1])
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    # Run the SimCNN with standards settings
    simcnn()
    # Run the ResNet50 with standard settings
    resnet50()
    # Run the ResNet50 with the optimized settings
    resnet50_custom()
