import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import datasets, layers, models


def cnn():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()

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

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print(test_acc)

    plt.show()


def resnet50():
    BATCH_SIZE = 32
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.rc('image', cmap='gray')
    plt.rc('grid', linewidth=0)
    plt.rc('xtick', top=False, bottom=False, labelsize='large')
    plt.rc('ytick', left=False, right=False, labelsize='large')
    plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
    plt.rc('text', color='a8151a')
    plt.rc('figure', facecolor='F0F0F0')  # Matplotlib fonts

    MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")

    # utility to display a row of digits with their predictions
    def display_images(digits, predictions, labels, title):
        n = 10

        indexes = np.random.choice(len(predictions), size=n)
        n_digits = digits[indexes]
        n_predictions = predictions[indexes]
        n_predictions = n_predictions.reshape((n,))
        n_labels = labels[indexes]

        fig = plt.figure(figsize=(20, 4))
        plt.title(title)
        plt.yticks([])
        plt.xticks([])

        for i in range(10):
            ax = fig.add_subplot(1, 10, i + 1)
            class_index = n_predictions[i]

            plt.xlabel(classes[class_index])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(n_digits[i])

    # utility to display training and validation curves
    def plot_metrics(metric_name, title, ylim=5):
        plt.title(title)
        plt.ylim(0, ylim)
        plt.plot(history.history[metric_name], color='blue', label=metric_name)
        plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)

    (training_images, training_labels), (validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()

    display_images(training_images, training_labels, training_labels, "Training Data")

    display_images(validation_images, validation_labels, validation_labels, "Training Data" )

    def preprocess_image_input(input_images):
        input_images = input_images.astype('float32')
        output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
        return output_ims

    train_X = preprocess_image_input(training_images)
    valid_X = preprocess_image_input(validation_images)

    '''
    Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
    Input size is 224 x 224.
    '''

    def feature_extractor(inputs):
        feature_extractor_ = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                                                  include_top=False,
                                                                  weights='imagenet')(inputs)
        return feature_extractor_

    '''
    Defines final dense layers and subsequent softmax layer for classification.
    '''

    def classifier(inputs):
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
        return x

    '''
    Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
    Connect the feature extraction and "classifier" layers to build the model.
    '''

    def final_model(inputs):
        resize = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)

        resnet_feature_extractor = feature_extractor(resize)
        classification_output = classifier(resnet_feature_extractor)

        return classification_output

    '''
    Define the model and compile it. 
    Use Stochastic Gradient Descent as the optimizer.
    Use Sparse Categorical CrossEntropy as the loss function.
    '''

    def define_compile_model():
        inputs = tf.keras.layers.Input(shape=(32, 32, 3))

        classification_output = final_model(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=classification_output)

        model.compile(optimizer='SGD',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    model = define_compile_model()

    model.summary()

    EPOCHS = 3
    history = model.fit(train_X, training_labels, epochs=EPOCHS, validation_data=(valid_X, validation_labels),
                        batch_size=64)

    loss, accuracy = model.evaluate(valid_X, validation_labels, batch_size=64)

    plot_metrics("loss", "Loss")
    plot_metrics("accuracy", "Accuracy")

    probabilities = model.predict(valid_X, batch_size=64)
    probabilities = np.argmax(probabilities, axis=1)

    display_images(validation_images, probabilities, validation_labels, "Bad predictions indicated in red.")


if __name__ == '__main__':
    resnet50()
    # cnn()
