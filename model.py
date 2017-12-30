import numpy as np
from keras.models import Sequential, Model
from keras.layers import Lambda, Flatten, Dense
import csv
import os
import cv2
import sklearn
from sklearn.utils import shuffle
import matplotlib
from matplotlib import pyplot as plt
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D, Dropout
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


def process_image(source_path):
    filename = source_path.split('\\')[-1]
    img_path = 'data/IMG/' + filename
    img = cv2.imread(img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def augmented_images(car_images, steering_angles):
    #using data augmentation to create more training dataset by flipping images
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(car_images, steering_angles):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement*-1.0)
    return [augmented_images, augmented_measurements]


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])

                # create adjusted steering measurements for the side camera images
                correction = 0.2
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left, right cameras
                img_center = process_image(batch_sample[0])
                img_left = process_image(batch_sample[1])
                img_right = process_image(batch_sample[2])

                images.extend([img_center, img_left, img_right])
                angles.extend([steering_center, steering_left, steering_right])

            augmented_output = augmented_images(images, angles)
            X_train = np.array(augmented_output[0])
            y_train = np.array(augmented_output[1])
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model  using generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#Using Nvidia Model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3,  activation="relu"))
model.add(Convolution2D(64,3,3,  activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
model.save('model.h5')
