from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Lambda, Cropping2D, MaxPooling2D

import csv, time
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import datetime

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        # shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            for batch_sample in batch_samples:
                name = 'data/data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(cv2.flip(center_image, 0))
                angles.append(center_angle*-1.0)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# data_path = 'data/data/'

# lines = []

# with open(data_path + 'driving_log.csv') as csvfile:
# 	reader = csv.reader(csvfile)

# 	for line in reader:
# 		lines.append(line)

# images, measurements = [], []
# aug_images, aug_measurements = [], []

# for line in lines[1:]:
# 	image = mpimg.imread(data_path + line[0])
# 	images.append(image)
# 	measurements.append(float(line[3]))

# 	images.append(cv2.flip(image))
# 	measurements.append(-1.0 * float(line[3]))

samples = []
with open('data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for i, line in enumerate(reader):
        if i != 0:
            samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*2, validation_data=validation_generator,
            nb_val_samples=len(validation_samples)*2, nb_epoch=3)

model.save('modeel' + '.h5')
# model.save('mod' + str(datetime.datetime.now()) + '.h5')

