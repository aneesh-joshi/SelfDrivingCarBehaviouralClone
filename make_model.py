import keras

from pprint import pprint

from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Lambda, Cropping2D, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping

import csv, cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import model_from_json

from keras.layers.advanced_activations import ELU
from utils import darken_image, add_shadow

def generator(samples, batch_size=32, steering_coef=0.2, flip_prob=0, shadow_prob=0, darken_prob=0):
    num_samples = len(samples)
    while True:
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                for i in range(3):

                    name = batch_sample[i].split('/')[-1]

                    center_image = mpimg.imread(name)

                    if i == 0:
                        center_angle = float(batch_sample[3])
                    elif i == 1:
                        center_angle = float(batch_sample[3]) + steering_coef
                    else:
                        center_angle = float(batch_sample[3]) - steering_coef


                    # RANDOM FLIP
                    if flip_prob > 0:
                        if np.random.random() < flip_prob:
                            center_image = cv2.flip(center_image, 1)
                            center_angle = -1. * center_angle

                    made_shadow = False
                    # SHADOW ADD
                    if shadow_prob > 0:
                        if np.random.random() < shadow_prob:
                            made_shadow = True
                            center_image = add_shadow(center_image)
                            
                            
                    # MAKE DARK
                    if darken_prob > 0 and made_shadow == False:
                        if np.random.random() < darken_prob:
                            center_image = darken_image(center_image)

                    images.append(center_image)
                    angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

# DATA PATH ===============================================================
path = 'data/data/'
path = 'data/mouse/'
path = 'data/mouse_data_almost_correct/'
path = 'data/last_hope/'
path = 'data/new_hope_w_corrections_reverse/'
path = 'data/mouse_data/'
path = 'data/mouse_data_w_reverse/'
path = 'data/finale/'
# =========================================================================

samples = []
with open(path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for i, line in enumerate(reader):
        if i != 0:
            samples.append(line)

samples = shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('We have %d samples to train on ' % len(samples))
print('We have %d training samples ' % len(train_samples))
print('We have %d validation samples samples ' % len(validation_samples))

print('After adding right and left augmentation:')

print('We have %d samples to train on ' % (len(samples)*3))
print('We have %d training samples ' % (len(train_samples)*3))
print('We have %d validation samples samples ' % (len(validation_samples)*3))

n_train_samples = len(train_samples)*3
n_validation_samples = len(validation_samples)*3

# PARAMS ====================================================================
BATCH_SIZE = 32
STEERING_COEFF = 0.35
FLIP_PROB = 0.25
SHADOW_PROB = 0.5
DARKEN_PROB = 0.4

# MAKE GENERATOR ============================================================
train_generator = generator(train_samples, BATCH_SIZE, STEERING_COEFF, FLIP_PROB, SHADOW_PROB, DARKEN_PROB)
validation_generator = generator(validation_samples, BATCH_SIZE, STEERING_COEFF, FLIP_PROB)

check_samples = False

if check_samples:
    for train_sample in train_generator:
        x, y = train_sample
        for i in range(len(y)):
            plt.imshow(x[i])
            plt.title(str(y[i]))
            plt.show()


# HYPERPARAMETERS ==========================================================
learning_rate = 1e-4
activation = 'relu'

# DEFINE MODEL=================================================================
model = Sequential()

# PREPROCESSING =============================================
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# FEATURE EXTRACTION =========================================
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation=activation))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation=activation))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation=activation))

model.add(Convolution2D(64, 3, 3, activation=activation))
model.add(Convolution2D(64, 3, 3, activation=activation))

# CLASSIFIER============================================================
model.add(Flatten())
model.add(Dense(100, activation=activation))
model.add(Dense(50, activation=activation))
model.add(Dense(10))
model.add(Dense(1))

model.summary()



# RETRAIN MODEL =====================================================

model_name = 'finale'

retrain_model = False
if retrain_model:
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_name + ".h5")
    print("Loaded model from disk")

    model = loaded_model

# COMPILE MODEL =======================================================
model.compile(  loss='mse',
                optimizer='adadelta',
                metrics=['mse']
                )

# TRAIN MODEL ==========================================================
history = model.fit_generator( train_generator,
                     samples_per_epoch=n_train_samples,
                     validation_data=validation_generator,
                     nb_val_samples=n_validation_samples,
                     nb_epoch=4,
                     verbose=1)

model.save(model_name + '.h5')
print('MODEL SAVED')

# model_json = model.to_json()
# with open(model_name + ".json", "w") as json_file:
#     json_file.write(model_json)

# PLOT METRICS ========================================================
print(history.history['val_mean_squared_error'])
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
# # plt.plot(history.history['mean_absolute_error'])
# # plt.plot(history.history['mean_absolute_percentage_error'])
# # plt.plot(history.history['cosine_proximity'])
plt.show()

# plt.plot(history.history['val_mean_squared_error'])
# # plt.plot(history.history['val_mean_absolute_error'])
# # plt.plot(history.history['val_mean_absolute_percentage_error'])
# # plt.plot(history.history['val_cosine_proximity'])
# plt.show()



# plt.plot(history.history['mse'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
