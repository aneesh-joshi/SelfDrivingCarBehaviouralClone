# **Behavioral Cloning WriteUp** 
---
[//]: # (Image References)

[image1]: ./writeup_images/val_dec_inc.png "Validation Loss Decreases and Increases"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

### Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

### Are all required files submitted?
My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup_report.md` summarizing the results

### Quality of Code
#### Is the code functional?
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### Is the code usable and readable?
The code written by is mostly present in `model.py`
Here, I have 
* loaded the training data (lines 70 to 75)
* Split it into train and test (line 79)
* made a generator to provide the images on demand with a radom flip, darkening or shadow (lines 17 to 64)
* defined a keras model (lines 120 to 139)
* trained the model (lines 163 to 168)
* plotted the metrics of training and validation mean squared error (lines 178 to 181)

### Model Architecture and Training Strategy
#### Has an appropriate model architecture been employed for the task?

I experimented a lot with the architectures to get a good model. This was based on the initial assumption that my model's performance was very strongly dependent on the architecture. (an assumption which was somewhat incorrect (more on this later))

As a result, I ended up trying several different architectures:
First, I tried an architecture with fewer convolutions and more fully connected layers. However, this made the number of parameters blow up to 16 million. This was unnecessary and simply unfeasible on my 8GBs of RAM with only CPU

Searching for a better method, I decided to give transfer learning a chance. I downloaded VGG16, added fresh fully connected layers and trained on that. This made the parameters lesser but not by much; it was still in the millions. I tried freezing the VGG16 parameters, but to no avail.

Finally, I settled on using the NVIDIA architecture as it had the least parameters (348 thousand).

This exercise in architecture search made me realize the importance of keeping the number of parameters less.
I learned that adding more convolutions will reduce the feature space, resulting in lesser parameters for the first fully connected layer after the convolutions are flattened.

The NVIDIA model trained fast. This was a big advantage as I could now turn my focus to collecting and improving the training data.
It consists of:
1.) 3 5x5 convolutions with relu
2.) 2 3x3 convolutions with relu
3.) 100 Fully Connected Neurons with relu
4.) 50 Fully Connected Neurons with relu
5.) 10 Fully Connected Neurons
6.) 1 Fully Connected Neuron

Before these model layers, there is 
1.) a Cropping2D layer which crops the image to just get the bottom half
2.) a Lambda layer which normalises the input image

In a more verbose form:

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]
====================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
```

#### Has an attempt been made to reduce overfitting of the model?

For a long time, I was using the wrong metrics to evaluate my model as it trained. I was intially using validation accuracy, but this metric is incorrect for a regression problem. I kept getting a validation accuracy of 40% and I was confused why it didn't improve.Later, I moved to mean squared error which gave more  meaningful results.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 102 to 103). These data sets were made by splitting the total training samples into a train-test split of 80-20 respectively.
The model was then tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I also monitored the metrics of the training to ensure that overfitting hasn't occured. For example, while experimenting, I would make the model run for several epochs. Typically, the validation loss would decrease upto a certain epoch length and begin increasing again. The next time, I would train only upto that epoch.
[image1]

I didn't feel the need to add dropout and felt that the model performed worse with it.

#### Have the model parameters been tuned appropriately?

The model was uses the Adadelta optimiser. As a result, I didn't need to tune the learnig rate.
Other than model architecture, the only hyperparameter was number of epochs.
I tuned this by studying the validation loss metrics and stopping training when it indicated over fitting.

#### Is the training data chosen appropriately?

I made several data sets until I got one which works. [TODO add image]
The finalised data set was made by including:
[TODO see how to make a list]
1.) A normal lap

2.) A lap in the opposite direction

3.) A lap with only recording the recovery scenarios

The recovery scenarios were made by 
* going to the edge of the road
* stopping the car
* turning the wheel
* recording the car in a stationary position for a while
* moving the vehicle into the center
* stopping recording

The reason behind stopping the car was to ensure that several frames were captured with the recovery conditions.

The initial model I trained only used the center image. However, it didn't perform well. When I added the right and left offset images with the corrected steering angle, the model became a lot better.
I suppose the reason is that:
The whole problem of driving on the track can be solved simply by turning left when you heading towards the right edge, turning right when you're heading towards the left track and doing nothing otherwise.
Using only the center images don't give a good idea of the right and left edge cases. But the same left steer when seen from the right camera strongly reinforces that the steering angle should be sharply to the left.

**Data Augmentation:**

I made several augmentations in the generator to make it more general:
1. Flipping:
I made the generator flip the image and negate the steering angle with a probablity of `FLIP_PROB` which was set to 0.25
This was done to ensure the model wouldn't be biased towards turning left (as the track was mostly leftwards).
2. Shadows:
My initial models were easily distracted by the tree shadows. To tackle this, I introduced `SHADOW_PROB` which was set to 0.5
This added a random rectangular shadow patch in the images with a probablity of 50%
[TODO] add sample images
3. Image Darkening:
When I tried my model on the challenge track, I saw that the model often had to handle low lighting condition. I introduced `DARKEN_PROB` which was set to 0.4 to randomly darken the image.
[TODO add image]


examples of images from the dataset must be included. [TODO]