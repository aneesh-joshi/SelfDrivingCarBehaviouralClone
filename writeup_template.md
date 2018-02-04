# **Behavioral Cloning WriteUp** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup_report.md` summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
The specific code can be seen from [TODO]

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the NVIDIA model for making my regression model.
It consists of:
1.) 3 5x5 convolutions
2.) 2 3x3 convolutions
3.) 100 Fully Connected Neurons
4.) 50 Fully Connected Neurons
5.) 10 Fully Connected Neurons
6.) 1 Fully Connected Neuron

More details on this later

#### 2. Attempts to reduce overfitting in the model

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

For a long time, I was using the wrong metrics to evaluate my model as it trained. I was intially using validation accuracy, but this metric is incorrect for a regression problem. I kept getting a validation accuracy of 40% and I was confused why it didn't improve.
Later, I moved to mean squared error which gave more  meaningful results.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). These data sets were made by splitting the total training samples into  a train-test split of 80-20 respectively.
The model was then tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I also monitored the metrics of the training to ensure that overfitting hasn't occured. For example, while experimenting, I would make the model run for several epochs. Typically, the validation loss would decrease upto a certain epoch length and begin increasing again. The next time, I would train only upto that epoch.
[TODO insert image]

#### 3. Model parameter tuning

The model used an Adadelta optimizer, so the learning rate was not tuned manually (model.py line [TODO]).

#### 4. Appropriate training data

I made several data sets until I got one which works. [TODO add image]
The finalised data set was made by including:

1.) A normal lap

2.) A lap in the opposite direction

3.) A lap with only recording the recovery scenarios


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I experimented a lot with the architectures to get a good model. This was based on the initial assumption that my model's performance was very strongly dependent on the architecture. (an assumption which was somewhat incorrect (more on this later))

As a result, I ended up trying several different architectures:
First, I tried an architecture with fewer convolutions and more fully connected layers. However, this made the number of parameters blow up to 16 million. This was unnecessary and simply unfeasible on my 8GBs of RAM with only CPU

Searching for a better method, I decided to give transfer learning a chance. I downloaded VGG16, added fresh fully connected layers and trained on that. This made the parameters lesser but not by much; it was still in the millions. I tried freezing the VGG16 parameters, but to no avail.

Finally, I settled on using the NVIDIA architecture as it had the least parameters (348 thousand).

This exercise in architecture search made me realize the importance of keeping the number of parameters less.
I learned that adding more convolutions will reduce the feature space, resulting in lesser parameters for the first fully connected layer after the convolutions are flattened.

The NVIDIA model trained fast. This was a big advantage as I could now turn my focus to collecting and improving the training data.

#### 2. Final Model Architecture

The final model architecture (model.py lines [TODO]) 
It is represented below:

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


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)
[TODO]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
