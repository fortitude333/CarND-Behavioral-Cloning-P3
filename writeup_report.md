# **Behavioral Cloning** 

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Normal Image"
[image2]: ./examples/center_flipped.jpg "Flipped Image"

---

### Model Architecture and Training Strategy

#### 1. The first architecture and development

At first LeNet was used and tested but the results were not up to the mark, the car would not be able to handle recovery behaviour.
I did not record data for recovery behaviour specifically instead I decided to train recovery only on the basis of the left and right camera images and hence decided to follow the NVIDIA's architecture.

#### 2. The model has been adapted from NVIDIA's project

Network architecture
* conv - 5x5
* conv - 5x5
* conv - 5x5
* conv - 3x3
* conv - 3x3
* Fully connected - 1164
* Fully connected - 100
* Fully connected - 50
* Fully connected - 10

The model includes RELU layers to introduce nonlinearity (code line 116-120), and the data is normalized in the model using a Keras lambda layer (code line 114).

#### 3. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 123,125). 

The model was trained and validated on different data sets to ensure that the model was not overfitting.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 4. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 131).

#### 5. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a just center lane driving and used the left and right camera images to simulate recovery driving.

For details about how I created the training data, see the next section. 

#### 6. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

To augment the data set, I also flipped images and angles thinking that this would help in generalising as the motion on the track was generally leftward movement. For example, here is an image that has then been flipped:

![alt text][image1]
![alt text][image2]

To simulate recovery driving, I used the left and right camera images with an angle correction of 0.2.

After the collection process, I had 14660 number of data points. I then preprocessed this data by normalisation the images and zero-centering the mean.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4. I used an adam optimizer so that manually training the learning rate wasn't necessary.
