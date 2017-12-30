# **Behavioral Cloning** 

## Writeup Template

### Final Video :
#### Model trained using only center images : https://youtu.be/u7TOier5U_k
#### Model trained using center, left and right images : https://youtu.be/Gy6C2jLZZ6U

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
* model.py containing the script to create and train the model
* BehavioudCloning-final.ipynb containing desccription, images and code of the whole pipeline
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py and BehaviourCloning-final.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes using ma pooling of 2x2 and "relu" as activation function  (model.py lines 76-89) 

The model includes RELU layers to introduce nonlinearity (code line 79-83), and the data is normalized in the model using a Keras lambda layer (code line 77). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layer (0.2) between Dense layer of size 50 and Dense layer of size 10,  in order to reduce overfitting (model.py lines 87). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 91).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet Architecture. I trained the model using 10 epochs. I found that even with very good data collection, there were issues of car going off the track especially where there is curve and open road on the side. 

Then, I used the Nvidia Model. I saw the training error decreasing till 3 epochs. I used this model and saw that now the car was able to recover and drove properly when used left and right image correction.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding the dropout layer between the dense layer of size 100 and 50. I also added the maz pooling layer in the first three CNN layers. 

I also used the Keras, cropping2D to crop the part of the image which are not useful. It also resulted in training the model faster.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially on the curve and open road. To improve the driving behavior in these cases, I include more training dataset to train the model to recover from such situation. More detail in section collection training dataset.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 76-89) using Nvidia Architecture [refered from the lecture video] consisted of a five convolution neural network and four fully connected layer.

Here is a visualization of the architecture
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_1 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        


=================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0


#### 3. Creation of the Training Set & Training Process
Following steps where taken to generate training dataset :
1. Recorded two laps on track one using center lane driving. I ensured that the car stayed in the center of the road as much as possible.
2. To enable the vehicle from recovering from bad situation like going out of the track, I recorded the vehicle recovering from the left side and right side of the road moving back to the center. This was very helpful as show in Video :  https://youtu.be/Gy6C2jLZZ6U
3. I tried to collect more training data where there are curves in the path. Turning the car smoothly on these paths.
4. To generalize the model, I repeated the above steps for track 2 to collect enough training data so that it doesn't overfit.
5. In my model.py, I flipped the center, right, left images to augment the training dataset. This helped me in dealing with the left turn or rigght turn bias. The steering angle was negated for each image flip. 


After the collection process, I had 18,515 number of data points. For the preprocessing step, I normalized the data by using Kera's Lambda Layer. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model.The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by line 92 in model.py I used an adam optimizer so that manually training the learning rate wasn't necessary.

I trained two models :
1. First with only center images
2. Second with all center, left and right images. I found that using left and right image with correction helped the model to remain in center as much as possible.

Both models video link is posted in the top of the template. 
