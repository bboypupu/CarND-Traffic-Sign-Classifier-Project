## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][demo1.png]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I convert the image to YUV space and take only the Y channel to have clearer pattern with size (32, 32, 1). And then normalize the 1-channel image to 0 to 1 for better training performance.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

| Layer | Description |
| ------------- | ------------- |
| Input | 32 x 32 x 1 (Y channel in YUV space)  |
| Convolution 1 (3x3) | 1x1 stride, valid padding, Relu, outputs 30x30x32  |
| Convolution 2 (3x3) | 1x1 stride, valid padding, Relu, outputs 28x28x32  |
| Max pooling | 2x2 stride, outputs 14x14x32  |
| Dropout | keep_prob = 0.6  |
| Convolution 3 (3x3) | 1x1 stride, valid padding, Relu, outputs 12x12x64  |
| Convolution 4 (3x3) | 1x1 stride, valid padding, Relu, outputs 10x10x64  |
| Max pooling | 2x2 stride, outputs 5x5x64  |
| Dropout | keep_prob = 0.6  |
| Convolution 5 (3x3) | 1x1 stride, valid padding, Relu, outputs 3x3x128  |
| Flatten | Input 3x3x128, output 1152 |
| Fully connected | Relu, ouput 1024 |
| Fully connected | Relu, ouput 512 |
| Fully connected | Relu, ouput 43 |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Hyperparameters:
EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.0008

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 97.6%
* validation set accuracy of 98.2%
* test set accuracy of 96.3%

If an iterative approach was chosen:
1. What was the first architecture that was tried and why was it chosen? What were some problems with the initial architecture?
At first place I used the LeNet as we did in the lesson, but the result was not good enough and the problem of overfitting was not taken care of.
2. How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I tried more convolution layers with smaller filters to develope detail in the image. Meanwhile, I also added dropout layers to eliminate over fitting.
3. Which parameters were tuned? How were they adjusted and why?
I only adjusted the learning rate to have better result.
4. What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
As the answer in question 2.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][demo2.png]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction (result: 80%):

| Image                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Yield                 | Yield                                         |
| No passing            | No passing                                    | 
| Speed limit (70km/h)  | Speed limit (70km/h)                          |
| Pedestrians           | General caution                                   |
| Turn left ahead       | Turn left ahead                               |

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The only wrong answer in the 5 test image was mistaking a Pedestrinas to a General caution. However, the correct answer is still at second place with 31.1% of posibility.


| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 1.0                   | Yield                                         | 
| 0.998                 | No passing                                    |
| 0.997                 | Speed limit (70km/h)                          |
| 0.686                 | General caution                               |
| 1.0                   | Turn left ahead                               |



### 2. Identify potential shortcomings with your current solution

The recognition still can't be better than human reaction.


### 3. Suggest possible improvements to your solution

Maybe trying a different model would help, or add more augmentation scenario.

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

