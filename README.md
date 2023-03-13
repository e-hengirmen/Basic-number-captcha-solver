# Basic-number-captcha-solver with tensorflow CNN

This is a real-life implementation of [CNN](https://www.tensorflow.org/tutorials/images/cnn?hl=tr) on Python 3, Keras, and TensorFlow. We work in 2 main steps using [`Preprocess.py`](Preprocess.py) which creates artifactless digit images and [`train.py`](train.py) which trains and saves the CNN model.

Lastly the digit images are fed to CNN to train it.



# CNN Model Description
I trained a CNN using TensorFlow to classify images of captcha digits after digit extraction. The model consists of 2 convolutional layers and pooling layers followed by a fully connected layer. I trained the model on a dataset of 10000 captcha images which was later on divided into 60000 digit images and achieved a test accuracy of 99.8% on a 12000 digit img test set.

Train and test accuracy of the model are:  
* Train accuracy: 0.9993541836738586  
* Test  accuracy: 0.9980000257492065  

Note: Some of the images of activation neuron responses can be seen below the 4 Preprocess steps

# Steps

To help with understanding the digit extractor and the model we show the visualization of their steps

The [`Preprocess.py`](Preprocess.py) works in 4 steps to create digit images which are shown below`with the model activations at the end

## 1 Artifact removel
In first step we clear the artifacts of the captcha images which was collected using [`SampleCollecter.py`](SampleCollecter.py)

Original Image            |  After artifact removel
:-------------------------:|:-------------------------:
<img src="captchas/1901-497350.jpg" width="300">  |  <img src="visualized_steps/1-artifacts_removed/1901-497350.jpg" width="300">
<img src="captchas/1902-202236.jpg" width="300">  |  <img src="visualized_steps/1-artifacts_removed/1902-202236.jpg" width="300">

## 2 Clipping
Later on the artifactless image is clipped to bound the digits of the captcha

After artifact removel            |  Clipped
:-------------------------:|:-------------------------:
<img src="visualized_steps/1-artifacts_removed/1901-497350.jpg" width="300">  |  <img src="visualized_steps/2-clipped/1901-497350.jpg" width="300">
<img src="visualized_steps/1-artifacts_removed/1902-202236.jpg" width="300">  |  <img src="visualized_steps/2-clipped/1902-202236.jpg" width="300">

## 3 K_means
Than K-means is used to find the centers of the each digit

Clipped            |  Centroids
:-------------------------:|:-------------------------:
<img src="visualized_steps/2-clipped/1901-497350.jpg" width="300">  |  <img src="visualized_steps/3-k_means/1901-497350.jpg" width="300">
<img src="visualized_steps/2-clipped/1902-202236.jpg" width="300">  |  <img src="visualized_steps/3-k_means/1902-202236.jpg" width="300">

## 4 Digit Extraction
Lastly we create digit images using the centroids found from k-means

Centroids            |  Digits
:-------------------------:|:-------------------------:
<img src="visualized_steps/3-k_means/1901-497350.jpg" width="300">  |  <img src="visualized_steps/digits/1901,1-4.jpg" width="50"> <img src="visualized_steps/digits/1901,2-9.jpg" width="50"> <img src="visualized_steps/digits/1901,3-7.jpg" width="50"> <img src="visualized_steps/digits/1901,4-3.jpg" width="50"> <img src="visualized_steps/digits/1901,5-5.jpg" width="50"> <img src="visualized_steps/digits/1901,6-0.jpg" width="50">
<img src="visualized_steps/3-k_means/1902-202236.jpg" width="300">  |  <img src="visualized_steps/digits/1902,1-2.jpg" width="50"> <img src="visualized_steps/digits/1902,2-0.jpg" width="50"> <img src="visualized_steps/digits/1902,3-2.jpg" width="50"> <img src="visualized_steps/digits/1902,4-2.jpg" width="50"> <img src="visualized_steps/digits/1902,5-3.jpg" width="50"> <img src="visualized_steps/digits/1902,6-6.jpg" width="50">

## 5 Activation Images
Here are generated activation images for the last convolutional layer of the CNN by extracting the output of the layer and visualizing it as an image. These activation images show the response of some of the filters in the last convolutional layer to 2 images.

Extracted Digit            |  activation 10 |  activation 18 |  activation 26 |  activation 29
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
<img src="visualized_steps/digits/1962,6-8.jpg" width="50"> | <img src="visualized_steps/5-activations/activations8/activation_conv2d_10.jpg" width="50"> | <img src="visualized_steps/5-activations/activations8/activation_conv2d_18.jpg" width="50">| <img src="visualized_steps/5-activations/activations8/activation_conv2d_26.jpg" width="50">| <img src="visualized_steps/5-activations/activations8/activation_conv2d_29.jpg" width="50">
<img src="visualized_steps/digits/194,1-1.jpg" width="50">  | <img src="visualized_steps/5-activations/activations1/activation_conv2d_10.jpg" width="50">| <img src="visualized_steps/5-activations/activations1/activation_conv2d_18.jpg" width="50">| <img src="visualized_steps/5-activations/activations1/activation_conv2d_26.jpg" width="50">| <img src="visualized_steps/5-activations/activations1/activation_conv2d_29.jpg" width="50">

