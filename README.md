# Basic-number-captcha-solver with tensorflow CNN

This is a real-life implementation of [CNN](https://www.tensorflow.org/tutorials/images/cnn?hl=tr) on Python 3, Keras, and TensorFlow. We work in 2 main steps using [`Preprocess.py`](Preprocess.py) which creates artifactless digit images and [`train.py`](train.py) which trains and saves the CNN model.

Lastly the digit images are fed to CNN to train it.

Note: Train and test accuracy of the model are:
Train accuracy: 0.9993541836738586
Test  accuracy: 0.9980000257492065

# Step by Step 
To help with understanding the digit extractor and the model we show the visualization of their steps 

The [`Preprocess.py`](Preprocess.py) works in 4 steps to create digit images which are shown below
## 1.1 Artifact removel
In first step we clear the artifacts of the captcha images which was collected using [`SampleCollecter.py`](SampleCollecter.py)

Original Image            |  After artifact removel
:-------------------------:|:-------------------------:
<img src="captchas/1901-497350.jpg" width="300">  |  <img src="visualized_steps/1-artifacts_removed/1901-497350.jpg" width="300">
<img src="captchas/1902-202236.jpg" width="300">  |  <img src="visualized_steps/1-artifacts_removed/1902-202236.jpg" width="300">




## 1.2 Clipping
Later on the artifactless image is clipped to bound the digits of the captcha

After artifact removel            |  Clipped
:-------------------------:|:-------------------------:
<img src="visualized_steps/1-artifacts_removed/1901-497350.jpg" width="300">  |  <img src="visualized_steps/2-clipped/1901-497350.jpg" width="300">
<img src="visualized_steps/1-artifacts_removed/1902-202236.jpg" width="300">  |  <img src="visualized_steps/2-clipped/1902-202236.jpg" width="300">

## 1.3. K_means
Than K-means is used to find the centers of the each digit

Clipped            |  Centroids
:-------------------------:|:-------------------------:
<img src="visualized_steps/2-clipped/1901-497350.jpg" width="300">  |  <img src="visualized_steps/3-k_means/1901-497350.jpg" width="300">
<img src="visualized_steps/2-clipped/1902-202236.jpg" width="300">  |  <img src="visualized_steps/3-k_means/1902-202236.jpg" width="300">

## 1.4 Digit Extraction
Lastly we create digit images using the centroids found from k-means

Centroids            |  Digits
:-------------------------:|:-------------------------:
<img src="visualized_steps/3-k_means/1901-497350.jpg" width="300">  |  <img src="visualized_steps/digits/1901,1-4.jpg" width="50"> <img src="visualized_steps/digits/1901,2-9.jpg" width="50"> <img src="visualized_steps/digits/1901,3-7.jpg" width="50"> <img src="visualized_steps/digits/1901,4-3.jpg" width="50"> <img src="visualized_steps/digits/1901,5-5.jpg" width="50"> <img src="visualized_steps/digits/1901,6-0.jpg" width="50">
<img src="visualized_steps/3-k_means/1902-202236.jpg" width="300">  |  <img src="visualized_steps/digits/1902,1-2.jpg" width="50"> <img src="visualized_steps/digits/1902,2-0.jpg" width="50"> <img src="visualized_steps/digits/1902,3-2.jpg" width="50"> <img src="visualized_steps/digits/1902,4-2.jpg" width="50"> <img src="visualized_steps/digits/1902,5-3.jpg" width="50"> <img src="visualized_steps/digits/1902,6-6.jpg" width="50">
