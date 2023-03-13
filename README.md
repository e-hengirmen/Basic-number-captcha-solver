# Basic-number-captcha-solver with tensorflow CNN

This is a real-life implementation of [CNN](https://www.tensorflow.org/tutorials/images/cnn?hl=tr) on Python 3, Keras, and TensorFlow. We work in 2 main steps using [`Preprocess.py`](Preprocess.py) which creates artifactless digit images and [`train.py`](train.py) which trains and saves the CNN model.
Also the [`Preprocess.py`](Preprocess.py) works in 4 steps to create digit images:
 1. First clears the artifacts of the captcha images which was collected using [`SampleCollecter.py`](SampleCollecter.py)
 2. Clips the artifactless image and bounds the digits of the captcha in 1 bounded image
 3. Uses K-means to find the center of the each digit
 4. Lastly creates digit images using the centroids found from k-means
Lastly the digit images are fed to CNN to train it.

# Step by Step 
To help with understanding the digit extractor and the model we show the visualization of their steps 

## 1.1 Artifact removel

![](captchas/1901-497350.jpg)->![](visualized_steps/1-artifacts_removed/1901-497350.jpg)

![](captchas/1902-202236.jpg)->![](visualized_steps/1-artifacts_removed/1902-202236.jpg)

## 1.2 Clipping

![](visualized_steps/1-artifacts_removed/1901-497350.jpg)->![](visualized_steps/2-clipped/1901-497350.jpg)

![](visualized_steps/1-artifacts_removed/1902-202236.jpg)->![](visualized_steps/2-clipped/1902-202236.jpg)

## 1.3. K_means

![](visualized_steps/2-clipped/1901-497350.jpg)->![](visualized_steps/3-k_means/1901-497350.jpg)

![](visualized_steps/2-clipped/1902-202236.jpg)->![](visualized_steps/3-k_means/1902-202236.jpg)

## 1.4 Digit Extraction

![](visualized_steps/3-k_means/1901-497350.jpg)->,![](visualized_steps/digits/1901,1-4.jpg),![](visualized_steps/digits/1901,2-9.jpg),![](visualized_steps/digits/1901,3-7.jpg),![](visualized_steps/digits/1901,4-3.jpg),![](visualized_steps/digits/1901,5-5.jpg),![](visualized_steps/digits/1901,6-0.jpg)

![](visualized_steps/3-k_means/1902-202236.jpg)->,![](visualized_steps/digits/1902,1-2.jpg),![](visualized_steps/digits/1902,2-0.jpg),![](visualized_steps/digits/1902,3-2.jpg),![](visualized_steps/digits/1902,4-2.jpg),![](visualized_steps/digits/1902,5-3.jpg),![](visualized_steps/digits/1902,6-6.jpg)
