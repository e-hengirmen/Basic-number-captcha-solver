import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import numpy as np
import os

# check data size ranges
'''sample_size=len(os.listdir("digits"))


n_low=m_low=low=9999
n_high=m_high=high=0
average_n=average_m=average=0
for filename in os.listdir("digits"):
    img=cv2.imread("digits/"+filename,cv2.IMREAD_GRAYSCALE)
    n,m=img.shape
    average_m+=m
    average_n+=n
    average+=n*m
    low=min(low,n*m)
    n_low=min(n_low,n)
    m_low=min(m_low,m)
    high=max(high,n*m)
    n_high=max(n_high,n)
    m_high=max(m_high,m)
average_n/=sample_size
average_m/=sample_size
average/=sample_size
print("nlimits=",(n_low,n_high),"mlimits=",(m_low,m_high),"n*mlimits=",(low,high),"average n,m=",average_n,average_m,average)'''
#nlimits= (19, 26) mlimits= (13, 15) n*mlimits= (260, 390) average n,m= 22.4547 14.9942 336.6911666666667





# Load the dataset
data_dir = "digits"
img_size = (23, 15) # fixed size for resizing the images
X = []
y = []
for filename in os.listdir(data_dir):
    img_path = os.path.join(data_dir, filename)
    img = load_img(img_path, target_size=img_size,color_mode = "grayscale")
    img = img_to_array(img)
    X.append(img)
    y.append(int(filename.split("-")[1].split(".")[0]))

X = np.array(X)
y = np.array(y)
print(X.shape,y.shape)

# Normalize+split
X = X / 255.0 # normalize pixel values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=333)

# Define the model
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size[0], img_size[1],1)), #1 for grayscale it turns out convs needs an additional dimension for filter channels
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the test set
train_loss, train_acc = model.evaluate(X_train, y_train)
test_loss, test_acc = model.evaluate(X_test, y_test)

print("Train accuracy:", train_acc,1-train_acc)
print("Test  accuracy:", test_acc,1-test_acc)

#saving the model for later use
model.save("model/captcha-CNN-1",overwrite=True)


y_pred = model.predict(X)
y_pred = np.argmax(y_pred, axis=1)

# Check if the predicted labels match the true labels
y_pred = model.predict(X)
y_pred = np.argmax(y_pred, axis=1)


if not os.path.exists("missclassified"):
    os.mkdir("missclassified")
for i in range(len(y)):
    if y_pred[i] != y[i]:
        # Save the misclassified image along with its predicted and true labels
        filename = f"{i}#True_{y[i]}###predicted_{y_pred[i]}.jpg"
        filepath = os.path.join("missclassified", filename)
        tf.keras.preprocessing.image.save_img(filepath, X[i])
