import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
import pandas as pd
import os
import re
import random 
currency_list = os.listdir("/content/drive/MyDrive/indian_currency_new/training")
currency_file_list = []

for i in range(len(currency_list)):

  currency_file_list.append(os.listdir(str("/content/drive/MyDrive/indian_currency_new/training/" + currency_list[i])))  
  n = len(currency_file_list[i])
  print('There are', n , currency_list[i] , 'rupee images.')
from PIL import Image 
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
from skimage import io
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
from torch.utils.data import (Dataset, DataLoader)  # Gives easier dataset managment and creates mini batches
from torchvision.datasets import ImageFolder
import torchvision.models as models

w=10
h=10
fig=plt.figure(figsize=(16, 16))
columns = 4
rows = 5

for i in range(1, len(currency_list)+1):
    img = mpimg.imread(str("/content/drive/MyDrive/indian_currency_new/training/"+ currency_list[i-1] + "/"+ currency_file_list[i-1][0]))
    compose = transforms.Compose([transforms.ToPILImage(),transforms.Resize((256,256))])
    img = compose(img)
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.title(currency_list[i-1])
    plt.imshow(img)
plt.show()

import cv2
Datadir = '/content/drive/MyDrive/indian_currency_new/training'
# Datadir = 'B:/work/Data/images_original'
category = ["10", "100", "20", "200", "2000", "50", "500", "Background"]
dataset = []
size1 = 256
size2 = 256
ct = 0
for types in category:
    path = os.path.join(Datadir, types)
    for img in os.listdir(path):
        if types == '10':
            ex = 0
            ct = ct + 1
        elif types == '100':
            ex = 1
        elif types == '20':
            ex = 2
        elif types == '200':
            ex = 3
        elif types == '2000':
            ex = 4
        elif types == '50':
            ex = 5
        elif types == '500':
            ex = 6
        elif types == 'Background':
            ex = 7
        try:
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_img = cv2.resize(img_arr, (size2, size1), interpolation=cv2.INTER_AREA)
            dataset.append([new_img, ex])
        except Exception as e:
            pass
          
X = []
Y = []
print("ph1 pass")
for features, label in dataset:
    X.append(features)
    Y.append(label)
X_train = np.array(X, np.float32) / 255.
image_labels = to_categorical(Y)

mean_img = X_train.mean(axis=0)
std_dev = X_train.std(axis = 0)
X_norm = (X_train - mean_img)/ std_dev
X_norm, image_labels = shuffle(X_norm, image_labels, random_state=0)

Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(X_norm, image_labels, test_size=0.2, random_state=7)
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout,MaxPooling2D

model = Sequential()
model.add(BatchNormalization(input_shape=Xtrain.shape[1:]))
model.add(Conv2D(32, kernel_size=(3, 3), activation= 'relu',padding= 'same'))
model.add(Conv2D(32, kernel_size=(3, 3), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=(3, 3), activation= 'relu', padding='same'))
model.add(Conv2D(64, kernel_size=(3, 3), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=(3, 3), activation= 'relu', padding='same'))
model.add(Conv2D(128, kernel_size=(3, 3), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.3))

model.add(Conv2D(256, kernel_size=(3, 3), activation= 'relu', padding='same'))
model.add(Conv2D(256, kernel_size=(3, 3), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape			
model.layers[0].output_shape			
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable
early_stops = EarlyStopping(patience=3, monitor='val_acc')

model.save('model1_final_B8.h5')
