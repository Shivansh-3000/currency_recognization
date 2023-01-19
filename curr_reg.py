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
