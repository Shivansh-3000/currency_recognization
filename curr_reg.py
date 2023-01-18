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
