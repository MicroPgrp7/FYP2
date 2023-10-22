"""
The CINC data is provided by https://physionet.org/challenge/2017/ 
"""
from __future__ import division, print_function
import numpy as np
from config import get_config
from utils import *
import os 
import tensorflow as tf
from tensorflow.keras import backend as K

def cincData(config):
    if config.cinc_download:
      cmd = "curl -O https://archive.physionet.org/challenge/2017/training2017.zip"
      os.system(cmd)
      os.system("unzip training2017.zip")
    num = config.num
    import csv
    testlabel = []

    with open('training2017/REFERENCE.csv') as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      line_count = 0
      for row in csv_reader:
        testlabel.append([row[0],row[1]])
        #print(row[0], row[1])
        line_count += 1
      print(f'Processed {line_count} lines.')
    if num == None:
      high = len(testlabel)-1
      num = np.random.randint(1,high)
    filename , label = testlabel[num-1]
    filename = 'training2017/'+ filename + '.mat'
    from scipy.io import loadmat
    data = loadmat(filename)
    print("The record of "+ filename)
    if not config.upload:
        data = data['val']
        _, size = data.shape
        data = data.reshape(size,)
    else:
        data = np.array(data)
    return data, label



if __name__=='__main__':
  config = get_config()
  main(config)
