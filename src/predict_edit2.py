from __future__ import division, print_function
import numpy as np
from config import get_config
from utils import *
import os
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from pathlib import Path
import h5py
# Add the following import for Keras backend
from keras import backend as K

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
    
def predictByPart(data, peaks):
    classesM = ['N', 'Ventricular', 'Paced', 'A', 'F', 'Noise']
    predicted = list()
    result = ""
    counter = [0] * len(classesM)
    print("1234")
    
    # Use K.backend instead of tf.keras.backend
    #model = tf.keras.models.load_model('models/MLII-latest.hdf5')
    # Specify the path to your HDF5 file
    model= '/content/models/MLII-latest.hdf5'  # Change this to your actual file path

    # Open the HDF5 file
    with h5py.File(model, 'r') as file:
      # Access and print information about the contents of the HDF5 file
      print("Keys: %s" % list(file.keys()))
      for key in file.keys():
        print(key)
    print("12345")
    for i, peak in enumerate(peaks[3:-1]):
        total_n = len(peaks)
        start, end = peak - config.input_size // 2, peak + config.input_size // 2
        prob = model.predict(data[:, start:end])
        prob = prob[:, 0]
        ann = np.argmax(prob)
        counter[ann] += 1
        if classesM[ann] != "N":
            print("The {}/{}-record classified as {} with {:3.1f}% certainty".format(i, total_n, classesM[ann],
                                                                                        100 * prob[0, ann]))
        result += "(" + classesM[ann] + ":" + str(round(100 * prob[0, ann], 1)) + "%)"
        predicted.append([classesM[ann], prob])
        if classesM[ann] != 'N' and prob[0, ann] > 0.95:
            import matplotlib.pyplot as plt
            plt.plot(data[:, start:end][0, :, 0], )
            Path('results').mkdir(parents=True, exist_ok=True)
            plt.savefig('results/hazard-' + classesM[ann] + '.png', format="png", dpi=300)
            plt.close()

    result += "{}-N, {}-Ventricular, {}-Paced, {}-A, {}-F, {}-Noise".format(counter[0], counter[1], counter[2],
                                                                          counter[3], counter[4], counter[5])
    return predicted, result
    # ... rest of the function remains unchanged

def predict(data, label, peaks, config):
    classesM = ['N','Ventricular','Paced','A','F','Noise']
    predicted, result  = predictByPart(data, peaks)
    sumPredict = sum(predicted[x][1] for x in range(len(predicted)))
    avgPredict = sumPredict/len(predicted)
    print("The average of the predict is:", avgPredict)
    print("The most predicted label is {} with {:3.1f}% certainty".format(classesM[avgPredict.argmax()], 100*max(avgPredict[0])))
    sec_idx = avgPredict.argsort()[0][-2]
    print("The second predicted label is {} with {:3.1f}% certainty".format(classesM[sec_idx], 100*avgPredict[0][sec_idx]))
    print("The original label of the record is " + label)
    if config.upload:
      return predicted, classesM[avgPredict.argmax()], 100*max(avgPredict[0])


def main(config):
    classesM = ['N', 'Ventricular', 'Paced', 'A', 'F', 'Noise']

    if config.upload:
        data = uploadedData(file)
    else:
        data, label = cincData(config)
    data, peaks = preprocess(data, config)
    return predict(data, label, peaks, config)

if __name__ == '__main__':
    config = get_config()
    main(config)
