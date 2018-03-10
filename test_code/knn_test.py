# import cv2
# import numpy as np
#
# FNAME = 'digits.npz'
#
# def learningDigit():
#     img = cv2.imread('digits.png')
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
#     x = np.array(cells)
#     train = x[:,:].reshape(-1,400).astype(np.float32)
#
#     k = np.arange(10)
#     train_labels = np.repeat(k,500)[:,np.newaxis]
#
#     np.savez(FNAME,train=train,train_labels = train_labels)
#
# def loadTrainData(fname):
#     with np.load(fname) as data:
#         train = data['train']
#         train_labels = data['train_labels']
#
#     return train, train_labels
#
# learningDigit()
# train, train_labels = loadTrainData(FNAME)
#
# print(train_labels[499])

from __future__ import print_function
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import numpy as np
import imutils
import cv2

mnist = datasets.load_digits()

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data), mnist.target, test_size=0.25, random_state=42)