# import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Sequential, load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import joblib
import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import torch.optim as optim 
from torch_lr_finder import *
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy import asarray
from fastai import *
from fastai.vision import *
from sklearn.model_selection import StratifiedKFold
import glob

import soundfile as sf
import tensorflow as tf
from keras.utils import np_utils, to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    X = joblib.load('./vectors_data/X_test_specs.joblib')
    y = joblib.load('./vectors_data/y_test_specs.joblib')
    predictions = model.predict_classes(X_test)
    rounded_labels=np.argmax(y_test, axis=1)
    report = classification_report(rounded_labels, predictions)
    print(report)
    loss, acc =model.evaluate(X_test, y_test)
    print("Model accuracy: {:5.2f}%".format(100*acc))
    cm = confusion_matrix(rounded_labels, predictions)
    cm