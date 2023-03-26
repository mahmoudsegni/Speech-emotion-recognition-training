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
opt = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-07, decay=0.0)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

if __name__ == '__main__':
    # Create a transformer for image augmentation
    train_tfm = A.Compose([

                    A.ShiftScaleRotate(shift_limit = 0.1,
                                        scale_limit = 0.1,
                                        rotate_limit = 30),
                    A.Normalize(mean=[0.1307], std=[0.3081])

                ])



    def valid_tfm(size):
        return A.Compose([ A.Resize(size, size) ])

    tfms = transforms.Compose([

        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.RandomAffine(degrees=(-16, 16)),
        #transforms.Normalize(mean=(0.5,), std=(0.5,))
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))

                              ])
    v_tfms = transforms.Compose([
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    label = []
    path = []
    #create the map between labels and numbers
    label_conversion = {'01': 'neutral',
                                '02': 'disgust',
                                '03': 'happy',
                                '04': 'angry',
                                '05': 'fear',
                                '06' : 'sad'
                       }



    for subdirs, dir_, files in os.walk('/mnt/workspace/segni/utils/iemocap-dataset-spectrograms/spectrograms'):
        for filenames in files:
            for emotion_number, emotions in label_conversion.items():
                if(filenames[6:8] == emotion_number):
                    path.append(os.path.join(subdirs, filenames))
                    label.append(emotions)
    df = pd.DataFrame(label, columns = ['label'])
    df = pd.concat([df,pd.DataFrame(path, columns = ['path'])],axis=1)
    # df.label.value_counts()
    # df.to_csv('spec.csv', index = False)
    data = []
    label =[]

    for lab, img in enumerate(df.path):
#         Load the image and create the labels vector
      try:
        image = load_img(img, target_size=(224, 224))
        # convert the image pixels to a numpy array 
        image = img_to_array(image) 
        # reshape data for the model
        image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        data.append(image)
        label.append(df.label[lab])
        print(lab, img)
      except:
        print("Error loading image", lab)


    #Converting lists into numpy arrays
    data = np.array(data)
    label = np.array(label)
    X=data
    y=label
#     Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=42)
    os.mkdir('vectors_data')
#     save the data in the folder
    X_name = 'X_test_specs.joblib'
    y_name = 'y_test_specs.joblib'
    save_dir = './vectors_data'

    savedX = joblib.dump(data, os.path.join(save_dir, X_name))
    savedy = joblib.dump(label, os.path.join(save_dir, y_name))

    
    
    #label encoder
    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))
    os.mkdir('labels')
    model = ResNet50(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
    for layer in model.layers[:]:
        layer.trainable = False
    
    new_model = Sequential()
    new_model.add(model)
    # new_model.add(Dropout(0.2))
    new_model.add(Flatten())
    new_model.add(Dense(1024, activation='relu'))
    # new_model.add(Dense(128, activation='relu')) 
    new_model.add(Dense( 6 , activation='softmax'))
    new_model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    
    EPOCHS = 30
    os.mkdir('models')
#      save the model into models path 
    checkpoint_filepath = './models/model_resnet50.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath= checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
#     fit the model 
    new_model.fit(X_train, y_train, batch_size=16, epochs=EPOCHS,validation_split=0.15, callbacks=[model_checkpoint_callback,es_callback])
    
    
    
     


    
    
    
    
    
