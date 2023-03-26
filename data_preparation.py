# Basic imports
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import glob 
import os
import sys
from tqdm import tqdm, tqdm_pandas
import scipy
from scipy.stats import skew
import librosa
import librosa.display
import json
import joblib
from sklearn import preprocessing
from augmentation import *

# constants definition
sampling_rate=44100
audio_duration=2.5
n_mfcc = 30
n_melspec = 60
data_all=[]
# The Audio Transformations for data augmentation
transformations=[
TimeShifting(p=1.0),CutOut(p=2.0),Gain(p=1.0,max_gain_in_db=-800,min_gain_in_db=-900),
AddGaussianNoise(p=1.0),AddGaussianNoise(p=15.0),PitchShift(p=4.0,n_steps=4),
PitchShift(p=1.0,n_steps=4),StretchAudio(p=1.0),StretchAudio(p=2.0),StretchAudio(p=1.0),
SpeedTuning(p=1.0),SpeedTuning(p=2.0), TimeShifting(p=1.0)
]

def speedNpitch(data,transform):
    """
    Data augmentation
    """
    data=transform(data=data)['data']
    return data 



def prepare_data(df, n, aug, mfcc,transform):
    
    '''
    2. Extracting the MFCC feature as an image (Matrix format).  
    '''    
    
    X = np.empty(shape=(df.shape[0], n, 216, 1))
    input_length = sampling_rate * audio_duration
    
    cnt = 0
    for fname in tqdm(df.Path):
        file_path = fname
        data, _ = librosa.load(file_path, sr=sampling_rate
                               ,res_type="kaiser_fast"
                               ,duration=2.5
                               ,offset=0.5
                              )

        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")

        # Augmentation? 
        if aug == 1:
#             for i in l :
            data = speedNpitch(data,transform)
        
        # which feature?
        if mfcc == 1:
            # MFCC extraction 
            MFCC = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc)
            MFCC = np.expand_dims(MFCC, axis=-1)
            X[cnt,] = MFCC
            
        else:
            # Log-melspectogram
            melspec = librosa.feature.melspectrogram(data, n_mels = n_melspec)   
            logspec = librosa.amplitude_to_db(melspec)
            logspec = np.expand_dims(logspec, axis=-1)
            X[cnt,] = logspec
            
        cnt += 1
    
    return X


def read_the_file(path,aug_arg,data_extraction):
    '''
    read the csv file containing the data 
    
    '''
    flat_list = []
    ref = pd.read_csv(path)
    le = preprocessing.LabelEncoder()
    ref['labels']=le.fit_transform(ref['Emotions'])
    ref=ref[ref['Emotions']!='calm']
    for transform in transformations:
        mfcc = prepare_data(ref, n = n_mfcc, aug = aug_arg, mfcc = data_extraction, transform=transform)
        data_all.append(mfcc)
    i=0
    labels=pd.DataFrame(ref.labels)
    labels0=pd.DataFrame(ref.labels)
    while i<12:
    #     print(i)
        labels=pd.concat([labels,labels0],ignore_index=True, sort=False)

        i=i+1
        
    
    return data_all,labels
        
def main():
    
    print('Initializing Data Preparation Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_of_the_csv_file', default="/mnt/workspace/segni/data_path_all.csv")
    parser.add_argument('--output_data_directory', default='./data')
    parser.add_argument('--augment_data', default=0)
    parser.add_argument('--data_extraction_type', default=1)
    a = parser.parse_args()
    path=a.path_of_the_csv_file
    aug_arg=a.augment_data
    save_dir=a.output_data_directory
    data_extraction=a.data_extraction_type
    data,labels=read_the_file(path,aug_arg,data_extraction)
    X_name = 'data.joblib'
    y_name = 'labels.joblib'
    savedX = joblib.dump(data, os.path.join(save_dir, X_name))
    savedy = joblib.dump(labels, os.path.join(save_dir, y_name))

    
        

        
        
        
        
        
        
if __name__ == '__main__':
    
    main()
