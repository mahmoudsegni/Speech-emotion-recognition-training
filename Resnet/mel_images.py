import librosa
from librosa import display
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
path = "./data_spectrograms/"
folder_path = "./mel_images/"
class Spectrograms_to_emotion_folders:

  @staticmethod
  def create_folders(path, folder_path):


    label_conversion = {'01': 'neutral',
                            '02': 'happy',
                            '03': 'sad',
                            '04': 'angry',
                            '05': 'fear',
                            '06': 'disgust'}

    for emotion_number, emotions in label_conversion.items():      
        # creating folders as the name convention given in the label_conversion dictinary above
        new_folder = emotions
        new_folder_path = folder_path + new_folder + '/'
        os.mkdir(new_folder_path)                                     
        # create folders in the name of emotions, if folder are already exist then it will do nothing

    for subdir, dirs, files in os.walk(path):
        for filenames in files:
            print(filenames)
            x, sr = librosa.load (subdir+'/'+filenames, sr=22050)           # loading audio file using librosa module
            S = librosa.feature.melspectrogram(x, sr=sr, n_mels=128)                 # converting audio file to mel-spectrogram
            log_S = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
            fig1 = plt.gcf()
            plt.axis('off')
            for emotion_number, emotions in label_conversion.items():
                if(filenames[6:8] == emotion_number):
                    image_fname = filenames.split('.')[0] + '.png'
                    fig1.savefig(folder_path + emotions + '/' + image_fname, dpi=100)   # saving figure to the folder according to the emotion of audiofile



if __name__ == '__main__':
    os.mkdir('mel_images')
    Spectrograms_to_emotion_folders.create_folders(path, folder_path)
