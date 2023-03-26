import librosa
from librosa import display
import os
import shutil
import random
import numpy as np 
import argparse



# TRAINING_FILES_PATH = '/mnt/workspace/segni/all_data/'


class TESSPipeline:

    @staticmethod
    def create_tess_folders(path):
       
        counter = 0

        label_conversion = {'01': 'neutral',
                            '02': 'happy',
                            '03': 'sad',
                            '04': 'angry',
                            '05': 'fear',
                            '06': 'disgust'}

        for subdir, dirs, files in os.walk(path):
            for filename in files:
                if filename.startswith('OAF'): 
                    destination_path = TRAINING_FILES_PATH + 'Actor_26/'         # this is where the files of tess i.e OAF (old) in Actor_26 folder inside of TRAINING_FILES_PATH
                    old_file_path = os.path.join(os.path.abspath(subdir), filename)

                    # Separate base from extension
                    base, extension = os.path.splitext(filename)

                    for key, value in label_conversion.items():
                        if base.endswith(value):
                            random_list = random.sample(range(10, 99), 7)
                            file_name = '-'.join([str(i) for i in random_list])
                            file_name_with_correct_emotion = file_name[:6] + key + file_name[8:] + extension
                            new_file_path = destination_path + file_name_with_correct_emotion
                            shutil.copy(old_file_path, new_file_path)

                else:
                    destination_path = TRAINING_FILES_PATH + 'Actor_25/'       # this is where the files of tess i.e YAF (young) in Actor_25 folder inside of TRAINING_FILES_PATH
                    old_file_path = os.path.join(os.path.abspath(subdir), filename)

                    # Separate base from extension
                    base, extension = os.path.splitext(filename)

                    for key, value in label_conversion.items():
                        if base.endswith(value):
                            random_list = random.sample(range(10, 99), 7)
                            file_name = '-'.join([str(i) for i in random_list])
                            file_name_with_correct_emotion = (file_name[:6] + key + file_name[8:] + extension).strip()
                            new_file_path = destination_path + file_name_with_correct_emotion
                            shutil.copy(old_file_path, new_file_path)


                            
                            
                            
                            
                            
                            
class SaveePipeline:

    @staticmethod
    def create_savee_folder(path):
       
        counter = 0

        label_conversion = {'01': 'n',
                            '02': 'h',
                            '03': 'sa',
                            '04': 'a',
                            '05': 'f',
                            '06': 'd'}

        for subdir, dirs, files in os.walk(path):
            for filename in files:
              destination_path = TRAINING_FILES_PATH + 'savee/'     # this is where the files of savee audio in 'savee' folder inside of TRAINING_FILES_PATH
              old_file_path = os.path.join(os.path.abspath(subdir), filename)

              base, extension = os.path.splitext(filename)

              for key, value in label_conversion.items():
                if(((filename.split('.')[0]).split('_')[1][:-2]) == value and ((filename.split('.')[0]).split('_')[1][:-2]) != 'su'):
                  random_list = random.sample(range(10, 99), 7)
                  file_name = '-'.join([str(i) for i in random_list])
                  file_name_with_correct_emotion = file_name[:6] + key + file_name[8:] + extension
                  new_file_path = destination_path + file_name_with_correct_emotion
                  shutil.copy(old_file_path, new_file_path)                            
                            

                    
                    
class CremaPipeline:

    @staticmethod
    def create_crema_folder(path):
       
        counter = 0

        label_conversion = {'01': 'NEU',
                            '02': 'HAP',
                            '03': 'SAD',
                            '04': 'ANG',
                            '05': 'FEA',
                            '06': 'DIS'}
        

        for subdir, dirs, files in os.walk(path):
            for filename in files:
              destination_path = TRAINING_FILES_PATH + 'crema/' # this is where the files of crema audio in 'crema' folder inside of TRAINING_FILES_PATH
              old_file_path = os.path.join(os.path.abspath(subdir), filename)

              base, extension = os.path.splitext(filename)

              for key, value in label_conversion.items():
                if(filename.split('_')[2] == value):
                  random_list = random.sample(range(10, 99), 7)
                  file_name = '-'.join([str(i) for i in random_list])
                  file_name_with_correct_emotion = file_name[:6] + key + file_name[8:] + extension
                  new_file_path = destination_path + file_name_with_correct_emotion
                  shutil.copy(old_file_path, new_file_path)                    

                
                

class ravPipeline:

    @staticmethod
    def create_rav_folder(path):
       
        counter = 0

        label_conversion = {'01': '01',
                            '02': '03',
                            '03': '04',
                            '04': '05',
                            '05': '06',
                            '06': '07'}
        

        for subdir, dirs, files in os.walk(path):
            for filename in files:
                if(filename[6:8] != '02' and filename[6:8] != '08'):
                    destination_path = TRAINING_FILES_PATH + 'RT_6_/' # this is where the files of ravdess in RT_6 folder inside of TRAINING_FILES_PATH
                    old_file_path = os.path.join(os.path.abspath(subdir), filename)

                    base, extension = os.path.splitext(filename)

                    for key, value in label_conversion.items():
                        if(filename[6:8] == value):
                            file_name_with_correct_emotion = filename[:6] + key + filename[8:] + extension
                            new_file_path = destination_path + file_name_with_correct_emotion
                            shutil.copy(old_file_path, new_file_path)                
                
                
              
                
                
def main():
    print('Start data Processing..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--TESS_ORIGINAL_FOLDER_PATH', default=None)
    parser.add_argument('--Savee_original_data_path', default=None)
    parser.add_argument('--crema_original_data_path', default=None)
    parser.add_argument('--rav_path', default=None)
    a = parser.parse_args()
    os.mkdir('data_spectrograms')
    
    
    TESSPipeline.create_tess_folders(a.TESS_ORIGINAL_FOLDER_PATH)
    SaveePipeline.create_savee_folder(a.Savee_original_data_path)
    CremaPipeline.create_crema_folder(a.crema_original_data_path)
    ravPipeline.create_rav_folder(a.rav_path)
    
    
    
    
    
if __name__ == '__main__':
    
    main()
    
    