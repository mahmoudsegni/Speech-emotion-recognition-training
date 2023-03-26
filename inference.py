from data_preparation import *
from train import *
from keras.models import load_model

sampling_rate=44100
audio_duration=2.5
n_mfcc = 30















def main():
    print('Initializing inference process')
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_of_the_wav_file', default="/mnt/corpus/data/emotion_data/all_data/savee/43-97-02-83-77-54-31.wav")
    parser.add_argument('--model', default='./models/conv2d_mfcc_aug.h5')
    parser.add_argument('--encoder', default='./models/onehot.joblib')
    
   
    a = parser.parse_args()
    path=a.path_of_the_wav_file
    model_path=a.model
    encoder_path=a.encoder
    encoder=joblib.load(encoder_path)
    model = load_model(model_path)
    lst = [[path]]
    # Calling DataFrame constructor on list  
    ref=pd.DataFrame(lst,columns=['Path'])  
    mfcc = prepare_data(ref, n = n_mfcc, aug = 0, mfcc = 1,transform=None)
    X_train=mfcc
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    preds = model.predict(X_train)
    preds=preds.argmax(axis=1)
    preds = preds.astype(int).flatten()
    preds = (encoder.inverse_transform((preds)))
    print('*****************')
    print('The predicted Emotion is:',preds)
#     print(preds)
    return preds
 
    

    
        

        
        
        
        
        
        
if __name__ == '__main__':
    
    main()