import argparse
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from keras.utils import np_utils

from cnn_models import *

'''
# 4. Create the 2D CNN model 
'''
def get_2d_conv_model_new(n):
    ''' Create a standard deep 2D convolutional neural network'''
    nclass = 8
    inp = Input(shape=(n,216,1))  #2D matrix of 30 MFCC bands by 216 audio length.
    x = Convolution2D(254, (4,10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)
    
    x = Convolution2D(254, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)
    
    x = Convolution2D(128, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)
    
    x = Convolution2D(128, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)
    
    x = Flatten()(x)
    x = Dense(254)(x)
    x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(rate=0.2)(x)
    
    out = Dense(nclass, activation=softmax)(x)
    model = models.Model(inputs=inp, outputs=out)
#     opt = tf.keras.optimizers.Adam(0.001)
#     opt=keras.optimizer_v2.Adam(0.001)
#     opt = tf.keras.optimizers.Adam(learning_rate=0.001)
#     opt = optimizers.SGD(lr = 0.001)
    model.compile(optimizer="adam", loss=losses.categorical_crossentropy, metrics=['acc'])
    return model



def main():
    print('Start Training..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_of_the_data', default='./vectors_data/data.joblib')
    parser.add_argument('--path_of_the_labels', default='./vectors_data/labels.joblib')
    parser.add_argument('--augment_data', default=0)
    parser.add_argument('--data_extraction_type', default=1)
    a = parser.parse_args()
    path_of_the_data=a.path_of_the_data
    path_of_the_labels=a.path_of_the_labels
    data = joblib.load(path_of_the_data)
    labels= joblib.load(path_of_the_labels)

    X_train, X_test, y_train, y_test = train_test_split(data
                                                        , labels
                                                        , test_size=0.03
                                                        , shuffle=True
                                                        , random_state=42
                                                       )
    # one hot encode the target 
    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))
    checkpoint_filepath = 'models/conv2d_mfcc_aug_up.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath= checkpoint_filepath,
        save_weights_only=False,
        monitor='val_acc',
        mode='max',
        save_best_only=True)


    n_mfcc = 30
    model = get_2d_conv_model_new(n=n_mfcc)
    model_history = model.fit(X_train, y_train, validation_split=0.1, 
                        batch_size=15, verbose = 1, epochs=100,callbacks=[model_checkpoint_callback])
    
    print(model_history)
    
    
    
    
    
    
    
    
    
        
if __name__ == '__main__':
    
    main()