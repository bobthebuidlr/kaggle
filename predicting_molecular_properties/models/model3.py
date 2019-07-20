import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model, models
from tensorflow.keras.layers import (Activation, Average, BatchNormalization,
                                     Dense, Dropout, Input, concatenate)

try:
	os.chdir('/Users/bobvanderhelm/datasets/champs-scalar-coupling/')
except:
	pass

print('Loading in processed data...')
_train = pd.read_csv('processed/train.csv')
_test = pd.read_csv('processed/test.csv')

Ytrain = _train.pop('scalar_coupling_constant')

print('Scaling the data...')
Xtrain = StandardScaler().fit_transform(_train)
test = StandardScaler().fit_transform(_test)

LOAD_EPOCHS = 20
ADD_EPOCHS = 10
MODEL = 5

def nn_model():
    i = Input(shape=(15,))
     
    # Branch 1
    x1  = Dense(128,activation = 'relu')(i)
    x1  = BatchNormalization()(x1)
    x1  = Dense(64,activation = 'relu')(x1)
    x1  = BatchNormalization()(x1)
    x1  = Dense(32,activation = 'relu')(x1)
    x1  = BatchNormalization()(x1)
    x1  = Dense(16,activation = 'relu')(x1)
    x1  = BatchNormalization()(x1)

    x1_output = Dense(1,activation = 'linear')(x1)
    
    # Branch 2
    x2  = Dense(128,activation = 'relu')(i)
    x2  = BatchNormalization()(x2)
    x2  = Dense(64,activation = 'relu')(x2)
    x2  = BatchNormalization()(x2)
    x2  = Dense(32,activation = 'relu')(x2)
    x2  = BatchNormalization()(x2)
    x2  = Dense(16,activation = 'relu')(x2)
    x2  = BatchNormalization()(x2)

    x2_output = Dense(1,activation = 'linear')(x2)

    # Branch 3
    x3  = Dense(128,activation = 'relu')(i)
    x3  = BatchNormalization()(x3)
    x3  = Dense(64,activation = 'relu')(x3)
    x3  = BatchNormalization()(x3)
    x3  = Dense(32,activation = 'relu')(x3)
    x3  = BatchNormalization()(x3)
    x3  = Dense(16,activation = 'relu')(x3)
    x3  = BatchNormalization()(x3)

    x3_output = Dense(1,activation = 'linear')(x3)

    scalars = Average()([x1_output, x2_output, x3_output])

    return Model(inputs = [i] , outputs = [scalars])

model = nn_model()
print('Compiling the model...') 
model.compile(loss='mean_absolute_error', optimizer='adam')

def train_model(load=False):
    print('Start training the model')
    if load:
        model.load_weights('models/model%s_%s-epochs.h5' % (MODEL, LOAD_EPOCHS))
    model.fit(Xtrain, Ytrain, validation_split=0.1, epochs=ADD_EPOCHS, batch_size=1024)
    model.save_weights('models/model%s_%s-epochs.h5' % (MODEL, LOAD_EPOCHS+ADD_EPOCHS))

def create_submission(load=True):
    print('Creating submission')
    if load:
        model.load_weights('models/model%s_%s-epochs.h5' % (MODEL, LOAD_EPOCHS))
    predictions = model.predict(test)
    submission = pd.read_csv('sample_submission.csv')
    submission['scalar_coupling_constant'] = predictions
    submission.to_csv('submissions/model%s_%s-epochs.csv' % (MODEL, LOAD_EPOCHS), index=False)

create_submission(True)
