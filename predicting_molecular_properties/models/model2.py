import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input

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

LOAD_EPOCHS = 30
ADD_EPOCHS = 20
MODEL = 2

model = models.Sequential([
    Input(shape=(15,)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='linear')
])

print('Compiling the model...') 
model.compile(loss='mean_absolute_error', optimizer='adam')

def train_model(load=False):
    print('Start training the model')
    if load:
        model.load_weights('models/model%s_%s-epochs.h5' % (MODEL, LOAD_EPOCHS))
    model.fit(Xtrain, Ytrain, validation_split=0.1, epochs=LOAD_EPOCHS+ADD_EPOCHS, batch_size=1024)
    model.save_weights('models/model%s_%s-epochs.h5' % (MODEL, LOAD_EPOCHS+ADD_EPOCHS))

def create_submission(load=True):
    print('Creating submission')
    if load:
        model.load_weights('models/model%s_%s-epochs.h5' % (MODEL, LOAD_EPOCHS))
    predictions = model.predict(test)
    submission = pd.read_csv('sample_submission.csv')
    submission['scalar_coupling_constant'] = predictions
    submission.to_csv('submissions/model%s_%s-epochs.csv' % (MODEL, LOAD_EPOCHS), index=False)

create_submission()
