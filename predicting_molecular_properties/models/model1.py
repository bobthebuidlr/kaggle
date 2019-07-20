import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models
from tensorflow.keras.layers import BatchNormalization, Dense, Input

try:
	os.chdir('/Users/bobvanderhelm/datasets/champs-scalar-coupling/')
except:
	pass

_train = pd.read_csv('processed/train.csv')
_test = pd.read_csv('processed/test.csv')

Ytrain = _train.pop('scalar_coupling_constant')

Xtrain = StandardScaler().fit_transform(_train)
test = StandardScaler().fit_transform(_test)


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
    Dense(1, activation='linear')
])
 
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

def train_model(load=False):
    if load:
        model.load_weights('models/model1_30_epochs.h5')
    model.fit(Xtrain, Ytrain, validation_split=0.1, epochs=1, batch_size=1024)
    model.save_weights('models/model1_30_epochs.h5')

def create_submission(load=True):
    if load:
        model.load_weights('models/model1_30_epochs.h5')
    predictions = model.predict(test)
    submission = pd.read_csv('sample_submission.csv')
    submission['scalar_coupling_constant'] = predictions
    submission.to_csv('submissions/model1.csv', index=False)

create_submission()
