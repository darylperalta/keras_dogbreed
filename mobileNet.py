import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import Callback, ModelCheckpoint

import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2


from random import randint
im_size = 90

num_samples = 10222

num_class = 120
#steps_per_epoch = num_samples//batch_size

#print(steps_per_epoch)
#epochs = 2

def data_gen(batch_size, image_size):

    df_train = pd.read_csv('../input/labels.csv')

    targets_series = pd.Series(df_train['breed'])
    one_hot = pd.get_dummies(targets_series, sparse = True)
    one_hot_labels = np.asarray(one_hot)



    fn = pd.Series(df_train['id'])


    x_train = []
    y_train = []


    while True:
        for i in range(batch_size):
            #index = np.random.choice(fn.shape[0],1)
            index = randint(0,fn.shape[0]-1)
            img = cv2.imread('../input/train/{}.jpg'.format(fn[index]))
            x_train.append(cv2.resize(img,(image_size,image_size)))
            label = one_hot_labels[index]
            y_train.append(label)
        y_train_raw = np.array(y_train, np.uint8)
        x_train_raw = np.array(x_train, np.float32) / 255.
#        print(i)
        yield x_train_raw, y_train_raw

base_model = MobileNet(#weights='imagenet',
    weights = 'imagenet', include_top=False, input_shape=(im_size, im_size, 3))

# Add a new top layer
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_class, activation='softmax')(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
model.summary()


checkpointpath="/media/airscan/Data/AIRSCAN/EE298F/dogbreed/mbnet-weights-improvement-{:02d}.hdf5"
checkpoint = ModelCheckpoint(checkpointpath, verbose=1)

batch_size = 16
model.fit_generator(data_gen( batch_size=batch_size, image_size = im_size),
                    steps_per_epoch=32,
                    epochs=10, verbose =1)


x_test = []


df_test = pd.read_csv('../input/sample_submission.csv')

for f in tqdm(df_test['id'].values):
    img = cv2.imread('../input/test/{}.jpg'.format(f))
    x_test.append(cv2.resize(img, (im_size, im_size)))

x_test  = np.array(x_test, np.float32) / 255.

preds = model.predict(x_test, verbose=1)

sub = pd.DataFrame(preds)
# Set column names to those generated by the one-hot encoding earlier
col_names = one_hot.columns.values
sub.columns = col_names
# Insert the column id from the sample_submission at the start of the data frame
sub.insert(0, 'id', df_test['id'])
sub.head(10358)
df_test.to_csv('pred_mnet.csv', index=None)
