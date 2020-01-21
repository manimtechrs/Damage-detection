# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 12:31:02 2019

@author: user
"""

from tensorflow.keras.applications import densenet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import tensorflow.keras as keras
keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator

traindf=pd.read_csv('D:/My files/courses/108-1/Machine learning & DL/Final project/Task2/train.csv',dtype=str)
testdf=pd.read_csv('D:/My files/courses/108-1/Machine learning & DL/Final project/Task2/test.csv',dtype=str)
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)


train_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="D:/My files/courses/108-1/Machine learning & DL/Final project/Task2/train_img/",
x_col="image",
y_col="label",
subset="training",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(224,224))
valid_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="D:/My files/courses/108-1/Machine learning & DL/Final project/Task2/train_img/",
x_col="image",
y_col="label",
subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(224,224))

test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
dataframe=testdf,
directory="D:/My files/courses/108-1/Machine learning & DL/Final project/Task2/test_img/",
x_col="image",
y_col=None,
batch_size=1,
seed=42,
shuffle=False,
class_mode=None,
target_size=(224,224))




number_of_classes =2
#base_model = densenet.DenseNet121(weights='densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
base_model=densenet.DenseNet121(weights='imagenet',include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
preds = Dense(number_of_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=preds)
# Print the updated layer names.
# for i,layer in enumerate(model.layers): print(i,layer.name)
# Set the first n_freeze layers of the network to be non-trainable.
n_freeze = 300
for layer in model.layers[:n_freeze]:
 layer.trainable=False
for layer in model.layers[n_freeze:]:
 layer.trainable=True
 
 '''
for layer in model.layers: 
layer.trainable = False
'''
from tensorflow.keras import regularizers, optimizers
#adam=keras.optimizers.Adam(lr=0.001)
model.compile (optimizers.RMSprop(lr=0.0001, decay=1e-6), loss="categorical_crossentropy",metrics=["accuracy"]) 
#step_size_train = train_batches.n // train_batches.batch_size
 


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size


'''
model.fit_generator(train_batches,
                        steps_per_epoch = step_size_train,
                        validation_data = valid_batches,
                        validation_steps = 5,
                        epochs = 5,
                        shuffle=True,verbose=1)


'''

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=5,shuffle=True,verbose=1)




model.evaluate_generator(generator=valid_generator,steps=STEP_SIZE_TEST)

test_generator.reset()
pred=model.predict_generator(test_generator,steps=STEP_SIZE_TEST,verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("D:/My files/courses/108-1/Machine learning & DL/Final project/Task2/rsv3.csv",index=False)
