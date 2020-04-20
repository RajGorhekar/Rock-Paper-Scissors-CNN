import cv2
import os
import sys
import time
basePath = 'RawImages'

def getImagesOf(move):
    classPath = os.path.join(basePath,move)
    try:
        os.mkdir(basePath)
    except:
        pass
    try:
        os.mkdir(classPath)
    except:
        pass
    
    window = cv2.VideoCapture(0)
    count = 0
    while True:
        val,frame = window.read()
        frame = cv2.flip(frame,1)
        cv2.rectangle(frame, (200, 80), (600, 400), (153, 110, 0), 2)
        mainFrame = frame[80:400, 200:600]
        if not val:
            continue
        if count == 150:
            break
        
        if count == 75:
            time.sleep(15)
        imagePath = os.path.join(classPath, '{}{}.jpg'.format(move,count + 1))
        cv2.imwrite(imagePath, mainFrame)
        count += 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "{} Images Collected ".format(count),(10, 460), font, 0.7,(225, 203, 71), 2, cv2.LINE_AA)
        s = "Collecting images for "+ move
        cv2.imshow(s, frame)
        cv2.waitKey(10)
        time.sleep(0.25)
    window.release()
    cv2.destroyAllWindows()


def getAllImages():
    getImagesOf('rock')
    time.sleep(10)
    getImagesOf('paper')
    time.sleep(10)
    getImagesOf('scissors')
    time.sleep(10)
    getImagesOf('none')


# getAllImages()


dict = {"rock": 0,"paper": 1,"scissors": 2,"none": 3}


# import keras_preprocessing

import numpy as np
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
import tensorflow as tf


x=[]
y=[] # labels
for folder in os.listdir(basePath):
    classPath = os.path.join(basePath,folder)
    for image in os.listdir(classPath):
        imgPath = os.path.join(classPath, image)
        img = cv2.imread(imgPath,1)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(227,227))
        y.append(dict[str(folder)])
        x.append(img)
y = np_utils.to_categorical(y)


model = Sequential([
    SqueezeNet(input_shape = (227,227,3),include_top = False),
    Dropout(0.5),
    Convolution2D(4,(1,1),padding = 'valid'),
    Activation('relu'),
    GlobalAveragePooling2D(),
    Activation('softmax'),
])


# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(227, 227, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(4, activation='softmax')
# ])
model.compile(loss="categorical_crossentropy",optimizer = Adam(lr =0.0001), metrics =['accuracy'])
history = model.fit(np.array(x),np.array(y),epochs = 12 , verbose = 1)
model.save("rps.h5")