from itertools import chain

import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg') # -----(1)
import matplotlib.pyplot as plt

import keras.backend as K
from keras.optimizers import Adam, SGD, Adagrad
from keras.datasets import cifar100
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model
from keras.utils import np_utils

import DenseNet_BC



def plot_history(history, path_save="", num=0):
    for tag in ['acc', 'loss']:
        list_key = []
    
        for key, graph in history.history.items():
            if tag in key:
                list_key.append(key)
                plt.plot(graph)
            
        plt.title(f'model {tag}')
        plt.xlabel('epoch')
        plt.ylabel(tag)
        plt.legend(list_key, loc='lower right')
        plt.savefig(path.join(path_save, f'{tag}_{num}.png'))
        plt.clf()


def load_dataset(resize=False, img_size=(224, 224)):
    (X_train_tmp, y_train), (X_test_tmp, y_test) = cifar100.load_data()
    enc = OneHotEncoder()
    y_train = enc.fit_transform(y_train).toarray()
    y_test = enc.fit_transform(y_test).toarray()

    if resize:
        X_train = np.empty((len(X_train_tmp), img_size[0], img_size[1], 3), dtype=np.float32)
        X_test = np.empty((len(X_test_tmp), img_size[0], img_size[1], 3), dtype=np.float32)
    
        for idx, img in enumerate(X_train_tmp):
            X_train[idx] = cv2.resize(img, img_size).astype(np.float32)
    
        for idx, img in enumerate(X_test_tmp):
            X_test[idx] = cv2.resize(img, img_size).astype(np.float32)
    else:
        X_train = X_train_tmp
        X_test = X_test_tmp

    img_dim = X_train.shape[1:]
    
    if K.image_data_format() == "channels_first":
        n_channels = X_train.shape[1]
    else:
        n_channels = X_train.shape[-1]
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    # Normalisation
    X = np.vstack((X_train, X_test))
    # 2 cases depending on the image ordering
    if K.image_data_format() == "channels_first":
        for i in range(n_channels):
            mean = np.mean(X[:, i, :, :])
            std = np.std(X[:, i, :, :])
            X_train[:, i, :, :] = (X_train[:, i, :, :] - mean) / std
            X_test[:, i, :, :] = (X_test[:, i, :, :] - mean) / std
    
    elif K.image_data_format() == "channels_last":
        for i in range(n_channels):
            mean = np.mean(X[:, :, :, i])
            std = np.std(X[:, :, :, i])
            X_train[:, :, :, i] = (X_train[:, :, :, i] - mean) / std
            X_test[:, :, :, i] = (X_test[:, :, :, i] - mean) / std

    return (X_train, y_train), (X_test, y_test)





(X_train, y_train), (X_test, y_test) = load_dataset()

batch_size = 64
nb_epoch = 300
optimizer = Adam(lr=1e-3)
# SGD(lr=1e-2, momentum=0.9, decay=1e-4, nesterov=True)

model = DenseNet_BC.DenseNet().build()
plot_model(model, to_file='model.png')
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
history = model.fit(
    X_train, y_train, 
    batch_size=batch_size, 
    epochs=nb_epoch, 
    verbose=1, 
    validation_data=(X_test, y_test)
)

plot_history(history)