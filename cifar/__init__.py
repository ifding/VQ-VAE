import numpy as np
import tensorflow as tf
import os


class DataSampler(object):
    def __init__(self):
        self.shape = [32, 32, 3]
        train_data, test_data = tf.keras.datasets.cifar10.load_data()
        self.X_train, self.y_train = train_data[0], train_data[1].flatten()
        self.X_test, self.y_test = test_data[0], test_data[1].flatten()
        self.X_train = self.X_train/255.0
        self.X_test = self.X_test/255.0
        self.data_variance = np.var(self.X_train)
        self.train_size = self.X_train.shape[0]       
        print(self.X_train.shape, self.y_train.shape)
        print(self.X_test.shape, self.y_test.shape)

        self.X_total = np.concatenate((self.X_train, self.X_test), axis=0)
        self.y_total = np.concatenate((self.y_train, self.y_test), axis=0)
        
        print(np.unique(self.y_total))


    def train(self, batch_size, label=False):
        indx = np.random.randint(low = 0, high = self.train_size, size = batch_size)
        if label:
           return self.X_train[indx, :], self.y_train[indx].flatten()
        else:
           return self.X_train[indx, :]

    def test(self):
        return self.X_test, self.y_test

    def validation(self):
        return self.X_train[-1000:,:], self.y_train[-1000:].flatten()


    def data2img(self, data):
        return np.reshape(data, [data.shape[0]] + self.shape)

    def load_all(self):
        return self.X_total, self.y_total