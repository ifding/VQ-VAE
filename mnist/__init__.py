import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./data/mnist')


class DataSampler(object):
    def __init__(self):
        self.shape = [28, 28, 1]
        self.data_variance = 1
        self.z_size = 7

    def train(self, batch_size, label=False):
        train_images = mnist.train.next_batch(batch_size)[0]
        if label:
           return train_images, mnist.train.next_batch(batch_size)[1]
        else:
           return np.reshape(train_images, [train_images.shape[0]] + self.shape)

    def test(self):
        data_dir = './data/adv'
        if os.path.exists(data_dir):
            normal_images = np.load(os.path.join(data_dir, 'imagesfile.npy'))
            adv_images = np.load(os.path.join(data_dir, 'adversarialfile.npy'))
            test_images = np.concatenate((normal_images, adv_images))
            test_labels = np.concatenate((np.ones(normal_images.shape[0]), np.zeros(adv_images.shape[0])))
        else:                  
            test_images = mnist.test.images
            test_images = np.reshape(test_images, [test_images.shape[0]] + self.shape)
            test_labels = mnist.test.labels
        return test_images, test_labels

    def validation(self):
        validation_images = mnist.validation.images
        return np.reshape(validation_images, [validation_images.shape[0]] + self.shape), mnist.validation.labels


    def data2img(self, data):
        return np.reshape(data, [data.shape[0]] + self.shape)

    def load_all(self):
        X_train = mnist.train.images
        X_val = mnist.validation.images
        X_test = mnist.test.images

        Y_train = mnist.train.labels
        Y_val = mnist.validation.labels
        Y_test = mnist.test.labels

        X = np.concatenate((X_train, X_val, X_test))
        Y = np.concatenate((Y_train, Y_val, Y_test))

        return np.reshape(X, [X.shape[0]] + self.shape), Y.flatten()
