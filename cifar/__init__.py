import numpy as np
import tensorflow as tf
import os
import tarfile
from six.moves import cPickle
from six.moves import urllib

def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    Arguments:
      fpath: path the file to parse.
      label_key: key for label data in the retrieve
          dictionary.
    Returns:
      A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


class DataSampler(object):
    def __init__(self):
        self.shape = [32, 32, 3]
        train_data, test_data = self._load_data('data/cifar-10')
        self.X_train, self.y_train = train_data[0], train_data[1].flatten()
        self.X_test, self.y_test = test_data[0], test_data[1].flatten()
        self.X_train = self.X_train/255.0
        self.X_test = self.X_test/255.0
        self.data_variance = np.var(self.X_train)
        self.train_size = self.X_train.shape[0]

        self.z_size = 8      

        self.X_total = np.concatenate((self.X_train, self.X_test), axis=0)
        self.y_total = np.concatenate((self.y_train, self.y_test), axis=0)
        
        print(np.unique(self.y_total))

    def _load_data(self, data_dir):
        """Loads CIFAR10 dataset.
        Returns:
          Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        """
        if not os.path.exists(data_dir):
            self._download_data(data_dir)

        dir_path = os.path.join(data_dir, 'cifar-10-batches-py')
        num_train_samples = 50000

        x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
        y_train = np.empty((num_train_samples,), dtype='uint8')

        for i in range(1, 6):
            fpath = os.path.join(dir_path, 'data_batch_' + str(i))
            (x_train[(i - 1) * 10000:i * 10000, :, :, :],
             y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

        fpath = os.path.join(dir_path, 'test_batch')
        x_test, y_test = load_batch(fpath)

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))

        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

        x_test = x_test.astype(x_train.dtype)
        y_test = y_test.astype(y_train.dtype)

        return (x_train, y_train), (x_test, y_test)


    def _download_data(self, local_data_dir):
        data_path = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

        tf.gfile.MakeDirs(local_data_dir)

        url = urllib.request.urlopen(data_path)
        archive = tarfile.open(fileobj=url, mode='r|gz') # read a .tar.gz stream
        archive.extractall(local_data_dir)
        url.close()
        archive.close()
        print('extracted data files to %s' % local_data_dir)

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