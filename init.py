# Things required to unpack the CIFAR-10 library
import os
# import h5py
import six
from six.moves import range, cPickle
import tarfile

# Main Library for Matrices manipulation
import numpy as np

# To draw the images
import matplotlib.pyplot as plt

import pickle

def pydump(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def pyload(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def cifar_10():
    # LOAD TRAINING DATA
    tar_file = tarfile.open("cifar-10-python.tar.gz", 'r:gz')
    train_batches = []
    for batch in range(1, 6):
        file = tar_file.extractfile(
            'cifar-10-batches-py/data_batch_%d' % batch)
        try:
            if six.PY3:
                array = cPickle.load(file, encoding='latin1')
            else:
                array = cPickle.load(file)
            train_batches.append(array)
        finally:
            file.close()

    train_features = np.concatenate(
        [batch['data'].reshape(batch['data'].shape[0], 3, 32, 32)
            for batch in train_batches])
    train_labels = np.concatenate(
        [np.array(batch['labels'], dtype=np.uint8)
            for batch in train_batches])
    train_labels = np.expand_dims(train_labels, 1)


    # LOAD TEST DATA
    file = tar_file.extractfile('cifar-10-batches-py/test_batch')
    try:
        if six.PY3:
            test = cPickle.load(file, encoding='latin1')
        else:
            test = cPickle.load(file)
    finally:
        file.close()

    test_features = test['data'].reshape(test['data'].shape[0],
                                         3, 32, 32)
    test_labels = np.array(test['labels'], dtype=np.uint8)
    test_labels = np.expand_dims(test_labels, 1)

    return train_features, train_labels, test_features, test_labels

train_features, train_labels, test_features, test_labels = cifar_10()
# train100_labels, train100_labels, test100_features, test100_labels = load_cifar(100)

X = train_features.reshape(50000, 3*32*32)
Xt = test_features.reshape(10000, 3*32*32)
y = train_labels.flatten()
yt = test_labels.flatten()

# X100 = train100_features.reshape(50000, 3*32*32)
# Xt100 = test100_features.reshape(10000, 3*32*32)
# y100 = train100_labels.flatten()
# yt100 = test100_labels.flatten()

# Helps with writing functions!
msg = lambda x: print("%s ... " % x, end="")
done = lambda: print("done.")

# Threading facility
import sys
IS_PY2 = sys.version_info < (3, 0)

if IS_PY2:
    from Queue import Queue
else:
    from queue import Queue

from threading import Thread


class Worker(Thread):
    """ Thread executing tasks from a given tasks queue """
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()

    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
            try:
                func(*args, **kargs)
            except Exception as e:
                # An exception happened in this thread
                print(e)
            finally:
                # Mark this task as done, whether an exception happened or not
                self.tasks.task_done()


class ThreadPool:
    """ Pool of threads consuming tasks from a queue """
    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        for _ in range(num_threads):
            Worker(self.tasks)

    def add_task(self, func, *args, **kargs):
        """ Add a task to the queue """
        self.tasks.put((func, args, kargs))

    def map(self, func, args_list):
        """ Add a list of tasks to the queue """
        for args in args_list:
            self.add_task(func, args)

    def wait_completion(self):
        """ Wait for completion of all the tasks in the queue """
        self.tasks.join()

class Schneller():
    def f(self, func, key, results, total, *args, **kwargs):
        results[key] = func(*args, **kwargs)
        print("\rCompleted: {} out of {}.".format(len(results.keys()), total), end="")

    def __init__(self, func, args_list, slice=1, num_threads=4):
        self.slice = slice
        self.func = func
        self.args_list = args_list
        self.results = {}
        self.thread_pool = ThreadPool(num_threads)

        if len(self.args_list) % self.slice is not 0:
            raise "Length of the contents array should be divisible by the provided slice."

        self.total = int(len(self.args_list)/self.slice)

    def compute(self):
        for i in range(self.total):
            self.thread_pool.add_task(self.f, self.func, i, self.results, self.total, self.args_list[i*self.slice:(i+1)*self.slice])
        self.thread_pool.wait_completion()

        return np.vstack(list(map(lambda i: self.results[i], range(self.total))))
