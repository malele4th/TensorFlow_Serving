#!coding=utf-8
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import math

def convert_to_one_hot(label, classes):   # label -->> one-hot
    label = np.array(label)
    label = np.eye(classes)[label.reshape(-1)]
    return label

def generate_data():
    data,label = make_blobs(n_samples=10000, n_features=4, centers=12, random_state=0)
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2)
    file = h5py.File('data/data_set.h5','w')
    file.create_dataset('train_x',data = train_x)
    file.create_dataset('train_y',data = train_y)
    file.create_dataset('test_x', data = test_x)
    file.create_dataset('test_y', data = test_y)
    file.close()

def read_data(path):
    filename = path + 'data_set.h5'
    file = h5py.File(filename,'r')
    train_x = file['train_x'][:]
    train_y = file['train_y'][:]
    test_x  = file['test_x'][:]
    test_y  = file['test_y'][:]
    train_y = convert_to_one_hot(train_y,12) 
    test_y  = convert_to_one_hot(test_y,12)
    print(train_x.shape,train_y.shape)
    print(test_x.shape,test_y.shape)
    return train_x,train_y,test_x,test_y

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]
    #shuffled_Y = Y[permutation,:].reshape((m,Y.shape[1]))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

if __name__=="__main__":
    path='data/'
    generate_data()
    train_x,train_y,test_x,test_y = read_data(path)

    mini_batches = random_mini_batches(train_x,train_y)
    print(len(mini_batches))
    for minibatch in mini_batches:
        minibatch_X, minibatch_Y = minibatch
    print(minibatch_X.shape,minibatch_Y.shape)
