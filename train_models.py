## train_models.py -- train the neural network models for attacking
##
## Copyright (C) 2017, Yash Sharma <ysharma1126@gmail.com>.
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
import os

def train(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1, init=None, adversarial=False, examples=None, labels=None):
    """
    Standard neural network training procedure.
    """

    if adversarial:
        data.train_data = np.concatenate((data.train_data, examples), axis=0)
        data.train_labels = np.concatenate((data.train_labels, labels), axis=0)
    model = Sequential()
    
    model.add(Conv2D(params[0], (3, 3),
                            input_shape=data.train_data.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(params[1], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(params[2], (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(params[3], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[4]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params[5]))
    model.add(Activation('relu'))
    model.add(Dense(10))
    
    if init != None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              nb_epoch=num_epochs,
              shuffle=True, verbose=1)
    

    if file_name != None:
        model.save(file_name)

    return model

def train_distillation(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1):
    """
    Train a network using defensive distillation.

    Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks
    Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami
    IEEE S&P, 2016.
    """
    if not os.path.exists(file_name+"_init"):
        # Train for one epoch to get a good starting point.
        train(data, file_name+"_init", params, 1, batch_size)
    
    # now train the teacher at the given temperature
    teacher = train(data, file_name+"_teacher", params, num_epochs, batch_size, train_temp,
                    init=file_name+"_init")

    # evaluate the labels at temperature t
    predicted = teacher.predict(data.train_data)
    with tf.Session() as sess:
        y = sess.run(tf.nn.softmax(predicted/train_temp))
        #print(y)
        data.train_labels = y

    # train the student model at temperature t
    student = train(data, file_name, params, num_epochs, batch_size, train_temp,
                    init=file_name+"_init")

    # and finally we predict at temperature 1
    predicted = teacher.predict(data.train_data)

    #print(predicted)

def main(args):
    if not os.path.isdir('models'):
        os.makedirs('models')

    if(not (args['adversarial'] or args['defensive'])):
        if(args['dataset'] == "mnist" or args['dataset'] == "all"):
            train(MNIST(), "models/mnist", [32, 32, 64, 64, 200, 200], num_epochs=50)
        if(args['dataset'] == 'cifar' or args['dataset'] == 'all'):
            train(CIFAR(), "models/cifar", [64, 64, 128, 128, 256, 256], num_epochs=50)

    if (args['adversarial']):
        cwl2 = np.load('train/L2_train.npy')
        cwl1 = np.load('train/L1_train.npy')
        cwen = np.load('train/EN_train.npy')

        labels = np.load('train/labels_train.npy')
        cwl2l1 = np.concatenate((cwl2,cwl1), axis=0)
        cwl2en = np.concatenate((cwl2,cwen), axis=0)
        labels_2 = np.concatenate((labels,labels), axis=0)

        train(MNIST(), "models/mnist_cwl2", [32, 32, 64, 64, 200, 200], num_epochs=50, adversarial=True, examples=cwl2, labels=labels)
        train(MNIST(), "models/mnist_cwl1", [32, 32, 64, 64, 200, 200], num_epochs=50, adversarial=True, examples=cwl1, labels=labels)
        train(MNIST(), "models/mnist_cwe", [32, 32, 64, 64, 200, 200], num_epochs=50, adversarial=True, examples=cwen, labels=labels)
        train(MNIST(), "models/mnist_cwl2l1", [32, 32, 64, 64, 200, 200], num_epochs=50, adversarial=True, examples=cwl2l1, labels=labels_2)
        train(MNIST(), "models/mnist_cwl2e", [32, 32, 64, 64, 200, 200], num_epochs=50, adversarial=True, examples=cwl2en, labels=labels_2)

    if (args['defensive']):
        if(args['temp'] == 0):
            temp = [1,10,20,30,40,50,60,70,80,90,100]
        else:
            temp = args['temp']
    if (args['defensive'] and (args['dataset'] == "mnist" or args['dataset'] == "all")):
        for t in temp:
            print('Mnist_'+str(t))
            train_distillation(MNIST(), "models/mnist-distilled-"+str(t), [32, 32, 64, 64, 200, 200],
                               num_epochs=50, train_temp=t)
    if (args['defensive'] and (args['dataset'] == "cifar" or args['dataset'] == "all")):
        for t in temp:
            print('Cifar_'+str(t))
            train_distillation(CIFAR(), "models/cifar-distilled-"+str(t), [64, 64, 128, 128, 256, 256],
                               num_epochs=50, train_temp=t)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["mnist", "cifar", "all"], default="all")
    parser.add_argument("-a", "--adversarial", action='store_true')
    parser.add_argument("-dd", "--defensive", action='store_true')
    parser.add_argument("-t", "--temp", nargs='+', type=int, default=0)
    args = vars(parser.parse_args())
    print(args)
    main(args)