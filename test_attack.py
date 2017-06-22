## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import time
import random

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi
from l1_attack import CarliniL1


def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    sample_set = random.sample(range(0,10000),samples)
    for i in sample_set:
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


if __name__ == "__main__":
    with tf.Session() as sess:
        data, model =  MNIST(), MNISTModel("models/mnist", sess)
        attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0)

        inputs, targets = generate_data(data, samples=1000, targeted=True, start=0, inception=False)
        #print(len(inputs))
        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()
        
        print("Took",timeend-timestart,"seconds to run",len(inputs)/9,"random instances.")

        r_best = []
        d_best = []
        r_average = []
        d_average = []
        r_worst = []
        d_worst = []
        for i in range(0,len(inputs),9):
            if((i % 900) == 0):
                for j in range(i,i+9):
                    print("Valid:")
                    show(inputs[i])
                    print("Adversarial:")
                    show(adv[i])
                    if(np.argmax(model.model.predict(adv[j:j+1]),1) == np.argmax(targets[j:j+1],1)):
                        print("Successful", model.model.predict(adv[j:j+1]))
                        print("Distortion", (np.sum((adv[j]-inputs[j])**2)**.5))
                    else:
                        print("Unsuccessful", model.model.predict(adv[j:j+1]))
            dist = 1e10
            dist_index = 1e10
            for j in range(i,i+9):
                if(np.argmax(model.model.predict(adv[j:j+1]),1) == np.argmax(targets[j:j+1],1)):
                    if((np.sum((adv[j]-inputs[j])**2)**.5) < dist):
                        dist = (np.sum((adv[j]-inputs[j])**2)**.5)
                        dist_index = j
            if(dist_index != 1e10):
                d_best.append((np.sum((adv[dist_index]-inputs[dist_index])**2)**.5))
                r_best.append(1)
            else:
                r_best.append(0)

            rand_int = np.random.randint(i,i+9)
            if(np.argmax(model.model.predict(adv[rand_int:rand_int+1]),1) == np.argmax(targets[rand_int:rand_int+1],1)):
                r_average.append(1)
                d_average.append(np.sum((adv[rand_int]-inputs[rand_int])**2)**.5)
            else:
                r_average.append(0)

            dist = 0
            dist_index = 1e10
            for j in range(i,i+9):
                if(np.argmax(model.model.predict(adv[j:j+1]),1) != np.argmax(targets[j:j+1],1)):
                    r_worst.append(0)
                    dist_index = 1e10
                    break
                else:
                    if((np.sum((adv[j]-inputs[j])**2)**.5) > dist):
                        dist = (np.sum((adv[j]-inputs[j])**2)**.5)
                        dist_index = j
            if(dist_index != 1e10):
                d_worst.append((np.sum((adv[dist_index]-inputs[dist_index])**2)**.5))
                r_worst.append(1)
            #print(targets[i:i+1])
            #print("Accuracy:", np.mean(r))

            #print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)
        print('best_case_mean', np.mean(d_best))
        print('best_case_prob', np.mean(r_best))
        print('average_case_mean', np.mean(d_average))
        print('average_case_prob', np.mean(r_average))
        print('worst_case_mean', np.mean(d_worst))
        print('worst_case_prob', np.mean(r_worst))