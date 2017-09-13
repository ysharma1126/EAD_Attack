## en_attack.py -- attack a network optimizing elastic-net distance with an en decision rule
##
## Copyright (C) 2017, Yash Sharma <ysharma1126@gmail.com>.
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np

BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e-3     # the initial constant c to pick as a first guess
BETA = 1e-3              # Hyperparameter trading off L2 minimization for L1 minimization

class EADEN:
    def __init__(self, sess, model, batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS,
                 abort_early = ABORT_EARLY, 
                 initial_const = INITIAL_CONST, beta = BETA):
        """
        EAD with EN Decision Rule 

        Returns adversarial examples for the supplied model.
        """

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.beta = beta
        self.beta_t = tf.cast(self.beta, tf.float32)

        self.repeat = binary_search_steps >= 10

        shape = (batch_size,image_size,image_size,num_channels)

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.newimg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.slack = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_newimg = tf.placeholder(tf.float32, shape)
        self.assign_slack = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size,num_labels))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        
        self.global_step = tf.Variable(0, trainable=False)
        self.global_step_t = tf.cast(self.global_step, tf.float32)

        """Fast Iterative Soft Thresholding"""
        """--------------------------------"""
        self.zt = tf.divide(self.global_step_t, self.global_step_t+tf.cast(3, tf.float32))

        cond1 = tf.cast(tf.greater(tf.subtract(self.slack, self.timg),self.beta_t), tf.float32)
        cond2 = tf.cast(tf.less_equal(tf.abs(tf.subtract(self.slack,self.timg)),self.beta_t), tf.float32)
        cond3 = tf.cast(tf.less(tf.subtract(self.slack, self.timg),tf.negative(self.beta_t)), tf.float32)

        upper = tf.minimum(tf.subtract(self.slack,self.beta_t), tf.cast(0.5, tf.float32))
        lower = tf.maximum(tf.add(self.slack,self.beta_t), tf.cast(-0.5, tf.float32))

        self.assign_newimg = tf.multiply(cond1,upper)+tf.multiply(cond2,self.timg)+tf.multiply(cond3,lower)
        self.assign_slack = self.assign_newimg+tf.multiply(self.zt, self.assign_newimg-self.newimg)
        self.setter = tf.assign(self.newimg, self.assign_newimg)
        self.setter_y = tf.assign(self.slack, self.assign_slack)
        """--------------------------------"""
        # prediction BEFORE-SOFTMAX of the model
        self.output = model.predict(self.newimg)
        self.output_y = model.predict(self.slack)
        
        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg-self.timg),[1,2,3])
        self.l2dist_y = tf.reduce_sum(tf.square(self.slack-self.timg),[1,2,3])
        self.l1dist = tf.reduce_sum(tf.abs(self.newimg-self.timg),[1,2,3])
        self.l1dist_y = tf.reduce_sum(tf.abs(self.slack-self.timg),[1,2,3])
        self.elasticdist = self.l2dist + tf.multiply(self.l1dist, self.beta_t)
        self.elasticdist_y = self.l2dist_y + tf.multiply(self.l1dist_y, self.beta_t)
        
        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum((self.tlab)*self.output,1)
        real_y = tf.reduce_sum((self.tlab)*self.output_y,1)
        other = tf.reduce_max((1-self.tlab)*self.output - (self.tlab*10000),1)
        other_y = tf.reduce_max((1-self.tlab)*self.output_y - (self.tlab*10000),1)
        if self.TARGETED:
            # if targeted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other-real+self.CONFIDENCE)
            loss1_y = tf.maximum(0.0, other_y-real_y+self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real-other+self.CONFIDENCE)
            loss1_y = tf.maximum(0.0, real_y-other_y+self.CONFIDENCE)

        # sum up the losses
        self.loss21 = tf.reduce_sum(self.l1dist)
        self.loss21_y = tf.reduce_sum(self.l1dist_y)
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss2_y = tf.reduce_sum(self.l2dist_y)
        self.loss1 = tf.reduce_sum(self.const*loss1)
        self.loss1_y = tf.reduce_sum(self.const*loss1_y)

        self.loss_opt = self.loss1_y+self.loss2_y
        self.loss = self.loss1+self.loss2+tf.multiply(self.beta_t,self.loss21)
        
        self.learning_rate = tf.train.polynomial_decay(self.LEARNING_RATE, self.global_step, self.MAX_ITERATIONS, 0, power=0.5) 
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train = optimizer.minimize(self.loss_opt, var_list=[self.slack], global_step=self.global_step)
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        
        self.init = tf.variables_initializer(var_list=[self.global_step]+[self.slack]+[self.newimg]+new_vars)

    def attack(self, imgs, targets):
        """
        Perform the EAD attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to',len(imgs))
        for i in range(0,len(imgs),self.batch_size):
            print('tick',i)
            r.extend(self.attack_batch(imgs[i:i+self.batch_size], targets[i:i+self.batch_size]))
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        def compare(x,y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                x[y] -= self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.initial_const
        upper_bound = np.ones(batch_size)*1e10

        # the best l2, score, and image attack
        o_besten = [1e10]*batch_size
        o_bestscore = [-1]*batch_size
        o_bestattack = [np.zeros(imgs[0].shape)]*batch_size
        
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]
    
            besten = [1e10]*batch_size
            bestscore = [-1]*batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS-1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST})
            self.sess.run(self.setter, feed_dict={self.assign_newimg: batch})
            self.sess.run(self.setter_y, feed_dict={self.assign_slack: batch})
            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack 
                self.sess.run([self.train])
                self.sess.run([self.setter, self.setter_y])
                l, l2s, l1s, elastic, scores, nimg = self.sess.run([self.loss, self.l2dist, self.l1dist, self.elasticdist, self.output, self.newimg])



                # print out the losses every 10%
                """
                if iteration%(self.MAX_ITERATIONS//10) == 0:
                    print(iteration,self.sess.run((self.loss,self.loss1,self.loss2,self.loss21)))
                """
                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration%(self.MAX_ITERATIONS//10) == 0:
                    if l > prev*.9999:
                        break
                    prev = l

                # adjust the best result found so far
                for e,(en,sc,ii) in enumerate(zip(elastic,scores,nimg)):
                    if en < besten[e] and compare(sc, np.argmax(batchlab[e])):
                        bestl1[e] = l1
                        bestscore[e] = np.argmax(sc)
                    if en < o_besten[e] and compare(sc, np.argmax(batchlab[e])):
                        o_besten[e] = en
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    else:
                        CONST[e] *= 10

        # return the best solution found
        o_besten = np.array(o_besten)
        return o_bestattack
