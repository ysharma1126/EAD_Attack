## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2017, Yash Sharma <ysharma1126@gmail.com>.
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import time
import random
import _pickle as pickle
import os
import scipy

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel

from l2_attack import CarliniL2
from l1_attack import EADL1
from en_attack import EADEN
from fgm import FGM
from ifgm import IFGM

from PIL import Image


def show(img, name = "output.png"):

	fig = (img + 0.5)*255
	fig = fig.astype(np.uint8).squeeze()
	pic = Image.fromarray(fig)
	# pic.resize((512,512), resample=PIL.Image.BICUBIC)
	pic.save(name)
	

def generate_data(data, model, samples, targeted=True, target_num=9, start=0, inception=False, handpick=True, train=False, leastlikely=False,
	sigma=0., seed=3):
	random.seed(seed)
	inputs = []
	targets = []
	labels = []
	true_ids = []
	sample_set = []

	"""
	Generate the input data to the attack algorithm.
	"""
	
	if train:
		data_d = data.train_data
		labels_d = data.train_labels
	else:
		data_d = data.test_data
		labels_d = data.test_labels
	
	if handpick:
		if inception:
			deck = list(range(0,int(1.5 * samples)))
		else:
			deck = list(range(0,10000))
		random.shuffle(deck)
		print('Handpicking')

		while(len(sample_set) < samples):
			rand_int = deck.pop()
			pred = model.model.predict(data_d[rand_int:rand_int+1])
			
			if inception:
				pred = np.reshape(pred, (labels_d[0:1].shape))
			
			if(np.argmax(pred,1) == np.argmax(labels_d[rand_int:rand_int+1],1)):
				sample_set.append(rand_int)
		print('Handpicked')
	else:
		if inception:
			sample_set = random.sample(range(0,int(1.5 * samples)),samples)
		else:
			sample_set = random.sample(range(0,10000),samples)
	
	for i in sample_set:
		if targeted:
			if inception:
				r = list(range(1,1001))
			else:
				r = list(range(labels_d.shape[1]))
			r.remove(np.argmax(labels_d[start+i]))
			seq = random.sample(r, target_num)
			for j in seq:
				inputs.append(data_d[start+i])
				targets.append(np.eye(labels_d.shape[1])[j])
				labels.append(labels_d[start+i])
				true_ids.append(start+i)
		else:
			inputs.append(data_d[start+i])
			targets.append(labels_d[start+i])
			labels.append(labels_d[start+i])
			true_ids.append(start+i)
	inputs = np.array(inputs)
	targets = np.array(targets)
	labels = np.array(labels)
	true_ids = np.array(true_ids)
	return inputs, targets, labels, true_ids

def main(args):
	with tf.Session() as sess:
		if (args['dataset'] == 'mnist'):
			data =  MNIST()
			inception=False
			if (args['adversarial'] != "none"):
				model = MNISTModel("models/mnist_cw"+str(args['adversarial']), sess)
			elif (args['temp']):
				model = MNISTModel("models/mnist-distilled-"+str(args['temp']), sess)
			else:		
				model = MNISTModel("models/mnist", sess)
		if (args['dataset'] == "cifar"):
			data = CIFAR()
			inception=False
			if (args['adversarial'] != "none"):
				model = CIFARModel("models/cifar_cw"+str(args['adversarial']), sess)
			elif (args['temp']):
				model = CIFARModel("models/cifar-distilled-"+str(args['temp']), sess)
			else:
				model = CIFARModel("models/cifar", sess)
		if (args['dataset'] == "imagenet"):
			data, model = ImageNet(args['seed_imagenet'], 2*args['numimg']), InceptionModel(sess)
			inception=True

		inputs, targets, labels, true_ids = generate_data(data, model, samples=args['numimg'], targeted = not args['untargeted'], target_num = args['targetnum'],
			inception=inception, train=args['train'], 
			seed=args['seed'])
		timestart = time.time()
		if(args['restore_np']):
			if(args['train']):
				adv = np.load(str(args['dataset'])+'_'+str(args['attack'])+'_train.npy')
			else:
				adv = np.load(str(args['dataset'])+'_'+str(args['attack'])+'.npy')
		else:
			if (args['attack'] == 'L2'):
				attack = CarliniL2(sess, model, batch_size=args['batch_size'], max_iterations=args['maxiter'], confidence=args['conf'], initial_const=args['init_const'],
					binary_search_steps=args['binary_steps'], targeted = not args['untargeted'], beta=args['beta'], abort_early=args['abort_early'])
				adv = attack.attack(inputs, targets)
			if (args['attack'] == 'L1'):
				attack = EADL1(sess, model, batch_size=args['batch_size'], max_iterations=args['maxiter'], confidence=args['conf'], initial_const=args['init_const'],
					binary_search_steps=args['binary_steps'], targeted = not args['untargeted'], beta=args['beta'], abort_early=args['abort_early'])
				adv = attack.attack(inputs, targets)
			if (args['attack'] == 'EN'):
				attack = EADEN(sess, model, batch_size=args['batch_size'], max_iterations=args['maxiter'], confidence=args['conf'], initial_const=args['init_const'], 
					binary_search_steps=args['binary_steps'], targeted = not args['untargeted'], beta=args['beta'], abort_early=args['abort_early'])
				adv = attack.attack(inputs, targets)

			"""If untargeted, pass labels instead of targets"""
			if (args['attack'] == 'FGSM'):
				attack = FGM(sess, model, batch_size=args['batch_size'], ord=np.inf, eps=args['eps'], inception=inception)
				adv = attack.attack(inputs, targets)
			if (args['attack'] == 'FGML1'):
				attack = FGM(sess, model, batch_size=args['batch_size'], ord=1, eps=args['eps'], inception=inception)
				adv = attack.attack(inputs, targets)
			if (args['attack'] == 'FGML2'):
				attack = FGM(sess, model, batch_size=args['batch_size'], ord=2, eps=args['eps'], inception=inception)
				adv = attack.attack(inputs, targets)
			
			if (args['attack'] == 'IFGSM'):
				attack = IFGM(sess, model, batch_size=args['batch_size'], ord=np.inf, eps=args['eps'], inception=inception)
				adv = attack.attack(inputs, targets)
			if (args['attack'] == 'IFGML1'):
				attack = IFGM(sess, model, batch_size=args['batch_size'], ord=1, eps=args['eps'], inception=inception)
				adv = attack.attack(inputs, targets)
			if (args['attack'] == 'IFGML2'):
				attack = IFGM(sess, model, batch_size=args['batch_size'], ord=2, eps=args['eps'], inception=inception)
				adv = attack.attack(inputs, targets)
		
		timeend = time.time()

		if args['untargeted']:
			num_targets = 1
		else:
			num_targets = args['targetnum']
		print("Took",timeend-timestart,"seconds to run",len(inputs)/num_targets,"random instances.")

		if(args['save_np']):
			if(args['train']):
				np.save(str(args['dataset'])+'_labels_train.npy',labels)
				np.save(str(args['dataset'])+'_'+str(args['attack'])+'_train.npy',adv)
			else:
				np.save(str(args['dataset'])+'_'+str(args['attack']+'.npy'),adv)
		
		r_best_ = []
		d_best_l1_ = []
		d_best_l2_ = []
		d_best_linf_ = []
		r_average_ = []
		d_average_l1_ = []
		d_average_l2_ = []
		d_average_linf_ = []
		r_worst_ = []
		d_worst_l1_ = []
		d_worst_l2_ = []
		d_worst_linf_ = []
		
		#Transferability Tests
		model_ = []
		model_.append(model)
		if (args['targetmodel'] != "same"):
			if(args['targetmodel'] == "dd_100"):
				model_.append(MNISTModel("models/mnist-distilled-100", sess))
		num_models = len(model_)

		if (args['show']):
			if not os.path.exists(str(args['save'])+"/"+str(args['dataset'])+"/"+str(args['attack'])):
				os.makedirs(str(args['save'])+"/"+str(args['dataset'])+"/"+str(args['attack']))
		for m,model in enumerate(model_):
			r_best = []
			d_best_l1 = []
			d_best_l2 = []
			d_best_linf = []
			r_average = []
			d_average_l1 = []
			d_average_l2 = []
			d_average_linf = []
			r_worst = []
			d_worst_l1 = []
			d_worst_l2 = []
			d_worst_linf = []
			for i in range(0,len(inputs),num_targets):
				pred = []
				for j in range(i,i+num_targets):
					if inception:
						pred.append(np.reshape(model.model.predict(adv[j:j+1]), (data.test_labels[0:1].shape)))
					else:
						pred.append(model.model.predict(adv[j:j+1]))

				dist_l1 = 1e10
				dist_l1_index = 1e10
				dist_linf = 1e10
				dist_linf_index = 1e10
				dist_l2 = 1e10
				dist_l2_index = 1e10
				for k,j in enumerate(range(i,i+num_targets)):
					success = False
					if(args['untargeted']):
						if(np.argmax(pred[k],1) != np.argmax(targets[j:j+1],1)):
							success = True
					else:
						if(np.argmax(pred[k],1) == np.argmax(targets[j:j+1],1)):
							success = True
					if(success):
						if(np.sum(np.abs(adv[j]-inputs[j])) < dist_l1):
							dist_l1 = np.sum(np.abs(adv[j]-inputs[j]))
							dist_l1_index = j
						if(np.amax(np.abs(adv[j]-inputs[j])) < dist_linf):
							dist_linf = np.amax(np.abs(adv[j]-inputs[j]))
							dist_linf_index = j
						if((np.sum((adv[j]-inputs[j])**2)**.5) < dist_l2):
							dist_l2 = (np.sum((adv[j]-inputs[j])**2)**.5)
							dist_l2_index = j
				if(dist_l1_index != 1e10):
					d_best_l2.append((np.sum((adv[dist_l2_index]-inputs[dist_l2_index])**2)**.5))
					d_best_l1.append(np.sum(np.abs(adv[dist_l1_index]-inputs[dist_l1_index])))
					d_best_linf.append(np.amax(np.abs(adv[dist_linf_index]-inputs[dist_linf_index])))
					r_best.append(1)
				else:
					r_best.append(0)

				rand_int = np.random.randint(i,i+num_targets)
				if inception:
					pred_r = np.reshape(model.model.predict(adv[rand_int:rand_int+1]), (data.test_labels[0:1].shape))
				else:
					pred_r = model.model.predict(adv[rand_int:rand_int+1])
				success_average = False
				if(args['untargeted']):			
					if(np.argmax(pred_r,1) != np.argmax(targets[rand_int:rand_int+1],1)):
						success_average = True
				else:
					if(np.argmax(pred_r,1) == np.argmax(targets[rand_int:rand_int+1],1)):
						success_average = True
				if success_average:				
					r_average.append(1)
					d_average_l2.append(np.sum((adv[rand_int]-inputs[rand_int])**2)**.5)
					d_average_l1.append(np.sum(np.abs(adv[rand_int]-inputs[rand_int])))
					d_average_linf.append(np.amax(np.abs(adv[rand_int]-inputs[rand_int])))

				else:
					r_average.append(0)

				dist_l1 = 0
				dist_l1_index = 1e10
				dist_linf = 0
				dist_linf_index = 1e10
				dist_l2 = 0
				dist_l2_index = 1e10
				for k,j in enumerate(range(i,i+num_targets)):
					failure = True
					if(args['untargeted']):
						if(np.argmax(pred[k],1) != np.argmax(targets[j:j+1],1)):
							failure = False
					else:
						if(np.argmax(pred[k],1) == np.argmax(targets[j:j+1],1)):
							failure = False
					if failure:
						r_worst.append(0)
						dist_l1_index = 1e10
						dist_l2_index = 1e10
						dist_linf_index = 1e10
						break
					else:
						if(np.sum(np.abs(adv[j]-inputs[j])) > dist_l1):
							dist_l1 = np.sum(np.abs(adv[j]-inputs[j]))
							dist_l1_index = j
						if(np.amax(np.abs(adv[j]-inputs[j])) > dist_linf):
							dist_linf = np.amax(np.abs(adv[j]-inputs[j]))
							dist_linf_index = j
						if((np.sum((adv[j]-inputs[j])**2)**.5) > dist_l2):
							dist_l2 = (np.sum((adv[j]-inputs[j])**2)**.5)
							dist_l2_index = j
				if(dist_l1_index != 1e10):
					d_worst_l2.append((np.sum((adv[dist_l2_index]-inputs[dist_l2_index])**2)**.5))
					d_worst_l1.append(np.sum(np.abs(adv[dist_l1_index]-inputs[dist_l1_index])))
					d_worst_linf.append(np.amax(np.abs(adv[dist_linf_index]-inputs[dist_linf_index])))
					r_worst.append(1)

				if(args['show'] and m == (num_models-1)):
					for j in range(i,i+num_targets):
						target_id = np.argmax(targets[j:j+1],1)
						label_id = np.argmax(labels[j:j+1],1)
						prev_id = np.argmax(np.reshape(model.model.predict(inputs[j:j+1]),(data.test_labels[0:1].shape)),1)
						adv_id = np.argmax(np.reshape(model.model.predict(adv[j:j+1]),(data.test_labels[0:1].shape)),1)
						suffix = "id{}_seq{}_lbl{}_prev{}_adv{}_{}_l1_{:.3f}_l2_{:.3f}_linf_{:.3f}".format(true_ids[i],
							target_id,
							label_id,
							prev_id,
							adv_id, adv_id == target_id,
							np.sum(np.abs(adv[j]-inputs[j])), np.sum((adv[j]-inputs[j])**2)**.5, np.amax(np.abs(adv[j]-inputs[j])))

						show(inputs[j:j+1], str(args['save'])+"/"+str(args['dataset'])+"/"+str(args['attack'])+"/original_{}.png".format(suffix))
						show(adv[j:j+1], str(args['save'])+"/"+str(args['dataset'])+"/"+str(args['attack'])+"/adversarial_{}.png".format(suffix))
			if(m != (num_models - 1)):
				lbl = "Src_"
				if(num_models > 2):
					lbl += str(m) + "_"
			else:
				lbl = "Tgt_"
			if(num_targets > 1):
				print(lbl+'best_case_L1_mean', np.mean(d_best_l1))
				print(lbl+'best_case_L2_mean', np.mean(d_best_l2))
				print(lbl+'best_case_Linf_mean', np.mean(d_best_linf))
				print(lbl+'best_case_prob', np.mean(r_best))
				print(lbl+'average_case_L1_mean', np.mean(d_average_l1))
				print(lbl+'average_case_L2_mean', np.mean(d_average_l2))
				print(lbl+'average_case_Linf_mean', np.mean(d_average_linf))
				print(lbl+'average_case_prob', np.mean(r_average))
				print(lbl+'worst_case_L1_mean', np.mean(d_worst_l1))
				print(lbl+'worst_case_L2_mean', np.mean(d_worst_l2))
				print(lbl+'worst_case_Linf_mean', np.mean(d_worst_linf))
				print(lbl+'worst_case_prob', np.mean(r_worst))
			else:
				print(lbl+'L1_mean', np.mean(d_average_l1))
				print(lbl+'L2_mean', np.mean(d_average_l2))
				print(lbl+'Linf_mean', np.mean(d_average_linf))
				print(lbl+'success_prob', np.mean(r_average))		

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("-d", "--dataset", choices=["mnist", "cifar", "imagenet"], default="mnist", help="dataset to use")
	parser.add_argument("-u", "--untargeted", action='store_true', help= "run non-targeted instead of targeted attack")
	parser.add_argument("-tg", "--targetnum", type=int, default=1, help= "number of targets per sample")
	parser.add_argument("-tm", "--targetmodel", choices=["same","dd_100"], default="same", help="target model of attack")
	parser.add_argument("-tr", "--train", action='store_true', help="save adversarial images generated from train set")
	parser.add_argument("-tp", "--temp", type=int, default=0, 
		help="attack defensively distilled network trained with this temperature")
	parser.add_argument("-adv", "--adversarial", choices=["none","l2","l1","en", "l2l1", "l2en"], default="none", 
		help="attack network adversarially trained under these examples")
	parser.add_argument("-s", "--save", default="./saves", help="save directory")
	parser.add_argument("-a", "--attack", choices=["L2", "L1", "EN", "IFGSM", "IFGML1", "IFGML2", "FGSM", "FGML1", "FGML2"], default="EN", help="attack algorithm")
	parser.add_argument("-n", "--numimg", type=int, default=1000, help = "number of images to attack")
	parser.add_argument("-m", "--maxiter", type=int, default=1000, help = "max iterations per bss")
	parser.add_argument("-bs", "--binary_steps", type=int, default=9, help = "number of bss")
	parser.add_argument("-b", "--batch_size", type=int, default=1, help= "batch size")
	parser.add_argument("-ae", "--abort_early", action='store_true', help="abort binary search step early when losses stop decreasing")
	parser.add_argument("-cf", "--conf", type=int, default=0, help='Set confidence score margin')
	parser.add_argument("-ic", "--init_const", type=float, default=1e-3, help='tradeoff constant')
	parser.add_argument("-be", "--beta", type=float, default=1e-2, help='beta hyperparameter')
	parser.add_argument("-ep", "--eps", type=float, default=0., help='eps hyperparameter (if 0, find lowest eps where example is successful')	
	parser.add_argument("-sh", "--show", action='store_true', help='save original and adversarial images to save directory')
	parser.add_argument("-sn", "--save_np", action='store_true', help='save adversarial examples for evaluation')
	parser.add_argument("-r", "--restore_np", action='store_true', help='restore saved adversarial examples for evaluation')
	parser.add_argument("-sd", "--seed", type=int, default=3, help='random seed for generate_data')
	parser.add_argument("-imgsd", "--seed_imagenet", type=int, default=4, help='random seed for pulling images from ImageNet test set')
	args = vars(parser.parse_args())
	print(args)
	main(args)
