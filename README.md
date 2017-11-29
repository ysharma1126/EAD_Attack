EAD: Elastic-Net Attacks to Deep Neural Networks 
=====================================

EAD is a **e**lastic-net **a**ttack to **d**eep neural networks (DNNs).  
We propose formulating the attack process as a elastic-net regularized optimization problem, featuring an attack which produces L1-oriented adversarial examples which includes the state-of-the-art L2 attack (C&W) as a special case. 

Experimental results on MNIST, CIFAR10, and ImageNet show that EAD can yield a distinct set of adversarial examples and attains similar attack performance to state-of-the-art methods in different attack scenarios. More importantly, EAD leads to improved attack transferability and complements adversarial training for DNNs, suggesting novel insights on leveraging L1 distortion in generating robust adversarial examples. 

For more details, please see our paper:

[EAD: Elastic-Net Attacks to Deep Neural Networks via Adversarial Examples](https://arxiv.org/abs/1709.04114)
by Yash Sharma\*, Pin-Yu Chen\*, Huan Zhang, Jinfeng Yi, Cho-Jui Hsieh (AAAI 2018)

\* Equal contribution

The experiment code is based on Carlini and Wagner's L2 attack. 
The inception model is updated to a new version (`inception_v3_2016_08_28.tar.gz`).


Setup and train models
-------------------------------------

The code is tested with python3 and TensorFlow v1.2 and v1.3. The following
packages are required:

```
sudo apt-get install python3-pip
sudo pip3 install --upgrade pip
sudo pip3 install pillow scipy numpy tensorflow-gpu keras h5py
```

Prepare the MNIST and CIFAR-10 data and models for attack:

```
python3 train_models.py
```

To download the inception model:

```
python3 setup_inception.py
```

To prepare the ImageNet dataset, download and unzip the following archive:

[ImageNet Test Set](http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/img.tar.gz)


and put the `imgs` folder in `../imagesnetdata`. This path can be changed
in `setup_inception.py`.

Train defensively distilled models
-------------------------------------

Train defensively distilled MNIST and CIFAR-10 models with temperature varying from 1 to 100:

```
python3 train_models.py -dd
```

Train defensively distilled MNIST and CIFAR-10 models under specified temperatures:

```
python3 train_models.py -dd -t 1 10 100
```

Run attacks
--------------------------------------

A unified attack interface, `test_attack.py` is provided. Run `python3 test_attack.py -h`
to get a list of arguments and help. Note the default values provided as well. 

The following are some examples of attacks:

Run the L1-oriented attack on the Inception model with 100 ImageNet images

```
python3 test_attack.py -a L1 -d imagenet -n 100
```

Run the EN-oriented attack on the defensively distilled (T=100) CIFAR model with 1000 images

```
python3 test_attack.py -d cifar -tp 100
```

Run the EN-oriented attack with 20 binary search steps, max 10,000 iterations each binary search step, and abort early (Greatly speeds up attack, hurts performance)

```
python3 test_attack.py -bs 20 -m 10000 -ae
```

Save original and adversarial images in the saves directory

```
python3 test_attack.py -sh
```

Generate adversarial images on undefended MNIST model with confidence (50), attack defensively distilled (T=100) MNIST model

```
python3 test_attack.py -cf 50
```

Adversarial Training
-------------------------------------

Adversarially train MNIST models by augmenting the training set with L2, EAD(L1), EAD(EN), L2+EAD(L1), and L2+EAD(EN)-based examples, respectively

```
python3 train_models.py -a
```

This will use the provided numpy save files in the train directory. 
Generate your own training set examples for use in adversarial training (ex - L1-oriented attack)

```
python3 test_attack.py -a L1 -tr
```

Now, attack an adversarially trained model (ex - L1-trained network)

```
python3 test_attack.py -adv l1
```
