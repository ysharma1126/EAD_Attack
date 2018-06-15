**As requested by IBM, this repository is temporarily closed because
part of the source code contains my internship work at IBM Research. We are working
hard on obtaining a clearance for releasing the code as open-source software.
Please contact Yash Sharma \<ysharma1126@gmail.com\> for the source code.**

EAD: Elastic-Net Attacks to Deep Neural Networks 
=====================================

EAD is a **e**lastic-net **a**ttack to **d**eep neural networks (DNNs).  
We propose formulating the attack process as a elastic-net regularized optimization problem, featuring an attack which produces L1-oriented adversarial examples which includes the state-of-the-art L2 attack (C&W) as a special case. 

Experimental results on MNIST, CIFAR-10, and ImageNet show that EAD yields a distinct set of adversarial examples and attains similar attack performance to state-of-the-art methods in different attack scenarios. More importantly, EAD leads to improved attack transferability and complements adversarial training for DNNs, suggesting novel insights on leveraging L1 distortion in generating robust adversarial examples. 

For more details, please see our paper:

[EAD: Elastic-Net Attacks to Deep Neural Networks via Adversarial Examples](https://arxiv.org/abs/1709.04114)
by Yash Sharma\*, Pin-Yu Chen\*, Huan Zhang, Jinfeng Yi, Cho-Jui Hsieh (AAAI 2018)

\* Equal contribution

The attack has also been used in the following works (incomplete):

[Attacking the Madry Defense Model with L1-based Adversarial Examples](https://arxiv.org/abs/1710.10733)
by Yash Sharma, Pin-Yu Chen (ICLR 2018 Workshop)

[Bypassing Feature Squeezing by Increasing Adversary Strength](https://arxiv.org/abs/1803.09868)
by Yash Sharma, Pin-Yu Chen

[On the Limitation of MagNet Defense against L1-based Adversarial Examples](https://arxiv.org/abs/1805.00310)
by Pei-Hsuan Lu, Pin-Yu Chen, Kang-Cheng Chen, Chia-Mu Yu (IEEE/IFIP DSN 2018 Workshop)

The experiment code is based on Carlini and Wagner's L2 attack. 
The attack (with EN rule) can also be found in the [Cleverhans Repository](http://cleverhans.readthedocs.io/en/latest/_modules/cleverhans/attacks.html#ElasticNetMethod).