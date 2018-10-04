# pytorch-maml
This is a PyTorch implementation of the supervised learning experiments from the paper
Model-Agnostic Meta-Learning (MAML): https://arxiv.org/abs/1703.03400

*Important*: You will need the latest version of PyTorch, v.0.2.0 to run this code (otherwise you will get errors about
double backwards not being supported).

Currently, only the Omniglot experiments have been replicated here. The hyper-parameters are the same as those used in the original
Tensorflow implementation, except that only 1 random seed is used here.

5-way 1-shot training, best performance 98.9%

![Alt text](https://dl.dropboxusercontent.com/s/qz77zyz64d9xbhh/omniglot-5way-1shot.png?dl=0)

20-way 1-shot training, best performance 92%

![Alt text](https://dl.dropboxusercontent.com/s/9mqjieu64d01d4z/omniglot-20way-1shot.png?dl=0)

Note: the 20-way performance is slightly lower than that reported in the paper (they report 95.8%). If you can see why this might be,
please let me know. Also in this experiment, we can see evidence of overfitting to the meta-training set.

The 5-way results are achieved by simply meta-testing the network trained on the 1-shot task on the 5-shot task (e.g. for the 5-way 5-shot result, test the 5-way 1-shot trained network with 5-shots). Again the 20-way result is lower here than reported in the paper.

This repo also contains code for running maml experiments on permuted MNIST (tasks are created by shuffling the labels).
This is a nice sanity check task.

### license
This software is distributed under the MIT license.

### to-do
 - [ ] port to pytorch 0.4 from 0.2 and python 3 from 2
 - [ ] investigate performance difference from TF version
 - [ ] add first-order version
