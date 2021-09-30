# _SmartRec_: A Holistic Approach for Reinforcement Learning Based Recommender System

## Overview
In this library, we present _SmartRec_, a holistic approach for reinforcement learning-based recommender systems. The library is experimental and is created to encourage collaborations between fellow researchers and industrial practitioners. 

The library consists of the following major parts. 
Tabular Data Mining toolkit: a component combining essential tabular data mining toolkit based on classical methods, especially lightGBM and XgBoost.
Deep Learning for Tabular Data: a section to improve the SOTA of existing deep learning methods for tabular data. 
Training Methods: a module to train supervised, semi-supervised and unsupervised models. 
Model-based Reinforcement Learning: model-based reinforcement learning mostly based on [DreamerV2](https://arxiv.org/abs/2010.02193).
RNN module: a module that collects common RNN structures for sequential structures used in reinforcement learning.
Unsupervised Encoding Refinement: a module that collects common unsupervised refinement methods. 
NAS module: An module based mostly on differentiable architecture search. 

## Details
### Tabular Data Mining toolkit
Wrapper for common models (XgBoost, LightGBM, CatBoost, and other Sklearn built-in models);
Feature generation from tree-based models: XgBoost,  LightGBM and IsolationForest);
Common encoders with unified naming conventions;
Automatic hyperparameter search methods based on [hyperopt](https://github.com/hyperopt/hyperopt)
### Deep Learning for Tabular Data
[Performer](https://arxiv.org/abs/2009.14794), with [Entmax-alpha](https://arxiv.org/abs/1905.05702) as attention activation functions.
Other residual connection methods, see [this](https://arxiv.org/abs/2003.04887) for example.
Other batch normalization methods, see [this](https://arxiv.org/abs/1906.03548) for example. 
Mixture-of-experts methods, see [this]()
[Lookahead](https://arxiv.org/abs/1907.08610) optimizer.
Pretraining methods based on masked language model.
Meta-learning methods for finding the correct sparsity. 

### Training Methods
Common adversarial training methods borrowed from [PyTorch Adversarial](https://github.com/Harry24k/adversarial-attacks-pytorch)
Common semi-supervised consistency regularization methods from [PyTorch Consistency](https://github.com/perrying/pytorch-consistency-regularization)
Triplet Mining and Contrast Learning Approaches, such as [SimpleCLR](https://arxiv.org/abs/2002.05709) and [MocoV2]()
### RNN Module
A matching module based on [compositional attention networks](https://arxiv.org/abs/1803.03067) to combine static information with dynamic information. 
An improved RNN-version based on [Hierachical Attention Network](https://arxiv.org/abs/2005.12981);
A multi-matching perspective based on cross attentions as the final head. 
### NAS Module

A differentiable search based on [DARTS](https://arxiv.org/abs/1806.09055) and [RobustSearch](https://arxiv.org/abs/1910.04465). The module will also offer permissible node definitions. 



