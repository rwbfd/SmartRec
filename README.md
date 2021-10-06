# _SmartRec_: A Holistic Approach for Reinforcement Learning Based Recommender System

## Overview
In this library, we present _SmartRec_, a holistic approach for reinforcement learning-based recommender systems. The library is experimental and is created to encourage collaborations between fellow researchers and industrial practitioners. 

The library consists of the following major parts. 
1. Tabular Data Mining toolkit: a component combining essential tabular data mining toolkit based on classical methods, especially lightGBM and XgBoost.
2. Deep Learning for Tabular Data: a section to improve the SOTA of existing deep learning methods for tabular data. 
3. Training Methods: a module to train supervised, semi-supervised and unsupervised models. 
4. Model-based Reinforcement Learning: model-based reinforcement learning mostly based on [DreamerV2](https://arxiv.org/abs/2010.02193).
5. RNN module: a module that collects common RNN structures for sequential structures used in reinforcement learning.
6. Unsupervised Encoding Refinement: a module that collects common unsupervised refinement methods. 
7. NAS module: An module based mostly on differentiable architecture search. 

## Details
### Tabular Data Mining toolkit
1. Wrapper for common models (XgBoost, LightGBM, CatBoost, and other Sklearn built-in models);
2. Feature generation from tree-based models: XgBoost,  LightGBM and IsolationForest);
3. Common encoders with unified naming conventions;
4. Automatic hyperparameter search methods based on [hyperopt](https://github.com/hyperopt/hyperopt)

### Deep Learning for Tabular Data
1. [Performer](https://arxiv.org/abs/2009.14794), with [Entmax-alpha](https://arxiv.org/abs/1905.05702) as attention activation functions.
2. Other residual connection methods, see [this](https://arxiv.org/abs/2003.04887) for example.
3. Other batch normalization methods, see [this](https://arxiv.org/abs/1906.03548) for example. 
4. Mixture-of-experts methods, see [this]()
5. [Lookahead](https://arxiv.org/abs/1907.08610) optimizer.
6. Pretraining methods based on masked language model.
7. Meta-learning methods for finding the correct sparsity. 

### Model-based Reinforcement Learning
1. A model-based RL method with VAE as world model and actor-critic as policy[dreamerV2](https://arxiv.org/abs/2010.02193)
2. An old method using VAE as world model but CEM as policy[PlaNet](https://arxiv.org/pdf/1811.04551.pdf)
3. An achiever and explorer based reward free model based RL method[Achiever](https://danijar.com/asset/lexa/paper.pdf)
4. Many tries on model-based RL design(https://arxiv.org/pdf/2012.04603.pdf)
5. Planning to explore via self-supervised world model, aka, no reward is needed[Plan2Explore](https://arxiv.org/pdf/2005.05960.pdf)
6. Joint KL divergence to adjust model-based RL reward(https://arxiv.org/pdf/2009.01791.pdf)
7. Explore and control with adversarial surprise(https://arxiv.org/pdf/2107.07394.pdf)

### Training Methods
1. Common adversarial training methods borrowed from [PyTorch Adversarial](https://github.com/Harry24k/adversarial-attacks-pytorch)
2. Common semi-supervised consistency regularization methods from [PyTorch Consistency](https://github.com/perrying/pytorch-consistency-regularization)
3. Triplet Mining and Contrast Learning Approaches, such as [SimpleCLR](https://arxiv.org/abs/2002.05709) and [MocoV2](https://arxiv.org/abs/2003.04297)

### RNN Module
1. A matching module based on [compositional attention networks] (https://arxiv.org/abs/1803.03067) to combine static information with dynamic information. 
2. An improved RNN-version based on [Hierachical Attention Network](https://arxiv.org/abs/2005.12981);
3. A multi-matching perspective based on cross attentions as the final head. 

### Unsupervised Encoding Refinement:
1. Best contrastive Learning methods to extract features in an unsupervised way [SimCLR](https://arxiv.org/pdf/2002.05709.pdf)
2. Unsupervised Learning methods for feature extraction by contrastive clustering assignment [SwA](https://arxiv.org/pdf/2006.09882.pdf)
3. Unsupervised feature extraction through network boostrap [BYOL](https://arxiv.org/pdf/2006.07733.pdf)
4. Feature extraction through visual transformer [ViT](https://arxiv.org/pdf/2104.02057.pdf)
5. Another Feature extraction through asymmetric contrastive learning [MoCoV3](https://arxiv.org/pdf/2003.04297.pdf)
6. Normalizing Flow method to learn latent space feature [RealNVP](https://arxiv.org/pdf/1605.08803.pdf)
7. Residual Flow method to learn latent space feature [ResFlow](https://arxiv.org/pdf/1906.02735.pdf)
8. Neural ODE method to learn latent space features [NeuralODE](https://arxiv.org/pdf/1806.07366.pdf)
9. A more powerful Neural ODE model with Stochastic loss estimation [FFJORD](https://arxiv.org/pdf/1810.01367.pdf)
10. Diffusion process augmented method to learn latent features [Diffusion Model](https://arxiv.org/pdf/2011.13456.pdf)

### NAS Module

A differentiable search based on [DARTS](https://arxiv.org/abs/1806.09055) and [RobustSearch](https://arxiv.org/abs/1910.04465). The module will also offer permissible node definitions. 



