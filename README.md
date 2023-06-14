# Geology-Constrained Dynamic Graph Convolutional Networks for Seismic Facies Classification
This repo will contain the codes used in the paper "Geology-Constrained Dynamic Graph Convolutional Networks for Seismic Facies Classification" submitted to Computers & Geosciences.

# Abstract
Knowing a land's facies type before drilling is an essential step in oil exploration. In seismic surveying, subsurface images are analyzed to segment and classify the facies in that volume. With the recent developments in deep learning, multiple works have utilized deep neural networks to classify facies from subsurface images. Unlike natural images, seismic data have different patterns and structures, which means that although general deep learning architectures can work with seismic data, it would be more effective if these architectures were optimized and refined specifically for such type of data. Most of the works in the seismic domain focus on convolution neural networks as the main backbone for the architectures, and more recently transformers started becoming more common in seismic data processing. Proposing a different approach that can capture unique correlations in the data, we introduce the use of dynamic graph convolutional networks as a method for capturing long-term dependencies for seismic facies classification. The proposed architecture combines the use of convolution neural networks and graph convolution networks to capture both global and local structures of the data. The performance of the model was evaluated on a facies classification dataset, and the proposed method provided state-of-the-art results while significantly reducing the number of parameters in the model compared to other architectures.

# Implementation
The implementation uses two codes as its main backbone, [Alaudah Facies Classification Benchmark](https://github.com/yalaudah/facies_classification_benchmark) and [Wang Dynamic Graph Convolutional Neural Networks](https://github.com/WangYueFt/dgcnn). To run this repository, install the prerequisites given in Alaudah's repository, and download the dataset as instructed by their repo.

Training the model:
```bash
python section_train.py --batch_size 21 --k 0.5 --depth_limit 1
```

Testing the model:
```bash
python section_test.py --model_path 'model_path/model.pkl'
```
