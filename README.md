# Python Deep Learning Project: Cat Detection with CNN and ResNet

## 0. Basic Information

**Project name:** Cat Detection with CNN and ResNet

**Version:** v1.0

**Author:** YutianW

**Date:** 2021 August

## 1. Introduction
This is a project that uses Deep Learning to recognize cats. We used Python PyTorch to train a Deep Neural Network including CNN and Residual Network. The Model achieves 92% accuracy on the individual test set of 1000 images. 

## 2. Dataset
The project used CIFAR-10 dataset containing 60,000 images of 32 * 32 RGB pixels. The training set contains 50,000 pictures and test set contains 10,000. 

The link to the CIFAR-10 dataset can be found [here](https://link-url-here.org).

## 3. Model
Given an input of a 3d array of RBG values of all pixels in the 32 * 32 picture, the model would return a float value between 0 and 1, where larger value indicates the input picture is more likely to be a cat. 

The Model consists of a Residual Network and several layers of common used convolutional networks together with pooling, batchnorm, etc. 

More information about the model can be found in the [model.py](/model.py) file. 

## 4. Training
We trained the model with cuda GPU for about half an hour. We trained the 50,000 training pictures with 8 epochs. 

The complete training process can be found in the Jupyter notebook file [catproj.ipynb](/catproj.ipynb).

## 5. Accuracy report
We tested our trained model on the 10,000 training set given by CIFAR-10. 

The testing result shows a 92% accuracy among the 10,000 test image. Same as intuition, a correct result means either the model considers a cat-picture as a cat, or saying a non-cat picture is not a cat. 

## 6. Files included in this repository
- *README.md*: This file.
- *catproj.ipynb*: a ipython notebook consisting exploring the dataset, building the model, training the data, and testing the data. Comments and interpretations are included in the file. 
- *model.py*: a python file consists of only the model itself. 
- *util.py*: a python file that can be used to test the model on customized pictures. Only a picture PATH is needed, and its functions will download the pictures, resize it to 32 * 32 pixels, change it into model format, pass into the model and return a human-readable output. 
- *cifar_resnet_binary08.pth*: the file storing the pre-trained weights in the deep network. It will be loaded automatically by the model when needed so the model does not need to be retrained everytime. Not a human-readable file.  

## 7. Frameworks and references
The frameworks used in this project including:
- Numpy: version 1.19.5
- PyTorch (with cuda): version 1.9.0+cu102
- TorchVision: version 0.10.0+cu102
- Pillow: version 7.1.2
- Matplotlib: version 3.2.2

The project was originally written with [Google Colab](https://colab.research.google.com/). 

References:
- https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- https://www.cs.toronto.edu/~kriz/cifar.html
- https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278


