# Mask Detector
# A Practice applying Neural Network for Object Detection
# Author: Yangjia Li (Francis)
# Date: Mar. 29, 2021
# Last Modeified: 
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

""" General Idea

Locate objects in the image; apply image classifier for labelling each object

Two main approaches for locating objects
- Single-shot detection. Ex. YOLO
    Higher speed
- Two-shot detection. Apply regional proposal model before detection.
    Better performance

Hybrid model like R-FCN (Region-Based Fully Convolutional Networks) is also recommended.
Basically a TSD architecture with some SSD features



"""
# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
