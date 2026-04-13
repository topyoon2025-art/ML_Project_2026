# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import matplotlib.pyplot as plt
import csv

#resnet18 for classification
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images, initially empty
    labels = [] # corresponding labels, initially empty
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels

def letterbox_resize(img, target_size=224, fill=0):
    """
    Letterbox resize for a single image tensor.

    Args:
        img (torch.Tensor): shape (3, H, W), float32 in [0, 1]
        target_size (int): output height and width (square), 224x224 for ResNet-18
        fill: padding color (0 = black), default is 0 for black padding

    Returns:
        torch.Tensor: shape (3, target_size, target_size)
    """
    # img in shape of (C, H, W)
    _, h, w = img.shape

    # Compute scale while preserving aspect ratio
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    img = F.resize(img, [new_h, new_w])  # (3, new_h, new_w)

    # Compute padding (left, top, right, bottom)
    pad_left   = (target_size - new_w) // 2
    pad_right  = target_size - new_w - pad_left
    pad_top    = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top

    # Apply padding
    img = F.pad(img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=fill)

    return img


class GTSRBDataset(Dataset):
    def __init__(self, images, labels, target_size=224):
        self.images = images
        self.labels = labels
        self.target_size = target_size

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])

        img = (
            torch.from_numpy(img.copy())
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        img = letterbox_resize(img, self.target_size)
        img = (img - self.mean) / self.std

        return img, label

    def __len__(self):
        return len(self.images)


print("Reading training data")
path = 'GTSRB/Training/Final_Training/Images'
trainImages, trainLabels = readTrafficSigns(path) #39209, 39209

###############################################
#Preprocess the images and labels for training#
train_dataset = GTSRBDataset(trainImages, trainLabels, target_size=224)
print(f"Total training samples: {len(train_dataset)}")
print(f"Example image shape after preprocessing: {train_dataset[0][0].shape}, label: {train_dataset[0][1]}")

#Use the batch size of 32 or 64 for training
batch_size_train = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)

#Initialization of the model, loss function, and optimizer
# Load the pre-trained ResNet-18 model with imagenet weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights) 
# Take care of input and output dimensions for the model
# GTSRB has 43 classes
num_classes = 43
# Modify the model.fc layer to output 43 classes using nn.linear
# for ResNet-18, the input features to the final layer is 512
# Loss function and optimizer: CrossEntropyLoss and Adam
model.fc = nn.Linear(model.fc.in_features, num_classes)

criterion = nn.CrossEntropyLoss()
# Try different learning rates like 0.01, 0.1, etc.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

epochs = 5
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'resnet18_gtsrb.pth')
print("Training complete. Model saved as resnet18_gtsrb.pth")

