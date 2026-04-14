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
import os
import cv2


#FGSM attack implementation
def fgsm_attack(model, image, label, epsilon, criterion, clamp_min=-3.0, clamp_max=3.0):
    """
    image: torch.Tensor (1,3,224,224)
    label: torch.Tensor (1,)
    epsilon: float
    criterion: nn.CrossEntropyLoss
    clamp_min: float
    clamp_max: float
    """
    image = image.clone().detach().requires_grad_(True)

    # Forward pass
    output = model(image)
    loss = criterion(output, label)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # FGSM perturbation
    # image.grad.sign() shape is (1,3,224,224)
    adv_image = image + epsilon * image.grad.sign()

    # Clamp (post-normalization space)
    # FGSM can push values outside a reasonable range, so we clamp to [-3, 3] which is a common range for normalized images
    # Normalized pixel values commonly fall in approximately [-2.5, +2.5]
    adv_image = torch.clamp(adv_image, clamp_min, clamp_max)

    return adv_image.detach()


def pgd_attack(model, image, label, epsilon, alpha, num_steps, criterion,
               clamp_min=-3.0, clamp_max=3.0):
    original = image.clone().detach()
    adv_image = original.clone()

    for _ in range(num_steps):
        adv_image.requires_grad_(True)

        output = model(adv_image)
        loss = criterion(output, label)

        model.zero_grad()
        loss.backward()

        adv_image = adv_image + alpha * adv_image.grad.sign()

        # Project to ε-ball
        delta = torch.clamp(
            adv_image - original,
            min=-epsilon,
            max=epsilon
        )
        adv_image = torch.clamp(original + delta, clamp_min, clamp_max).detach()

    return adv_image


# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images, initially empty
    labels = [] # corresponding labels, initially empty
    gtFile = open(os.path.join(rootpath, 'GT-final_test.csv'), 'r')
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    next(gtReader) # skip header
    # loop over all images in current annotations file
    for row in gtReader:
        images.append(cv2.imread(os.path.join(rootpath, row[0]))) # the 1th column is the filename
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


print("Reading testing data")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, 'GTSRB/Test/Final_Test/Images')
testImages, testLabels = readTrafficSigns(path) 

###############################################
#Preprocess the images and labels for training#
test_dataset = GTSRBDataset(testImages, testLabels, target_size=224)
print(f"Total testing samples: {len(test_dataset)}")
print(f"Example image shape after preprocessing: {test_dataset[0][0].shape}, label: {test_dataset[0][1]}")


# Model setup
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 43) # 43 classes in GTSRB
model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'resnet18_gtsrb.pth')))  # Load the trained model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(device)

epsilon = 0.03
alpha = epsilon / 4
num_steps = 10
criterion = nn.CrossEntropyLoss()
attack = "FGSM"  
# attack = "PGD"  
# Generate adversarial examples for all test samples and extract true labels for evaluation
examples_adv = []
true_labels = []
for i in range(len(test_dataset)):
    img, label = test_dataset[i]
    img = img.unsqueeze(0)  # Add batch dimension
    label = torch.tensor([label])  # Convert to tensor
    img = img.to(device)
    label = label.to(device)

    if attack == "FGSM":
        adv_img = fgsm_attack(model, img, label, epsilon, criterion)
    elif attack == "PGD":
        adv_img = pgd_attack(model, img, label, epsilon, alpha, num_steps, criterion)
    
    examples_adv.append(adv_img.detach().cpu())
    true_labels.append(label.detach().cpu())  

# Clean accuracy on original test set, baseline accuracy before attack

predictions_clean =[]
correct_clean = 0
total_clean = len(test_dataset)
with torch.no_grad():
    for i in range(total_clean):
        img_clean, label_clean = test_dataset[i]
        img_clean = img_clean.unsqueeze(0).to(device)
        label_clean = torch.tensor([label_clean]).to(device)
        output_clean = model(img_clean)
        # predicted_clean is the predicted class index, shape (1,)
        predicted_clean = output_clean.argmax(dim=1)
        if predicted_clean.item() == label_clean.item():
            correct_clean += 1
        predictions_clean.append(predicted_clean.detach().cpu())
clean_accuracy = correct_clean / total_clean
print(f"Clean accuracy: {clean_accuracy * 100:.2f}%")  


# Adversarial accuracy on the generated adversarial examples
predictions_adv =[]
correct_adv = 0
total_adv = len(examples_adv)
with torch.no_grad():
    for img_adv, label_adv in zip(examples_adv, true_labels):
        img_adv = img_adv.to(device)
        label_adv = label_adv.to(device)

        output_adv = model(img_adv)
        predicted_adv = output_adv.argmax(dim=1)

        if predicted_adv.item() == label_adv.item():
            correct_adv += 1
        predictions_adv.append(predicted_adv.detach().cpu())

adv_accuracy = correct_adv / total_adv
print(f"Adversarial accuracy: {adv_accuracy * 100:.2f}%")


# Attack Success Rate (ASR) = (Number of successful attacks) / (Number of samples that were correctly classified before attack)
successful_attacks = 0
correct_initially = 0
for i in range(len(test_dataset)):
    if predictions_clean[i].item() == true_labels[i].item():
        correct_initially += 1
        if predictions_adv[i].item() != true_labels[i].item():
            successful_attacks += 1

attack_success_rate = (successful_attacks / correct_initially if correct_initially > 0 else 0.0)

print(f"Attack success rate: {attack_success_rate * 100:.2f}%")

