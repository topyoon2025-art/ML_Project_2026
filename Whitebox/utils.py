#resnet18 for classification
from cProfile import label

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
import csv
import torch.nn as nn
import torchvision.models as models

class GTSRBDataset(Dataset):
    def __init__(self, images, labels, target_size=224):
        self.images = images
        self.labels = labels
        self.target_size = target_size

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

        return img, torch.tensor(label, dtype=torch.long)
        #return img, label

    def __len__(self):
        return len(self.images)

def normalize_batch(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    return (x - mean) / std 

class ResNet18_L3(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        base = models.resnet18(weights=weights)

        # ---- Encoder (up to layer3) ----
        self.encode_layers = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,   # ← latent space
        )

        # ---- Decoder (layer4 → avgpool → fc) ----
        self.decode_layers = nn.Sequential(
            base.layer4,
            base.avgpool,
        )

        self.fc = nn.Linear(base.fc.in_features, num_classes)  # 512 → num_classes

    def encode(self, x):
        return self.encode_layers(x)  # output of layer3

    def decode(self, z):
        x = self.decode_layers(z)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def forward(self, x):
        return self.decode(self.encode(x))
    

def readTrafficSigns_train(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images, initially empty
    labels = [] # corresponding labels, initially empty
    # loop over all 42 classes
    for c in range(0,43):
        prefix = os.path.join(rootpath, format(c, '05d'), '') # subdirectory for class
        gtFile = open(os.path.join(prefix, 'GT-' + format(c, '05d') + '.csv'), 'r') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(cv2.imread(os.path.join(prefix, row[0])).copy()) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels

def readTrafficSigns_test(rootpath):
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

def letterbox_resize(img, target_size=224, fill=0): #fill 0 for padding, 0 black color
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

def fgsm_attack(model, image, label, epsilon, criterion, device):

    adv_image = image.clone().detach().requires_grad_(True)

    # Forward pass
    output = model(normalize_batch(adv_image))
    loss = criterion(output, label)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # FGSM update in pixel space
    adv_image = adv_image + epsilon * adv_image.grad.sign()
    adv_image = adv_image.clamp(0.0, 1.0).detach()  # Ensure valid pixel range and detach from graph
    
    return adv_image


def pgd_attack(model, image, label, epsilon, alpha, num_steps, criterion, device):
    original = image.clone().detach()
    adv_image = original.clone().detach()

    for _ in range(num_steps):
        adv_image.requires_grad_(True)

        output = model(normalize_batch(adv_image))
        loss = criterion(output, label)

        model.zero_grad()
        loss.backward()

        with torch.no_grad(): # no_grad for projection and clamping
            adv_image = adv_image + alpha * adv_image.grad.sign()
            # Project to ε-ball
            delta = torch.clamp(
                adv_image - original,
                min=-epsilon,
                max=epsilon
            )
            adv_image = original + delta
            # clip to valid (normalized) image range
            adv_image = adv_image.clamp(0.0, 1.0)
            # detach so next loop starts from a leaf node without history
        adv_image = adv_image.detach_()

    return adv_image

def pgd_attack_latent_l3(model, image, label, epsilon, alpha, num_steps, criterion, device):
    """
    Latent PGD attack on ResNet-18 layer3 activations.
    model: must have model.encode(x) and model.decode(z)
    """

    image = image.to(device)
    label = label.to(device)

    # 1. Encode to latent space (layer3 output)
    with torch.no_grad():
        z0 = model.encode(normalize_batch(image))   # shape: (B, 256, H, W)

    # 2. Initialize adversarial latent
    z_adv = z0.clone().detach().requires_grad_(True)

    for _ in range(num_steps):
        # 3. Forward through the rest of the network
        logits = model.decode(z_adv)
        loss = criterion(logits, label)

        # 4. Compute gradient wrt latent
        grad = torch.autograd.grad(loss, z_adv)[0]

        # 5. PGD update in latent space (L∞)
        z_adv = z_adv + alpha * grad.sign()

        # 6. Project back to ε-ball around z0
        delta = torch.clamp(z_adv - z0, min=-epsilon, max=epsilon)
        z_adv = (z0 + delta).detach().requires_grad_(True)

    return z_adv

def generate_adversarial_examples_batched(
    model,
    test_loader,
    attack,
    device,
    epsilon,
    criterion,
    alpha=None,
    num_steps=None
):
    model.eval()

    examples_adv = []
    true_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        if attack == "FGSM":
            adv_images = fgsm_attack(model, images, labels, epsilon, criterion, device)
        elif attack == "PGD":
            adv_images = pgd_attack(model, images, labels, epsilon, alpha, num_steps, criterion, device)
        else:
            raise ValueError(f"Unknown attack type: {attack}")

        examples_adv.append(adv_images.detach().cpu())
        true_labels.append(labels.detach().cpu())

    # Concatenate all batches into single tensors
    examples_adv = torch.cat(examples_adv, dim=0)
    true_labels = torch.cat(true_labels, dim=0)

    return examples_adv, true_labels

def evaluate_clean_accuracy(model, test_loader, device):
    model.eval()

    predictions_clean = []
    correct_clean = 0
    total_clean = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(normalize_batch(images))
            predicted = outputs.argmax(dim=1)

            correct_clean += (predicted == labels).sum().item()
            total_clean += labels.size(0)

            predictions_clean.append(predicted.detach().cpu())

    predictions_clean = torch.cat(predictions_clean, dim=0)
    clean_accuracy = correct_clean / total_clean

    return clean_accuracy, predictions_clean

def evaluate_adversarial_accuracy(model, adv_loader, device):
    model.eval()

    predictions_adv = []
    wrong_indices = []
    wrong_true_labels = []
    wrong_pred_labels = []

    correct_adv = 0
    total_adv = 0
    idx_offset = 0

    with torch.no_grad():
        for imgs_adv, labels_adv in adv_loader:
            imgs_adv = imgs_adv.to(device)
            labels_adv = labels_adv.to(device)

            outputs = model(normalize_batch(imgs_adv))
            predicted = outputs.argmax(dim=1)

            correct_adv += (predicted == labels_adv).sum().item()
            total_adv += labels_adv.size(0)

            wrong_mask = predicted.ne(labels_adv)
            wrong_idx  = wrong_mask.nonzero(as_tuple=True)[0]

            wrong_indices.extend((wrong_idx + idx_offset).tolist())
            wrong_true_labels.extend(labels_adv[wrong_idx].cpu().tolist())
            wrong_pred_labels.extend(predicted[wrong_idx].cpu().tolist())

            predictions_adv.append(predicted.detach().cpu())
            idx_offset += labels_adv.size(0)

    predictions_adv = torch.cat(predictions_adv, dim=0)
    adv_accuracy = correct_adv / total_adv

    return adv_accuracy, predictions_adv, wrong_indices, wrong_true_labels, wrong_pred_labels

def compute_attack_success_rate(predictions_clean, predictions_adv, true_labels):
    successful_attacks = 0
    correct_initially = 0

    for i in range(len(true_labels)):
        if predictions_clean[i].item() == true_labels[i].item():
            correct_initially += 1
            if predictions_adv[i].item() != true_labels[i].item():
                successful_attacks += 1

    attack_success_rate = (
        successful_attacks / correct_initially if correct_initially > 0 else 0.0
    )

    return attack_success_rate, successful_attacks, correct_initially


def generate_adversarial_batch(
    model,
    images,
    labels,
    attack,
    device,
    epsilon,
    criterion,
    alpha=None,
    num_steps=None
):
    """
    Generate adversarial examples for a single batch during training.
    This version is differentiable and safe to use inside the training loop.
    """

    images = images.to(device)
    labels = labels.to(device)

    if attack == "FGSM":
        adv_images = fgsm_attack(
            model=model,
            image=images,
            label=labels,
            epsilon=epsilon,
            criterion=criterion,
            device=device
        )

    elif attack == "PGD":
        adv_images = pgd_attack(
            model=model,
            image=images,
            label=labels,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            criterion=criterion,
            device=device
        )

    else:
        raise ValueError(f"Unknown attack type: {attack}")

    return adv_images