
from xml.parsers.expat import model

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
import utils

def main():
    print("Reading training data...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, 'GTSRB/Training/Final_Training/Images')
    trainImages, trainLabels = utils.readTrafficSigns_train(path) #39209, 39209

    ###############################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Preprocess the images and labels for training#
    train_dataset = utils.GTSRBDataset(trainImages, trainLabels, target_size=224)
    print(f"Total training samples: {len(train_dataset)}")

    #Use the batch size of 32 or 64 for training
    batch_size_train = 64
    num_workers = 4
    pin_memory = True
    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size_train, 
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=pin_memory)

    #Initialization of the model, loss function, and optimizer
    # Load the pre-trained ResNet-18 model with imagenet weights
    num_classes = 43
    # weights = ResNet18_Weights.DEFAULT
    # model = resnet18(weights=weights) 
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
  
    model = utils.ResNet18_L3(num_classes=num_classes)
    model = model.to(device)
    print(f"Using device: {device}")
    criterion = nn.CrossEntropyLoss() # Cross-entropy loss for classification
    # Adam optimizer with weight decay for regularization 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 20

    # for epoch in range(epochs):
    #     training_correct = 0
    #     model.train() # Set model to training mode
    #     for images, labels in train_loader:
    #         images = images.to(device) # Send to GPU if available
    #         labels = labels.to(device) # Send to GPU if available
    #         optimizer.zero_grad() # Zero the gradients
    #         outputs = model(utils.normalize_batch(images)) # Forward pass
    #         loss = criterion(outputs, labels) # Compute loss
    #         loss.backward() # Backward pass
    #         optimizer.step() # Update weights
    #         # Calculate training accuracy
    #         _, predicted = torch.max(outputs.data, 1)
    #         training_correct += (predicted == labels).sum().item()

    #     print(f'Clean Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {100 * training_correct / len(train_dataset):.2f}%')
    # torch.save(model.state_dict(), os.path.join(BASE_DIR, 'resnet18_gtsrb_clean.pth'))
    # print("Clean training complete. Model saved as resnet18_gtsrb_clean.pth")

    
    # Adversarial training with input-level PGD (for comparison)
    model = utils.ResNet18_L3(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'resnet18_gtsrb_clean.pth')))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    epochs = 15
    lambda_clean = 1 / 2
    lambda_input_adv = 1 / 2
    for epoch in range(epochs):
        training_correct = 0
        model.train() 
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            clean_outputs = model(utils.normalize_batch(images))
            clean_loss = criterion(clean_outputs, labels)

            # Generate adversarial batch
            adv_input_images = utils.generate_adversarial_batch(
                model=model,
                images=images,
                labels=labels,
                attack="PGD",
                device=device,
                epsilon=8/255.0,
                alpha=2/255.0,
                num_steps=10,
                criterion=criterion
            )

            # Forward on adversarial examples
            adv_input_outputs = model(utils.normalize_batch(adv_input_images))
            adv_input_loss = criterion(adv_input_outputs, labels)

            
            # 3. Compute loss and update weights
            loss = (lambda_clean * clean_loss) + (lambda_input_adv * adv_input_loss) 
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()

            # 4. Accuracy (input-adv training accuracy)
            _, predicted = torch.max(adv_input_outputs.data, 1)
            training_correct += (predicted == labels).sum().item()

        print(f'Input-PGD Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, '
          f'Accuracy: {100 * training_correct / len(train_dataset):.2f}%')

    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'resnet18_gtsrb_input_adv.pth'))
    print("Saved input-PGD robust model.")


    # Adversarial training with latent-PGD (for comparison)
    model = utils.ResNet18_L3(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'resnet18_gtsrb_clean.pth')))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 15
    epsilon = 0.2
    lambda_latent_adv = 1 / 2   
    lambda_clean = 1 / 2

    for epoch in range(epochs):
        training_correct = 0
        model.train() 
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            clean_outputs = model(utils.normalize_batch(images))
            clean_loss = criterion(clean_outputs, labels)

            # 1. Generate latent adversarial representation
            z_adv = utils.pgd_attack_latent_l3(
                model=model,
                image=images,
                label=labels,
                epsilon=epsilon,#not adding same pixel-level perturbation, but only latent-level perturbation
                alpha=(epsilon / 4.0),
                num_steps=5,
                criterion=criterion,
                device=device
            )

            # 2. Forward pass using latent adversarial input
            adv_latent_outputs = model.decode(z_adv)
            adv_latent_loss = criterion(adv_latent_outputs, labels)

            # 3. Compute loss and update weights
            loss = (lambda_clean * clean_loss) + (lambda_latent_adv * adv_latent_loss)
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()

            # 4. Accuracy (latent-adv training accuracy)
            _, predicted = torch.max(adv_latent_outputs.data, 1)
            training_correct += (predicted == labels).sum().item()

        print(f'Latent-PGD Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, '
        f'Accuracy: {100 * training_correct / len(train_dataset):.2f}%')

    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'resnet18_gtsrb_latent_adv.pth'))
    print("Saved latent-PGD robust model.")

     # # Adversarial training with latent-PGD and input-level PGD (for comparison)
    model = utils.ResNet18_L3(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'resnet18_gtsrb_clean.pth')))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 15
    epsilon = 0.2
    lambda_latent_adv = 1 / 3   
    lambda_clean = 1 / 3
    lambda_input_adv = 1 / 3
    for epoch in range(epochs):
        training_correct = 0
        model.train() 
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            clean_outputs = model(utils.normalize_batch(images))
            clean_loss = criterion(clean_outputs, labels)

            # Generate adversarial batch
            adv_input_images = utils.generate_adversarial_batch(
                model=model,
                images=images,
                labels=labels,
                attack="PGD",
                device=device,
                epsilon=8/255.0,
                alpha=2/255.0,
                num_steps=10,
                criterion=criterion
            )

            # Forward on adversarial examples
            adv_input_outputs = model(utils.normalize_batch(adv_input_images))
            adv_input_loss = criterion(adv_input_outputs, labels)

            # 1. Generate latent adversarial representation
            z_adv = utils.pgd_attack_latent_l3(
                model=model,
                image=images,
                label=labels,
                epsilon=epsilon,#not adding same pixel-level perturbation, but only latent-level perturbation
                alpha=(epsilon / 4.0),
                num_steps=5,
                criterion=criterion,
                device=device
            )

            # 2. Forward pass using latent adversarial input
            adv_latent_outputs = model.decode(z_adv)
            adv_latent_loss = criterion(adv_latent_outputs, labels)

            # 3. Compute loss and update weights
            loss = (lambda_clean * clean_loss) + (lambda_input_adv * adv_input_loss) + (lambda_latent_adv * adv_latent_loss)
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()

            # 4. Accuracy (input-adv training accuracy)
            _, predicted = torch.max(adv_input_outputs.data, 1)
            training_correct += (predicted == labels).sum().item()

        print(f'Input-PGD Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, '
          f'Accuracy: {100 * training_correct / len(train_dataset):.2f}%')

    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'resnet18_gtsrb_input_latent_adv.pth'))
    print("Saved input-latent-PGD robust model.")

if __name__ == "__main__":    
    main()