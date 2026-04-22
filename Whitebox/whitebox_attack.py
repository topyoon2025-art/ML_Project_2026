from py_compile import main

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
import utils
import argparse
import gc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, default="FGSM", choices=["FGSM", "PGD"])
    parser.add_argument("--latent_adv", action="store_true", help="Whether to use the latent adversarially trained model")
    parser.add_argument("--input_latent_adv", action="store_true", help="Whether to use the model trained on both input and latent adversarial examples")
    parser.add_argument("--input_adv", action="store_true", help="Whether to use the model trained on input adversarial examples")
    args = parser.parse_args()

    attack = args.attack

    # function for reading the images
    # arguments: path to the traffic sign data, for example './GTSRB/Training'
    # returns: list of images, list of corresponding labels 
    print("Reading testing data")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, 'GTSRB', 'Test', 'Final_Test', 'Images')
    testImages, testLabels = utils.readTrafficSigns_test(path) 

    ###############################################
    #Preprocess the images and labels for training#
    test_dataset = utils.GTSRBDataset(testImages, testLabels, target_size=224)
    print(f"Total testing samples: {len(test_dataset)}")

    # Model setup for testing and attack generation
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 43) # 43 classes in GTSRB
    
    # model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'resnet18_gtsrb_latent_adv.pth')))  # Load the trained model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = 43

    # Create wrapper model
    model = utils.ResNet18_L3(num_classes=num_classes)

    # Load checkpoint directly into wrapper
    if args.latent_adv:
        # trained on latent adversarial examples
        state = torch.load(os.path.join(BASE_DIR, 'resnet18_gtsrb_latent_adv.pth'), map_location=device)
        # # trained on both input and latent adversarial examples
    elif args.input_latent_adv:
        state = torch.load(os.path.join(BASE_DIR, 'resnet18_gtsrb_input_latent_adv.pth'), map_location=device)
    elif args.input_adv:
        state = torch.load(os.path.join(BASE_DIR, 'resnet18_gtsrb_input_adv.pth'), map_location=device)
    else:
        state = torch.load(os.path.join(BASE_DIR, 'resnet18_gtsrb_clean.pth'), map_location=device)

    model.load_state_dict(state)

    model = model.to(device)
    model.eval()
    

    criterion = nn.CrossEntropyLoss()
    batch_size_test = 64
    num_workers = 0

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    # Clean accuracy (baseline)
    clean_accuracy, predictions_clean = utils.evaluate_clean_accuracy(
        model,
        test_loader,
        device
    )
    print(f"PGD trained model: {args.latent_adv} | Attack: {attack}")
    print(f"Clean accuracy: {clean_accuracy * 100:.2f}%")

    eps_list_255 = [2, 4, 8, 16]
    eps_list = [e / 255.0 for e in eps_list_255]
    num_steps = 10
    for eps in eps_list:
        epsilon = eps
        alpha = (epsilon / 4.0)  # standard PGD step size
        print(f"\nGenerating adversarial examples with {attack} attack (epsilon={epsilon:.4f})")
        # Generate adversarial examples
        examples_adv, true_labels = utils.generate_adversarial_examples_batched(
            model=model,
            test_loader=test_loader,
            attack=attack,
            device=device,
            epsilon=epsilon,
            criterion=criterion,
            alpha=alpha,
            num_steps=num_steps
        )

        # Adversarial accuracy
        adv_dataset = TensorDataset(examples_adv, true_labels)
        adv_loader = DataLoader(
            adv_dataset,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )
        adv_accuracy, predictions_adv = utils.evaluate_adversarial_accuracy(
            model,
            adv_loader,
            device
        )
        print(f"Adversarial accuracy: {adv_accuracy * 100:.2f}%")

        # Attack Success Rate
        asr, successful, initially_correct = utils.compute_attack_success_rate(
            predictions_clean,
            predictions_adv,
            true_labels
        )
        
        print(f"Attack Success Rate: {asr * 100:.2f}%")
        print(f"Successful attacks: {successful}")
        print(f"Initially correct samples: {initially_correct}")
        
        del adv_loader, adv_dataset, examples_adv, true_labels
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":    
    main()