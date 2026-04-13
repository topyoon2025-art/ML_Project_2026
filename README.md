# Update 13 Apr 26
 - Download GTSRB dataset to Whitebox folder via https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/jyoon72_jh_edu/IQBR6yDDbfQ0TrjVqHQyZNVPAa7FdF2yXnSah6D08IaXQXY?e=hO4hcw
 - Dataset: GTSRB
   - 43 classes  
   - Resized to (C, H, W) -> (3, 224, 224) for Resnet-18
   - Letterbox resizing while preserving aspect ratio and maintain the same H and W across all images with padding
   - Normalized by dividing with 255
   - Training set: 39209
   - Test set: 12630
 - Trained the classifier via whitebox_train.py:
   - Resnet-18:
     - Modified last layer to fit to 43 classes
     - Using pretained model with imagenet (comes with mean and std that we need to use for our image normalization)
     - Trained the model with epochs = 5 (each epoch took about 30-40 minuets)
     - Train batch size: 32
     - Criterion: Cross Entropy Loss
     - Optimizer: Adam with lr = 0.001
     - Trained model saved as resnet18_gtsrb.pth in Whitebox folder
  - Implemented FGSM and PGD via whtiebox_attack.py in Whitebox folder
    - Epsilon = 0.03
    - alpha = epsilon / 4, used for BGD
    - Criterion: Cross Entropy Loss
    - Can choose between FGSM and BGD
    - Produces: (preliminary results)
      - Clean accuracy: ~98%
      - Adversaril accuracy:
        - FGSM: ~25%
        - PGD: ~50%
      - Attack success rate:
        - FGSM: ~25%
        - PGD: ~50%
        







# ML_Project_2026 Proposal
Github Page for ML 2026 Spring Project

- Introduction v0: There are numerous image classification models developed as a result of recent advances in machine learning algorithms, especially over the past decade. Increasing the safety and robustness of these models remains an ongoing challenge for building confidence in their use. To support this effort, we introduce the creation of perturbed images to test the accuracy, reliability, and robustness of existing models. There are many methods for generating adversarial image attacks. For the scope of this project, we will first investigate the FGSM method to establish a baseline of current practices. We will then expand to modified versions of FGSM to increase the effectiveness of exposing weaknesses in existing machine learning models. If successful, we will explore combining different types of adversarial attack images to address a variety of situations and conditions. This is anlagous to the pen testing in cyberspace in an effort to make our system more secure and safe.

- Introduction v1: Over the past decade, rapid advances in machine learning have led to the development of numerous high‑performing image classification models. Ensuring the safety, reliability, and robustness of these models remains a central challenge, especially as they are deployed in increasingly sensitive real‑world applications. To support this effort, we explore the use of perturbed (adversarial) images as a way to evaluate how well existing models withstand intentional, carefully crafted disruptions. There are many established methods for generating adversarial image attacks. In this project, we begin with the Fast Gradient Sign Method (FGSM) to establish a baseline and understand current practices. From there, we extend our investigation to modified versions of FGSM designed to increase the effectiveness of the perturbations and more clearly expose model weaknesses. If these approaches prove successful, we plan to explore combinations of different adversarial attack techniques to simulate a wider range of challenging conditions (i.e blackbox, whitebox) and further stress‑test model robustness. This approach is analogous to penetration testing in cybersecurity, where controlled attacks are used to strengthen the overall safety and security of a system.

- Dataset and Features
  * Image Dataset
    * Which Dataset 
  * Features
    * Low, Mid, High, Global
       
- Method
  * Image Classifier: CNN (VIT, RNN options) based classifier ->Use this for the baseline and experiment
  * Attack Method:
    * FGSM
    * Modified FGSM
    * PGD
    * etc
    * Combined method
   
- Deliverables
  * Must achieve: Modified FGSM more effective than regular FGSM in exposing the weaknesses of the existing models
  * Expect to achieve: Dynamic integration of varying attack methods to fit to the given conditions such as blackbox or whitebox
  * Would like to: GAN doesn't detect (or good escape rate) the purturbed images as adversarial attack images.  

