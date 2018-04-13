# Robust Physical Adversarial Attack on Faster R-CNN Object Detector

## Overview

This is the code repository for the under-review ECML-PKDD 2018 paper: **Robust Physical Adversarial Attack on Faster R-CNN Object Detector**

The code included here reproduces our techniques presented in the paper. 

In this work, we tackle the more challenging problem of crafting physical adversarial perturbations to fool image-based object detectors like Faster R-CNN.
Attacking an object detector is more difficult than attacking an image classifier, as it needs to mislead the classification results in multiple bounding boxes with different scales.
Our approach can generate perturbed stop signs that are consistently mis-detected by Faster R-CNN as other objects, posing a potential threat to autonomous vehicles and other safety-critical computer vision systems.



## Videos of Targeted and Untargted Attacks

High-confidence Person Perturbation:
https://youtu.be/Zv__RSYIu8c

High-confidence Sports Ball Perturbation:
https://youtu.be/Idoc1vrN2T0

High-confidence Untargeted Attack:
https://youtu.be/8rwoJ93jhY4



## Researchers

|  Name                 | Affiliation                     |
|-----------------------|---------------------------------|
| Shang-Tse Chen        | Georgia Institute of Technology |
| Cory Cornelius        | Intel Corporation               |
| Jason Martin          | Intel Corporation               |
| Polo Chau             | Georgia Institute of Technology |
