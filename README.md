# Waste Classification

**Basil Muhamed Ashraf**  
20211226  

---

## Problem

The project aims to automate the classification of recyclable and household waste by classifying images into distinct categories such as plastic, paper, glass, metal, organic waste, and textiles.

**Input:** A 256×256 color image in PNG format representing a waste item.

**Output:** A categorical label from a set of 30 predefined classes indicating the type of waste item (e.g., `plastic_water_bottles`, `cardboard_boxes`, `coffee_grounds`).

## Dataset

[*Recyclable and Household Waste Classification (Kaggle) by Alistair King*](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification/data)

It comprises 15,000 images organized into 30 classes with 500 images per class.  
Each class includes two categories: *default* (studio-like images) and *real_world* (images in practical scenarios).

## Model Architecture

The architecture is inspired by classic CNN designs. It consists of:

- **Convolutional Blocks:** Three blocks, each with a `Conv2d` layer (filters increasing from 32 to 128) using a (3×3) kernel, followed by `ReLU` and `MaxPool2d` layers. These blocks progressively extract low- to high-level features from the images.
- **Dense Layers:** A flattening layer to convert features into a vector, followed by a `Linear` layer with 512 neurons and `ReLU` activation. A `Dropout` layer (rate of 0.5) is applied to reduce overfitting.
- **Output Layer:** A final `Linear` layer with 30 neurons using `Softmax` activation to output class probabilities. The loss used is `CrossEntropy` (with ADAM optimizer).
