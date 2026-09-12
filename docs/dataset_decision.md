# Dataset Selection

## Selected Dataset

**Fashion-MNIST**

Fashion-MNIST was selected as the image-classification dataset for the EN3150 Assignment 03.

The dataset contains grayscale images of clothing items belonging to 10 classes. Each image has a resolution of 28 × 28 pixels, which is already below the maximum 64 × 64 resolution required by the assignment.

## Why Fashion-MNIST Was Selected

Fashion-MNIST was chosen because:

- it is designed for image-classification experiments;
- its 28 × 28 resolution is suitable for resource-constrained CNNs;
- it contains multiple visually similar classes, making the classification task more meaningful than a very simple digit dataset;
- it contains enough samples to create reliable training, validation, and test subsets;
- it is small enough to allow repeated CNN and optimizer experiments within reasonable computational time;
- it provides a suitable benchmark for comparing a standard CNN, a sub-100k parameter lightweight CNN, and lightweight transfer-learning models.

## Assignment Data Split

The dataset will be divided according to the assignment specification:

- Training: 70%
- Validation: 15%
- Testing: 15%

A fixed random seed will be used so that all four team members use exactly the same split throughout the project.

## Image Resolution

The original Fashion-MNIST images are:

- Width: 28 pixels
- Height: 28 pixels
- Channels: 1 grayscale channel

The custom CNN experiments will retain the native 28 × 28 resolution unless further resizing is justified.

For pre-trained SOTA networks, the images may need to be resized and converted to three channels according to the input requirements of the selected architecture. These transformations will be applied without changing the train, validation, and test membership.

## Classes

Fashion-MNIST contains 10 classes:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot