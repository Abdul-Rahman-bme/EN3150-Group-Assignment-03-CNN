# Dataset Selection

## Selected Dataset

**EuroSAT RGB**

EuroSAT was selected as the image-classification dataset for EN3150 Assignment 03.

The dataset contains 27,000 RGB satellite images belonging to 10 land-use and land-cover classes. Each image has a resolution of 64 × 64 pixels, which directly satisfies the maximum image resolution specified in the assignment.

## Why EuroSAT Was Selected

EuroSAT was selected because:

- the images are already 64 × 64 pixels, matching the resource-constrained setting required by the assignment;
- the dataset contains RGB images, making it suitable for both custom CNNs and pretrained lightweight architectures;
- it contains 27,000 images, providing enough data for reliable training, validation, and test subsets;
- it contains 10 distinct land-use and land-cover classes;
- the dataset presents a more realistic image-classification problem than very simple grayscale benchmark datasets;
- the RGB format makes comparison with lightweight pretrained architectures such as MobileNet and EfficientNet more natural;
- the application is relevant to edge-computing scenarios such as satellite, drone, and remote sensing systems.

## Dataset Classes

EuroSAT RGB contains 10 classes:

1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

## Data Split

The dataset will be divided according to the assignment specification:

- Training: 70%
- Validation: 15%
- Testing: 15%

A fixed random seed will be used throughout the project.

Because the number of images is not exactly equal across all classes, stratified splitting will be used to preserve the class distribution as closely as possible in each subset.

## Image Format

Each image has:

- Width: 64 pixels
- Height: 64 pixels
- Channels: 3 RGB channels

The native resolution will therefore be retained for the custom CNN models.

For pretrained lightweight models, the same 64 × 64 dataset split will be used. Model-specific normalization required by pretrained weights will be applied where necessary.

## Edge-Computing Relevance

EuroSAT represents a realistic edge image-classification scenario. A lightweight classifier could potentially be deployed on a drone, satellite subsystem, Raspberry Pi, or remote sensing node to perform local land-use classification while reducing communication and computational requirements.