# Data Directory

This project uses the **EuroSAT RGB** dataset.

The dataset contains 27,000 RGB satellite images distributed across 10 land-use and land-cover classes. Each image has a resolution of 64 × 64 pixels.

Dataset files are not committed directly to this repository. The dataset will be downloaded programmatically so that the experiments can be reproduced on another machine.

## Structure

- `raw/` - Original EuroSAT dataset
- `processed/` - Generated split information or preprocessing artifacts

## Dataset Split

All experiments will use the same fixed split:

- 70% training
- 15% validation
- 15% testing

A stratified split will be used so that class proportions are maintained across the three subsets.