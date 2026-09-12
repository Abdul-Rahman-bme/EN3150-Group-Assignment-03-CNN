# Data Directory

The project uses the **Fashion-MNIST** dataset.

Dataset files are not manually committed to this repository. The dataset will be downloaded programmatically so that the experiments can be reproduced on another machine.

## Structure

- `raw/` - Original downloaded dataset files
- `processed/` - Any generated or preprocessed dataset artifacts

## Dataset Split

A fixed split will be used throughout the project:

- 70% training
- 15% validation
- 15% testing

All models will use the same split to ensure a fair comparison.
