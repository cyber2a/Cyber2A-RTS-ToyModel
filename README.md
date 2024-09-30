# Cyber2A Workshop: Toy Model for RTS Training and Inference

This repository demonstrates a simplified example of training and running inference on a toy model using the Retrogressive Thaw Slumps (RTS) dataset. It is adapted from the official [PyTorch Vision Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html).

The copyright for the tutorial content belongs to PyTorch. © Copyright 2024, PyTorch.

## Requirements

To set up the environment, I recommend using Conda. You can create the environment using the provided `requirements_conda.txt` file:

```bash
conda create --name <env_name> --file requirements_conda.txt
```

A `requirements.txt` file is also provided for pip installation, though it has not been fully tested:

## Usage

### 1. Training the Model
To train the model, run the following command in your terminal:

```bash
python train.py
```

This command trains a Mask R-CNN model with a ResNet-50 backbone using the RTS dataset.

### 2. Running Inference
To perform inference using the trained model, run:

```bash
python inference.py --image-path <path_to_image>
```

Replace `<path_to_image>` with the path to your image file.

### Pre-trained Model
If training cannot be completed due to time constraints, a pre-trained model is available. You can download the model weights from [this link](https://www.dropbox.com/scl/fi/rgnsan6ud1uwaw97q7o1h/rts_model.pth?rlkey=a9rk8jeac9hvx36xo7phl0hnb&st=t5mqmfu6&dl=0) and place the `rts_model.pth` file in the root directory of the repository.

## Notes
- This model is for demonstration purposes and has not been optimized for performance.

