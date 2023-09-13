# BaseModel for Image Segmentation

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Creating a Custom Model](#creating-a-custom-model)
  - [Batch Prediction](#batch-prediction)
  - [Single Image Prediction](#single-image-prediction)
- [Example](#example)

## Introduction

The `BaseModel` class serves as a foundational superclass for building image segmentation models using PyTorch. This class provides utility functions for making predictions on both single and multiple images.

## Installation

To use this class, simply download the `base_model.py` file and import it into your Python script where you define your custom model.

```python
from base_model import BaseModel
```

## Requirements

- Python 3.6+
- PyTorch 1.7+
- NumPy
- tqdm

## Usage

### Creating a Custom Model

To use `BaseModel` as a superclass, your custom model should inherit from it and call its `__init__` method in your model's constructor, along with the constructors for other classes inherited by the model. Here is an example:

```python
import torch.nn as nn
from base_model import BaseModel

class CustomSegmentationModel(nn.Module, BaseModel):
    def __init__(self, device):
        nn.Module.__init__(self)
        BaseModel.__init__(self, device)
        # Your model layers and operations here

    def forward(self, x):
        # Your forward pass logic here
```

Don't forget to implement the `forward` method, as it is required for PyTorch models.

### Batch Prediction

Once your custom model is set up, you can use the `predict_all` method to make batch predictions.

```python
predictions = custom_model.predict_all(dataloader, disable_progress=False)
```

It is absolutely important that a dataloader that yields a tuple of (image, mask) is passed to the `predict_all` method. The `predict_all` method returns a list of predictions, where each prediction is a PyTorch tensor of shape (num_images, channels, height, width).

The argument `disable_progress` can be set to True if you wish to disable the progress bar.

The output tensor contains the predicted masks for each image in the batch. The activation function applied depends on the number of classes predicted by the model. If the model predicts 1 class, then a `sigmoid` activation is applied. If the model predicts more than 1 class, then a `softmax` activation is applied. Resulting in a `ndarray` of shape `(num_images, num_classes, height, width)`.

### Single Image Prediction

For predicting a single image, you can use the `predict_image` method inherited from `BaseModel`.

```python
prediction = custom_model.predict_image(image)
```

The `image` argument should be a PyTorch tensor of shape (channels, height, width).

The output tensor contains the predicted mask for the image. The activation function applied depends on the number of classes predicted by the model. If the model predicts 1 class, then a `sigmoid` activation is applied. If the model predicts more than 1 class, then a `softmax` activation is applied. Resulting in a `ndarray` of shape `(num_classes, height, width)`.

## Example




