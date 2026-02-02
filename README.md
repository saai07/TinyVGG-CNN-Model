# TinyVGG-CNN-Model

A PyTorch implementation of the TinyVGG convolutional neural network architecture for image classification on custom datasets. This project replicates the TinyVGG model from the [CNN Explainer](https://poloclub.github.io/cnn-explainer/) visualization tool.

## 📋 Overview

This repository contains a complete implementation of the TinyVGG architecture, designed for learning and understanding CNN-based image classification. The model is trained on a custom food classification dataset (pizza, steak, and sushi images), which is a subset of the Food101 dataset.

## 🎯 Features

- **TinyVGG Architecture**: Lightweight CNN with two convolutional blocks
- **Custom Dataset Implementation**: Custom PyTorch Dataset class that replicates `ImageFolder` functionality
- **Data Augmentation**: Image transformations including resizing, random horizontal flipping, and normalization
- **Training & Evaluation**: Complete training loop with loss and accuracy tracking
- **Visualization**: Functions to plot transformed images and prediction results
- **Model Summary**: Integration with `torchinfo` for architecture visualization

## 🏗️ Architecture

The TinyVGG model consists of:

```
Input (3, 64, 64)
    ↓
Conv Block 1:
  - Conv2d(3 → hidden_units, kernel=3)
  - ReLU
  - Conv2d(hidden_units → hidden_units, kernel=3)
  - ReLU
  - MaxPool2d(kernel=2, stride=2)
    ↓
Conv Block 2:
  - Conv2d(hidden_units → hidden_units, kernel=3)
  - ReLU
  - Conv2d(hidden_units → hidden_units, kernel=3)
  - ReLU
  - MaxPool2d(kernel=2, stride=2)
    ↓
Classifier:
  - Flatten
  - Linear(hidden_units × 13 × 13 → num_classes)
    ↓
Output (num_classes)
```

## 📦 Requirements

```
Python 3.x
PyTorch
torchvision
matplotlib
requests
Pillow
torchinfo (optional, for model summary)
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/saai07/TinyVGG-CNN-Model.git
cd TinyVGG-CNN-Model
```

2. Install dependencies:
```bash
pip install torch torchvision matplotlib pillow torchinfo
```

## 📊 Dataset

The project uses a subset of the Food101 dataset containing three classes:
- **Pizza**
- **Steak**
- **Sushi**

Each class contains:
- 75 training images (10% of original Food101)
- 25 testing images

The dataset is automatically downloaded when running the script.

## 💻 Usage

### Running the Jupyter Notebook

```bash
jupyter notebook TinyVGG.ipynb
```

Or use the Google Colab version:
```bash
jupyter notebook TinyVGG_WITH_custom_dataset.ipynb
```

### Running the Python Script

```bash
python tinyvgg_with_custom_dataset.py
```

### Training Your Own Model

```python
import torch
from torch import nn

# Initialize model
model = TinyVGG(
    input_shape=3,          # RGB channels
    hidden_unit=10,         # Number of hidden units
    output_shape=3          # Number of classes
).to(device)

# Setup loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
results = train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=5
)
```

### Making Predictions

```python
# Predict on a custom image
pred_and_plot_image(
    model=model,
    image_path="path/to/your/image.jpg",
    class_names=["pizza", "steak", "sushi"],
    transform=test_transforms,
    device=device
)
```

## 📁 Project Structure

```
TinyVGG-CNN-Model/
├── TinyVGG.ipynb                          # Main Jupyter notebook
├── TinyVGG_WITH_custom_dataset.ipynb      # Custom dataset implementation
├── tinyvgg_with_custom_dataset.py         # Python script version
├── README.md                               # Project documentation
└── data/                                   # Dataset directory (auto-created)
```

## 🔑 Key Components

### Custom Dataset Class

The `ImageFolderCustom` class replicates PyTorch's `ImageFolder` with custom implementation:
- Automatic class discovery from directory structure
- Custom image loading
- Transform support
- Compatible with PyTorch DataLoader

### Data Transformations

```python
train_transforms = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])
```

### Training Functions

- `train_step()`: Single training epoch
- `test_step()`: Model evaluation
- `train()`: Complete training loop with metrics tracking

## 📈 Model Performance

The model is trained with:
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Input Size**: 64×64 pixels

## 🎓 Learning Resources

This project is inspired by:
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) - Interactive CNN visualization
- PyTorch Deep Learning course materials

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

**saai07**
- GitHub: [@saai07](https://github.com/saai07)

## 🙏 Acknowledgments

- CNN Explainer team for the original TinyVGG architecture visualization
- PyTorch team for the deep learning framework
- Food101 dataset creators

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

⭐ If you find this project helpful, please consider giving it a star!
