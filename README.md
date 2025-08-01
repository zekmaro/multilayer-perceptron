# multilayer-perceptron - Deep Learning from Scratch 🧠

## 🧪 Project Overview

`multilayer-perceptron` is a deep learning project from the 42 Mastery curriculum where you implement a neural network from scratch using **Python** and **NumPy**, without any deep learning frameworks like TensorFlow or PyTorch. The main objective is to gain a strong understanding of how forward propagation, backpropagation, gradient descent, and activation functions work at a low level.

This project includes building, training, and evaluating a fully-connected feedforward neural network (multilayer perceptron, or MLP) on a classification dataset.

## 🚀 Features

* **Custom MLP Architecture** - Define any number of layers and neurons
* **Forward Propagation** - Computes activations layer by layer
* **Backpropagation** - Calculates gradients using chain rule
* **Mini-Batch Gradient Descent** - Updates weights using batches of data
* **Weight Initialization** - Supports different strategies (random, He, Xavier)
* **Activation Functions** - ReLU, Sigmoid, Tanh, Softmax
* **Loss Functions** - Cross-entropy and MSE
* **Training Loop** - Tracks loss and accuracy over epochs
* **Model Evaluation** - Accuracy, confusion matrix, and prediction visualization

## 🧠 Concepts Covered

* Multilayer perceptrons (MLPs)
* Feedforward networks
* Activation functions
* Backpropagation and gradient computation
* Learning rate tuning
* Batch normalization (optional bonus)
* One-hot encoding for multi-class classification

## 📦 Requirements

* Python 3.x
* NumPy
* Matplotlib (for plotting)
* Pandas (for data handling)

Install dependencies:

```sh
pip install -r requirements.txt
```

## 🛠️ Usage

### 1. Training the Network

```sh
python train.py --epochs 100 --lr 0.01 --batch_size 32
```

Optional arguments:

* `--hidden_layers 64 64` – custom hidden layer sizes
* `--activation relu` – activation function (relu, tanh, sigmoid)
* `--dataset data.csv` – specify a CSV file for input

### 2. Predicting

```sh
python predict.py input.csv
```

### 3. Visualizing Training

```sh
python plot_loss.py
```

* Loss curve
* Accuracy per epoch

## 📁 Project Structure

```
📂 multilayer-perceptron/
├── train.py             # Training loop
├── predict.py           # Inference script
├── model.py             # MLP architecture & training logic
├── utils.py             # Helper functions
├── data/
│   ├── train.csv
│   └── test.csv
├── weights.npy          # Saved model weights
├── plot_loss.py         # Visualization of training
└── requirements.txt
```

## 📊 Example Output

```
Epoch 10/100: loss=0.354, accuracy=89.1%
Epoch 100/100: loss=0.082, accuracy=96.5%
Test accuracy: 95.8%
```

## 🧪 Dataset

You can use any dataset suitable for classification (e.g., digit classification, IRIS, MNIST subset). The format should be CSV with features and labels, where labels are one-hot encoded or categorical.

## 🏗️ Potential Enhancements

* Add batch normalization
* Dropout regularization
* L2 regularization
* Dynamic learning rate adjustment
* Command-line interface with better logging

## 🏆 Credits

* **Developer:** [zekmaro](https://github.com/zekmaro)
* **Project:** Part of the 42 Mastery Deep Learning track
* **Inspired by:** Yann LeCun’s foundational MLP work, Andrew Ng’s ML lectures

---

🧠 Build it. Train it. Understand it. Dive deep into neural networks from the ground up!
