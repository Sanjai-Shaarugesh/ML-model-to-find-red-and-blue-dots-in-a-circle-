import sklearn 
from sklearn.datasets import make_circles
from torch import nn
import torch as t
import pandas as pd

# Data generation
n_samples = 1000
x, y = make_circles(n_samples=n_samples, noise=0.05, random_state=42)
x, y = t.from_numpy(x).type(t.float), t.from_numpy(y).type(t.float)

# Device configuration
device = "cuda" if t.cuda.is_available() else 'cpu'

# Split data
from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=42)

# Loss function
ls_fun = nn.BCEWithLogitsLoss()

# Accuracy function
def acc_fun(y_true, y_pred):
    corr = t.eq(y_true, y_pred).sum().item()
    acc_c = (corr / len(y_true)) * 100
    return acc_c

# Improved Model with Dropout and BatchNorm
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features=2, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=64, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=64, out_features=1),  # Binary classification output
        )
       
    def forward(self, x: t.Tensor):
        return self.seq(x)

# Instantiate the model
model_v = Model().to(device)

# Optimizer (use Adam instead of SGD for better performance)
optim = t.optim.Adam(model_v.parameters(), lr=0.01)

# Learning rate scheduler (optional for learning rate decay)
scheduler = t.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.1)

# Set manual seed for reproducibility
t.manual_seed(42)

# Move data to device
x_test, x_train = x_test.to(device), x_train.to(device)
y_test, y_train = y_test.to(device), y_train.to(device)

# Train the model
epochs = 1000
for epoch in range(epochs):
    # Training mode
    model_v.train()
    
    # Forward pass
    y_logits = model_v(x_train).squeeze()
    y_preds = t.round(t.sigmoid(y_logits))  # Sigmoid for binary classification
    
    # Compute loss and accuracy
    loss = ls_fun(y_logits, y_train)  # Compare logits with true labels
    acc = acc_fun(y_train, y_preds)
    
    # Optimizer
    optim.zero_grad()
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optim.step()
    
    # Learning rate scheduling
    scheduler.step()
    
    # Evaluation mode
    model_v.eval()
    
    with t.no_grad():  # Disable gradient calculation for evaluation
        test_logits = model_v(x_test).squeeze()
        test_preds = t.round(t.sigmoid(test_logits))  # Test predictions
        
        # Compute test loss and accuracy
        test_loss = ls_fun(test_logits, y_test)  # Compare logits with true labels
        test_acc = acc_fun(y_test, test_preds)
        
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Train Acc: {acc:.2f}% | Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}%")
        
    # Early stopping condition (if you get 99% accuracy early)
    if test_acc >= 99.0:
        print(f"Stopping early at epoch {epoch} with test accuracy {test_acc:.2f}%")
        break

# Visualize decision boundaries (you can plot this after training)
import requests
from pathlib import Path 

# Helper function download logic
if not Path('helper_function.py').is_file():
    print("Downloading helper_function.py file")
    request = requests.get('https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py')
    with open('helper_function.py', 'wb') as f:
        f.write(request.content)

from helper_function import plot_decision_boundary
import matplotlib.pyplot as plt 

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_v, x_train, y_train)

plt.subplot(1, 2, 2)
plt.title('Test')
plot_decision_boundary(model_v, x_test, y_test)
plt.show()
