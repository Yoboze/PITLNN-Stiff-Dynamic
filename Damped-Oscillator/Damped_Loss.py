import os
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the Physics-Informed Transfer Learning Neural Network (PITLNN) class
class PITLNN(nn.Module):
    def __init__(self, layers, c_layers, k_layers):
        super(PITLNN, self).__init__()
        self.net = self.initialize_NN(layers)
        self.c_net = self.initialize_NN(c_layers)
        self.k_net = self.initialize_NN(k_layers)

    def initialize_NN(self, layers):
        modules = []
        for i in range(len(layers) - 2):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(nn.GELU())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        nn.init.xavier_uniform_(modules[-1].weight)
        nn.init.zeros_(modules[-1].bias)
        return nn.Sequential(*modules)

    def forward(self, t):
        z = self.net(t)
        c = self.c_net(t)
        k = self.k_net(t)
        z1 = z[:, 0].view(-1, 1)
        z2 = z[:, 1].view(-1, 1)
        return z1, z2, c, k

# Initialize your model
layers = [1, 64, 64, 64, 64, 64, 2]
c_layers = [1, 64, 64, 64, 64, 64, 1]
k_layers = [1, 64, 64, 64, 64, 64, 1]

model = PITLNN(layers, c_layers, k_layers)
model.eval()  # Set the model to evaluation mode

# Define directories
dir_path = os.getcwd()
figs_dir = os.path.join(dir_path, "figs3")
checkpoint_dir = os.path.join(dir_path)
os.makedirs(figs_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Load metrics from the CSV file
metrics_file = os.path.join(checkpoint_dir, "training_metrics.csv")
epochs, total_losses, data_losses, physics_losses, regularization_losses, gradient_norms = [], [], [], [], [], []
with open(metrics_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        epochs.append(int(row['Epoch']))
        total_losses.append(float(row['Total Loss']))
        data_losses.append(float(row['Data Loss']))
        physics_losses.append(float(row['Physics Loss']))
        regularization_losses.append(float(row['Regularization Loss']))
        # Clean the gradient norm value more thoroughly
        grad_norm_raw = row['Gradient Norm']
        grad_norm_clean = grad_norm_raw.replace('tensor(', '').replace(')', '').strip()
        gradient_norm = float(grad_norm_clean)
        gradient_norms.append(gradient_norm)

# Plot total losses, data losses, physics losses, and regularization losses
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, total_losses, label='Total Loss', alpha=0.8)
plt.plot(epochs, data_losses, label='Data Loss', alpha=0.8)
plt.plot(epochs, physics_losses, label='Physics Loss')
plt.plot(epochs, regularization_losses, label='Regularization Loss', alpha=0.8)
plt.title('Loss Components Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot gradient norms
plt.subplot(1, 2, 2)
plt.plot(epochs, gradient_norms, '-o', color='red', label='Gradient Norm')
plt.title('Gradient Norm over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(figs_dir, 'Loss_Gradient_Overview.png'))
plt.show()
