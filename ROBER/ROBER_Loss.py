import os
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the Physics-Informed Transfer Learning Neural Network (PITLNN) class
class PITLNN(nn.Module):
    def __init__(self, layers, k1_layers, k2_layers, k3_layers):
        super(PITLNN, self).__init__()
        self.net = self.initialize_NN(layers)
        self.k1_net = self.initialize_NN(k1_layers)
        self.k2_net = self.initialize_NN(k2_layers)
        self.k3_net = self.initialize_NN(k3_layers)
        

    def initialize_NN(self, layers):
        modules = []
        for i in range(len(layers)-2):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(nn.GELU())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        nn.init.xavier_uniform_(modules[-1].weight)
        nn.init.zeros_(modules[-1].bias)
        return nn.Sequential(*modules)

    def forward(self, t):
        y = self.net(t)
        k1 = self.k1_net(t)
        k2 = self.k2_net(t)
        k3 = self.k3_net(t)
        y1 = y[:, 0].view(-1, 1)
        y2 = y[:, 1].view(-1, 1)
        y3 = y[:, 2].view(-1, 1)
        return y1, y2, y3, k1, k2, k3


# Initialize your model
layers = [1, 32, 32, 32, 32, 32, 3]  
k1_layers = [1, 32, 32, 32, 32, 32, 1]
k2_layers = [1, 32, 32, 32, 32, 32, 1]
k3_layers = [1, 32, 32, 32, 32, 32, 1]

model = PITLNN(layers, k1_layers, k2_layers, k3_layers)
#model.eval()  # Set the model to evaluation mode

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
