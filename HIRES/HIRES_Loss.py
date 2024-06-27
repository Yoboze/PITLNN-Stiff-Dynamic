import os
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the Physics-Informed Transfer Learning Neural Network (PITLNN) class
class PITLNN(nn.Module):
    def __init__(self, layers, r1_layers, r2_layers, r3_layers, r5_layers, r6_layers, r10_layers):
        super(PITLNN, self).__init__()
        self.net = self.initialize_NN(layers)
        self.r1_net = self.initialize_NN(r1_layers)
        self.r2_net = self.initialize_NN(r2_layers)
        self.r3_net = self.initialize_NN(r3_layers)
        self.r5_net = self.initialize_NN(r5_layers)
        self.r6_net = self.initialize_NN(r6_layers)
        self.r10_net = self.initialize_NN(r10_layers)
        
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
        r1 = self.r1_net(t)
        r2 = self.r2_net(t)
        r3 = self.r3_net(t)
        r5 = self.r5_net(t)
        r6 = self.r6_net(t)
        r10 = self.r10_net(t)
        y1 = y[:, 0].view(-1, 1)
        y2 = y[:, 1].view(-1, 1)
        y3 = y[:, 2].view(-1, 1)
        y4 = y[:, 3].view(-1, 1)
        y5 = y[:, 4].view(-1, 1)
        y6 = y[:, 5].view(-1, 1)
        y7 = y[:, 6].view(-1, 1)
        y8 = y[:, 7].view(-1, 1)
        return y1, y2, y3, y4, y5, y6, y7, y8, r1, r2, r3, r5, r6, r10
    


# Define network layers
layers = [1, 64, 64, 64, 64, 64, 64, 64, 8]  
r1_layers = [1, 64, 64, 64, 64, 64, 64, 64, 1]
r2_layers = [1, 64, 64, 64, 64, 64, 64, 64, 1]
r3_layers = [1, 64, 64, 64, 64, 64, 64, 64, 1]
r5_layers = [1, 64, 64, 64, 64, 64, 64, 64, 1]
r6_layers = [1, 64, 64, 64, 64, 64, 64, 64, 1]
r10_layers = [1, 64, 64, 64, 64, 64, 64, 64, 1]


# Initialize the main model
main_model = PITLNN(layers, r1_layers, r2_layers, r3_layers, r5_layers, r6_layers, r10_layers)
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
