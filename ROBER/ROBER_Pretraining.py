import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
from tqdm import tqdm

# Define the ODE system
def ode_system(t, y, k1, k2, k3):
    y1, y2, y3 = y
    dy1dt = -k1 * y1 + k3 * y2 * y3
    dy2dt = k1 * y1 - k2 * y2**2 - k3 * y2 * y3
    dy3dt = k2 * y2**2
    return [dy1dt, dy2dt, dy3dt]

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

    
def net_l(self, t):
    t.requires_grad = True
    y1, y2, y3, k1, k2, k3 = self.forward(t)
    y1_t = torch.autograd.grad(y1, t, grad_outputs=torch.ones_like(y1), create_graph=True)[0]
    y2_t = torch.autograd.grad(y2, t, grad_outputs=torch.ones_like(y2), create_graph=True)[0]
    y3_t = torch.autograd.grad(y3, t, grad_outputs=torch.ones_like(y3), create_graph=True)[0]
    l1 = y1_t - (-k1 * y1 + k3 * y2 * y3)
    l2 = y2_t - (k1 * y1 - k2 * y2**2 - k3 * y2 * y3)
    l3 = y3_t - k2 * y2**2
    return l1, l2, l3



def custom_loss(model, t_data, y1_data, y2_data, y3_data, phy_coeff = 1e-7):
    y1_pred, y2_pred, y3_pred, k1, k2, k3 = model(t_data)
    l1, l2, l3 = net_l(model, t_data)
    data_loss = torch.mean((y1_pred - y1_data)**2 + (y2_pred - y2_data)**2 + (y3_pred - y3_data)**2)
    physics_loss = phy_coeff * torch.mean((l1 + l2 + l3)**2)
    total_loss = data_loss + physics_loss
    return total_loss 

# Define network layers
layers = [1, 64, 64, 64, 64, 64, 3]  
k1_layers = [1, 64, 64, 64, 64, 64, 1]
k2_layers = [1, 64, 64, 64, 64, 64, 1]
k3_layers = [1, 64, 64, 64, 64, 64, 1]




# Hyperparameters
learning_rate = 0.001
batch_size = 50
n_epochs = 5000
ntotal = 1000

# Training function with tqdm progress bar
def train(model, epochs, t_data, y1_data, y2_data, y3_data):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    progress_bar = tqdm(range(epochs), desc='Training Progress')
    for epoch in progress_bar:
        model.train()
        idx = np.random.choice(len(t_data), batch_size, replace=False)
        t_batch, y1_batch, y2_batch, y3_batch = t_data[idx], y1_data[idx], y2_data[idx], y3_data[idx]

        optimizer.zero_grad()
        loss = custom_loss(model, t_batch, y1_batch, y2_batch, y3_batch)
        loss.backward()
        optimizer.step()

        # Update tqdm progress bar with the current loss
        progress_bar.set_postfix({'loss': loss.item()})

# Saving the model
def save_model(model, filename):
    torch.save(model.state_dict(), filename)

# Transfer weights and biases
def transfer_learning(source_model, target_model):
    with torch.no_grad():
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(source_param.data)

# Main training loop with pretraining
t_span = [0.0, 1e5]
y_init = [1.0, 0.0, 0.0]
#t_eval = np.logspace(np.log10(t_span[0]), np.log10(t_span[1]), 1000)
t_eval = 10 ** np.linspace(-5, np.log10(t_span[1]), ntotal)

main_model = PITLNN(layers, k1_layers, k2_layers, k3_layers)

# Pretraining on different parameters
k1_values = [0.04 + delta for delta in np.linspace(-0.01, 0.01, 2)]
k2_values = [3e7 + delta for delta in np.linspace(-1e6, 1e6, 2)]
k3_values = [1e4 + delta for delta in np.linspace(-1000, 1000, 2)]


# Assuming k1_values, k2_values, k3_values are lists of values you want to use for training
for k1 in k1_values:
    for k2 in k2_values:
        for k3 in k3_values:
            sol = solve_ivp(lambda t, y: ode_system(t, y, k1, k2, k3), t_span, y_init, t_eval=t_eval,method='BDF', dense_output=True)
            t_data, y1_data, y2_data, y3_data = map(torch.tensor, [sol.t, sol.y[0], sol.y[1], sol.y[2]])
            t_data = t_data.float().view(-1, 1)
            y1_data = y1_data.float().view(-1, 1)
            y2_data = y2_data.float().view(-1, 1)
            y3_data = y3_data.float().view(-1, 1)

            # Initialize the model for this specific set of parameters
            model = PITLNN(layers, k1_layers, k2_layers, k3_layers)
            # Train the model
            train(model, n_epochs, t_data, y1_data, y2_data, y3_data)
            # Save the model
            save_model(model, f'model_k1_{k1}_k2_{k2}_k3_{k3}.pth')
            # Transfer learned parameters to the main model
            transfer_learning(model, main_model)

# Save the main model after all transfers
save_model(main_model, 'final_main_model.pth')
