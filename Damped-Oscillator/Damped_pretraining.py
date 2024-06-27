import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define the ODE system
def ode_system(t, y, c, k):
    z1, z2 = y
    return [z2, -c*z1 - k*z2]

# Define the Physics-Informed Transfer learning Neural Network (PITLNN) class
class PITLNN(nn.Module):
    def __init__(self, layers, c_layers, k_layers):
        super(PITLNN, self).__init__()
        self.net = self.initialize_NN(layers)
        self.c_net = self.initialize_NN(c_layers)
        self.k_net = self.initialize_NN(k_layers)

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
        z = self.net(t)
        c = self.c_net(t)
        k = self.k_net(t)
        z1 = z[:, 0].view(-1, 1)
        z2 = z[:, 1].view(-1, 1)
        return z1, z2, c, k

def net_l(model, t):
    t.requires_grad = True
    z1, z2, c, k = model(t)
    z1_t = torch.autograd.grad(z1, t, grad_outputs=torch.ones_like(z1), create_graph=True)[0]
    z2_t = torch.autograd.grad(z2, t, grad_outputs=torch.ones_like(z2), create_graph=True)[0]
    l1 = z1_t - z2
    l2 = z2_t + c * z2 + k * z1
    return l1, l2

# Custom loss function with regularization
def custom_loss(model, t_data, z1_data, z2_data):
    z1_pred, z2_pred, c, k = model(t_data)
    l1, l2 = net_l(model, t_data)
    data_loss = torch.mean((z1_pred - z1_data)**2 + (z2_pred - z2_data)**2)
    physics_loss = torch.mean((l1 + l2)**2)
    total_loss = data_loss + physics_loss
    return total_loss

# Define network layers
layers = [1, 64, 64, 64, 64, 64, 2]
c_layers = [1, 64, 64, 64, 64, 64, 1]
k_layers = [1, 64, 64, 64, 64, 64, 1]

# Hyperparameters
learning_rate = 0.001
batch_size = 50
n_epochs = 5000

# Training function with tqdm progress bar
def train(model, epochs, t_data, z1_data, z2_data):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    progress_bar = tqdm(range(epochs), desc='Training Progress')
    for epoch in progress_bar:
        model.train()
        idx = np.random.choice(len(t_data), batch_size, replace=False)
        t_batch, z1_batch, z2_batch = t_data[idx], z1_data[idx], z2_data[idx]

        optimizer.zero_grad()
        loss = custom_loss(model, t_batch, z1_batch, z2_batch)
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

# Plotting function
def plot_results(model, t_data, z1_data, z2_data, t_test, title_suffix):
    t_tensor = torch.tensor(t_test, dtype=torch.float32).view(-1, 1)
    
    model.eval()
    with torch.no_grad():
        z1_pred, z2_pred, c_pred, k_pred = model(t_tensor)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(t_test, z1_pred.numpy(), 'g', label='z1_pred')
    plt.plot(t_data.numpy(), z1_data.numpy(), 'm--', label='z1_data')
    plt.title(f'z1 over time {title_suffix}')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(t_test, z2_pred.numpy(),  'k', label='z2_pred')
    plt.plot(t_data.numpy(), z2_data.numpy(), 'b--', label='z2_data')
    plt.title(f'z2 over time {title_suffix}')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(t_test, c_pred.numpy(), 'r', label='Parameter c')
    plt.title(f'Parameter c over time {title_suffix}')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(t_test, k_pred.numpy(), 'k', label='Parameter k')
    plt.title(f'Parameter k over time {title_suffix}')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main training loop with pretraining
t_span = [0, 10]
y_init = [1.0, 1.0]
t_eval = np.linspace(t_span[0], t_span[1], 200)

main_model = PITLNN(layers, c_layers, k_layers)

# Pretraining on different parameters
cs = [1001 + delta for delta in np.linspace(-10, 10, 2)]
ks = [1000 + delta for delta in np.linspace(-10, 10, 2)]

for c in cs:
    for k in ks:
        sol = solve_ivp(lambda t, y: ode_system(t, y, c, k), t_span, y_init, t_eval=t_eval)
        t_data, z1_data, z2_data = map(torch.tensor, [sol.t, sol.y[0], sol.y[1]])
        t_data = t_data.float().view(-1, 1)
        z1_data = z1_data.float().view(-1, 1)
        z2_data = z2_data.float().view(-1, 1)

        model = PITLNN(layers, c_layers, k_layers)
        train(model, n_epochs, t_data, z1_data, z2_data)
        save_model(model, f'model_c{c}_k{k}.pth')
        transfer_learning(model, main_model)

save_model(main_model, 'final_main_model.pth')

# Define the time span and test points
t_span = [0, 10]
t_test = np.linspace(t_span[0], t_span[1], 200)

# Load each model, make predictions, and plot
for c in cs:
    for k in ks:
        model_filename = f'model_c{c}_k{k}.pth'
        model = PITLNN(layers, c_layers, k_layers)
        model.load_state_dict(torch.load(model_filename))

        # Load the corresponding training data
        sol = solve_ivp(lambda t, y: ode_system(t, y, c, k), t_span, [1.0, 1.0], t_eval=t_test)
        t_data, z1_data, z2_data = map(torch.tensor, [sol.t, sol.y[0], sol.y[1]])
        t_data = t_data.float().view(-1, 1)
        z1_data = z1_data.float().view(-1, 1)
        z2_data = z2_data.float().view(-1, 1)

        plot_results(model, t_data, z1_data, z2_data, t_test, f'for c={c}, k={k}')
