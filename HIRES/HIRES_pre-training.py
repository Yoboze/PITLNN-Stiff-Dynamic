import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
from tqdm import tqdm

# Define the ODE system
# def ode_system(t, y, k1, k2, k3):
#     y1, y2, y3 = y
#     dy1dt = -k1 * y1 + k3 * y2 * y3
#     dy2dt = k1 * y1 - k2 * y2**2 - k3 * y2 * y3
#     dy3dt = k2 * y2**2
#     return [dy1dt, dy2dt, dy3dt]

def hires_odes(t, y, r1, r2, r3, r5, r6, r10):
    y1, y2, y3, y4, y5, y6, y7, y8 = y
    dy1dt = -r1 * y1 + r2 * y2 + r3 * y3 + 0.0007
    dy2dt = r1 * y1 - r5 * y2
    dy3dt = -r6 * y3 + r2 * y4 + 0.035 * y5 
    dy4dt = r3 * y2 + r1 * y3 - r10 * y4
    dy5dt = -1.745 * y5 + r2 * y6 + r2 * y7  
    dy6dt = -280 * y6 * y8 + 0.69 * y4 + r1 * y5 - r2 * y6 + 0.69 * y7  
    dy7dt = 280 * y6 * y8 - 1.81 * y7 
    dy8dt = -280 * y6 * y8 + 1.81 * y7  
    return [dy1dt, dy2dt, dy3dt, dy4dt, dy5dt, dy6dt, dy7dt, dy8dt]



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

    
def net_l(self, t):
    t.requires_grad = True
    y1, y2, y3, y4, y5, y6, y7, y8, r1, r2, r3, r5, r6, r10 = self.forward(t)
    y1_t = torch.autograd.grad(y1, t, grad_outputs=torch.ones_like(y1), create_graph=True)[0]
    y2_t = torch.autograd.grad(y2, t, grad_outputs=torch.ones_like(y2), create_graph=True)[0]
    y3_t = torch.autograd.grad(y3, t, grad_outputs=torch.ones_like(y3), create_graph=True)[0]
    y4_t = torch.autograd.grad(y4, t, grad_outputs=torch.ones_like(y4), create_graph=True)[0]
    y5_t = torch.autograd.grad(y5, t, grad_outputs=torch.ones_like(y5), create_graph=True)[0]
    y6_t = torch.autograd.grad(y6, t, grad_outputs=torch.ones_like(y6), create_graph=True)[0]
    y7_t = torch.autograd.grad(y7, t, grad_outputs=torch.ones_like(y7), create_graph=True)[0]
    y8_t = torch.autograd.grad(y8, t, grad_outputs=torch.ones_like(y8), create_graph=True)[0]
    l1 = y1_t - (-r1 * y1 + r2 * y2 + r3 * y3 + 0.0007)
    l2 = y2_t - (r1 * y1 - r5 * y2)
    l3 = y3_t - (-r6 * y3 + r2 * y4 + 0.035 * y5)
    l4 = y4_t - (r3 * y2 + r1 * y3 - r10 * y4)
    l5 = y5_t - (-1.745 * y5 + r2 * y6 + r2 * y7)
    l6 = y6_t - (-280 * y6 * y8 + 0.69 * y4 + r1 * y5 - r2 * y6 + 0.69 * y7)
    l7 = y7_t - (280 * y6 * y8 - 1.81 * y7)
    l8 = y8_t - (-280 * y6 * y8 + 1.81 * y7)
    return l1, l2, l3, l4, l5, l6, l7, l8



def custom_loss(model, t_data, y1_data, y2_data, y3_data, y4_data, y5_data, y6_data, y7_data, y8_data):
    y1_pred, y2_pred, y3_pred, y4_pred, y5_pred, y6_pred, y7_pred, y8_pred, r1, r2, r3, r5, r6, r10 = model(t_data)
    l1, l2, l3, l4, l5, l6, l7, l8 = net_l(model, t_data)
    data_loss = torch.mean((y1_pred - y1_data)**2 + (y2_pred - y2_data)**2 + (y3_pred - y3_data)**2 + (y4_pred - y4_data)**2 + (y5_pred - y5_data)**2 + (y6_pred - y6_data)**2 + (y7_pred - y7_data)**2 + (y8_pred - y8_data)**2)
    physics_loss = torch.mean((l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8)**2)
    total_loss = data_loss + physics_loss
    return total_loss 

# Define network layers
# layers = [1, 128, 128, 128, 128, 128, 128, 128, 8]  
# r1_layers = [1, 128, 128, 128, 128, 128, 128, 128, 1]
# r2_layers = [1, 128, 128, 128, 128, 128, 128, 128, 1]
# r3_layers = [1, 128, 128, 128, 128, 128, 128, 128, 1]
# r5_layers = [1, 128, 128, 128, 128, 128, 128, 128, 1]
# r6_layers = [1, 128, 128, 128, 128, 128, 128, 128, 1]
# r10_layers = [1, 128, 128, 128, 128, 128, 128, 128, 1]

layers = [1, 64, 64, 64, 64, 64, 64, 64, 8]  
r1_layers = [1, 64, 64, 64, 64, 64, 64, 64, 1]
r2_layers = [1, 64, 64, 64, 64, 64, 64, 64, 1]
r3_layers = [1, 64, 64, 64, 64, 64, 64, 64, 1]
r5_layers = [1, 64, 64, 64, 64, 64, 64, 64, 1]
r6_layers = [1, 64, 64, 64, 64, 64, 64, 64, 1]
r10_layers = [1, 64, 64, 64, 64, 64, 64, 64, 1]






# Hyperparameters
learning_rate = 0.0001
batch_size = 50
n_epochs = 5000
ntotal = 1000

# Training function with tqdm progress bar
def train(model, epochs, t_data, y1_data, y2_data, y3_data, y4_data, y5_data, y6_data, y7_data, y8_data):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    progress_bar = tqdm(range(epochs), desc='Training Progress')
    for epoch in progress_bar:
        model.train()
        idx = np.random.choice(len(t_data), batch_size, replace=False)
        t_batch, y1_batch, y2_batch, y3_batch, y4_batch, y5_batch, y6_batch, y7_batch, y8_batch = t_data[idx], y1_data[idx], y2_data[idx], y3_data[idx], y4_data[idx], y5_data[idx], y6_data[idx], y7_data[idx], y8_data[idx]

        optimizer.zero_grad()
        # Assume custom_loss function can handle multiple outputs
        loss = custom_loss(model, t_batch, y1_batch, y2_batch, y3_batch, y4_batch, y5_batch, y6_batch, y7_batch, y8_batch)
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
t_span = [0, 321.8122]
y_init = [1, 0, 0, 0, 0, 0, 0, 0.0057]
t_eval = np.linspace(t_span[0], t_span[1], 1000)

main_model = PITLNN(layers, r1_layers, r2_layers, r3_layers, r5_layers, r6_layers, r10_layers)

# Pretraining on different parameters
r1_values = [1.71 + delta for delta in np.linspace(-0.01, 0.01, 2)]
r2_values = [0.43 + delta for delta in np.linspace(-0.03, 0.03, 2)]
r3_values = [8.32 + delta for delta in np.linspace(-0.02, 0.02, 2)]
r5_values = [8.75 + delta for delta in np.linspace(-0.05, 0.05, 2)]
r6_values = [10.03 + delta for delta in np.linspace(-0.03, 0.03, 2)]
r10_values = [1.12 + delta for delta in np.linspace(-0.02, 0.02, 2)]



for r1 in r1_values:
    for r2 in r2_values:
        for r3 in r3_values:
            for r5 in r5_values:
                for r6 in r6_values:
                    for r10 in r10_values:
                        sol = solve_ivp(lambda t, y: hires_odes(t, y, r1, r2, r3, r5, r6, r10), t_span, y_init, t_eval=t_eval, dense_output=True)
                        t_data, y1_data, y2_data, y3_data, y4_data, y5_data, y6_data, y7_data, y8_data = map(torch.tensor, [sol.t, *sol.y])
                        t_data = t_data.float().view(-1, 1)
                        y_data = [y.float().view(-1, 1) for y in [y1_data, y2_data, y3_data, y4_data, y5_data, y6_data, y7_data, y8_data]]


                        # Initialize the model for this specific set of parameters
                        model = PITLNN(layers, r1_layers, r2_layers, r3_layers, r5_layers, r6_layers, r10_layers)
                        # Train the model
                        train(model, n_epochs, t_data, *y_data)
                        # Save the model
                        save_model(model, f'model_r1_{r1}_r2_{r2}_r3_{r3}_r5_{r5}_r6_{r6}_r10_{r10}.pth')
                        # Transfer learned parameters to the main model
                        transfer_learning(model, main_model)

# Save the main model after all transfers
save_model(main_model, 'final_main_model.pth')
