import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
from tqdm import tqdm
import csv 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score

# Define the ODE system
def ode_system(t, y, c, k):
    z1, z2 = y
    return [z2, -c*z1 - k*z2]

# Define the Physics-Informed Transfer Learning Neural Network (PITLNN) class
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
    
    
    def predict(self, t_star):
        t_star_tensor = torch.tensor(t_star, dtype=torch.float32)
        with torch.no_grad():
            z1_star, z2_star, c_star, k_star = self.forward(t_star_tensor)
        return z1_star.numpy(), z2_star.numpy(), c_star.numpy(), k_star.numpy()

# Load model weights
def load_model_weights(model, filename):
    model.load_state_dict(torch.load(filename))

def net_l(self, t):
    t.requires_grad = True
    z1, z2, c, k = self.forward(t)
    z1_t = torch.autograd.grad(z1, t, grad_outputs=torch.ones_like(z1), create_graph=True)[0]
    z2_t = torch.autograd.grad(z2, t, grad_outputs=torch.ones_like(z2), create_graph=True)[0]
    l1 = z1_t - z2
    l2 = z2_t + c * z2 + k * z1
    return l1, l2

# Custom loss function with regularization
def custom_loss(model, t_data, z1_data, z2_data, phy_coeff = 10, reg_coeff = 1e-10):
    z1_pred, z2_pred, c, k = model(t_data)
    l1, l2 = net_l(model, t_data)
    data_loss = torch.mean((z1_pred - z1_data)**2 + (z2_pred - z2_data)**2)
    physics_loss = phy_coeff * torch.mean((l1 + l2)**2)
    regularization_loss = reg_coeff * sum(p.pow(2.0).sum() for p in model.parameters())
    total_loss = data_loss + physics_loss + regularization_loss
    return total_loss, data_loss, physics_loss, regularization_loss


# Define network layers
layers = [1, 64, 64, 64, 64, 64, 2]
c_layers = [1, 64, 64, 64, 64, 64, 1]
k_layers = [1, 64, 64, 64, 64, 64, 1]

# Hyperparameters
learning_rate = 0.0001
batch_size = 50
reg_coeff = 1e-10
n_epochs = 50000

# Initialize the main model
main_model = PITLNN(layers, c_layers, k_layers)

# Load weights from a pretrained model
pretrained_filenames = ['model_c991.0_k1010.0.pth']  # Example filenames
for filename in pretrained_filenames:
    load_model_weights(main_model, filename)  # Assuming you are aggregating or selecting the best

# Further training of the main model
t_span = [0, 10]
y_init = [1.0, 1.0]
t_eval = np.linspace(t_span[0], t_span[1], 1000)

sol = solve_ivp(lambda t, y: ode_system(t, y, 1001, 1000), t_span, y_init, t_eval=t_eval, method='BDF')
t_data, z1_data, z2_data = map(torch.tensor, [sol.t, sol.y[0], sol.y[1]])
t_data = t_data.float().view(-1, 1)
z1_data = z1_data.float().view(-1, 1)
z2_data = z2_data.float().view(-1, 1)


# Prepare metrics file
metrics_file = 'training_metrics.csv'  # Define the file path
with open(metrics_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Total Loss', 'Data Loss', 'Physics Loss', 'Regularization Loss', 'Gradient Norm'])



# Training loop
optimizer = optim.Adam(main_model.parameters(), lr=learning_rate)
progress_bar = tqdm(range(n_epochs), desc='Main Model Training')

for epoch in progress_bar:
    main_model.train()
    idx = np.random.choice(len(t_data), batch_size, replace=False)
    t_batch, z1_batch, z2_batch = t_data[idx], z1_data[idx], z2_data[idx]

    optimizer.zero_grad()
    # Use reg_coeff instead of reg_coef
    loss, data_loss, physics_loss, regularization_loss = custom_loss(main_model, t_batch, z1_batch, z2_batch, reg_coeff)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(main_model.parameters(), max_norm=10)
    optimizer.step()
    

    # Save metrics
    with open(metrics_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, loss.item(), data_loss.item(), physics_loss.item(), regularization_loss.item(), grad_norm])

    progress_bar.set_postfix({'loss': loss.item()})

    # Save model checkpoints at specific epochs
    if epoch % 1000 == 0 and epoch != 0:
        checkpoint_filename = f'main_model_checkpoint_{epoch}.pth'
        torch.save(main_model.state_dict(), checkpoint_filename)


# Save the final model after all epochs are completed
torch.save(main_model.state_dict(), 'final_trained_main_model.pth')