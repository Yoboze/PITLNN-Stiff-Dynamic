import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
import csv
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score


# Define the updated ODE system for the Rober problem
def rober(t, y, k1, k2, k3):
    y1, y2, y3 = y
    dydt = [-k1 * y1 + k3 * y2 * y3,
            k1 * y1 - k3 * y2 * y3 - k2 * y2**2,
            k2 * y2**2]
    return dydt

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
    
    def predict(self, t, yscale):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            y1_pred, y2_pred, y3_pred, k1_pred, k2_pred, k3_pred = self(t)
        # Denormalize predictions
        y1_pred *= yscale[0]
        y2_pred *= yscale[1]
        y3_pred *= yscale[2]
        return y1_pred, y2_pred, y3_pred, k1_pred, k2_pred, k3_pred


# Load model weights
def load_model_weights(model, filename):
    model.load_state_dict(torch.load(filename))

# Custom loss function with regularization
def custom_loss(model, t_data, y1_data, y2_data, y3_data, phy_coeff = 1e-7, reg_coeff=1e-10):
    y1_pred, y2_pred, y3_pred, k1, k2, k3 = model(t_data)
    l1 = y1_pred - (-k1 * y1_pred + k3 * y2_pred * y3_pred)
    l2 = y2_pred - (k1 * y1_pred - k3 * y2_pred * y3_pred - k2 * y2_pred**2)
    l3 = y3_pred - k2 * y2_pred**2
    data_loss = torch.mean((y1_pred - y1_data)**2 + (y2_pred - y2_data)**2 + (y3_pred - y3_data)**2)
    physics_loss = phy_coeff * torch.mean((l1 + l2 + l3)**2)
    regularization_loss = reg_coeff * sum(p.pow(2.0).sum() for p in model.parameters())
    total_loss = data_loss + physics_loss + regularization_loss
    return total_loss, data_loss, physics_loss, regularization_loss

# Define network layers
layers = [1, 64, 64, 64, 64, 64, 3]  
k1_layers = [1, 64, 64, 64, 64, 64, 1]
k2_layers = [1, 64, 64, 64, 64, 64, 1]
k3_layers = [1, 64, 64, 64, 64, 64, 1]


# Initialize the main model
main_model = PITLNN(layers, k1_layers, k2_layers, k3_layers)

# Load weights from a pretrained model
pretrained_filenames = ['model_k1_0.03_k2_29000000.0_k3_9000.0(1).pth']  # Example filenames
for filename in pretrained_filenames:
    load_model_weights(main_model, filename)

# Initial settings
ntotal = 50
batch_size = 50


u0 = [1.0, 0, 0]
tspan = (0.0, 1e5)
k = [0.04, 3e7, 1e4]
tsteps = 10 ** np.linspace(-5, np.log10(tspan[1]), num=ntotal)
sol = solve_ivp(lambda t, y: rober(t, y, *k), tspan, u0, method='BDF', max_step=1e4, t_eval=tsteps, atol=1e-9)

# Scale for each species
normdata = sol.y
yscale = np.max(normdata, axis=1)  # scale for each species

# Define the model dynamics within the ODE solver
def model_dynamics(t, y, model):
    with torch.no_grad():
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        dydt = model.forward(y_tensor).squeeze().numpy()
    return dydt * yscale / tspan[1]


# Parameters from ODE solution
t_data = tsteps  # Time data points
data_scaled = sol.y / np.max(sol.y, axis=1).reshape(-1, 1)  # Normalized data for training

# Initialize the AdamW optimizer (with specified parameters from the initial setup)
params = [p for p in main_model.parameters()]
optimizer = optim.AdamW(params, lr=0.0001, betas=(0.99, 0.9999), weight_decay=1e-4)

# Training parameters
n_epochs = 50000
batch_size = 50

# Prepare metrics file
metrics_file = 'training_metrics.csv'  # Define the file path
with open(metrics_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Total Loss', 'Data Loss', 'Physics Loss', 'Regularization Loss', 'Gradient Norm'])

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    y_true_np = y_true.detach().numpy()
    y_pred_np = y_pred.detach().numpy()
    metrics = {
        'L2 Norm': np.linalg.norm(y_true_np - y_pred_np, 2),
        'L Infinity Norm': np.linalg.norm(y_true_np - y_pred_np, np.inf),
        'R-squared': r2_score(y_true_np, y_pred_np),
        'MAE': mean_absolute_error(y_true_np, y_pred_np),
        'MSE': mean_squared_error(y_true_np, y_pred_np),
        #'RMSE': mean_squared_error(y_true_np, y_pred_np, squared=False),
        'Explained Variance': explained_variance_score(y_true_np, y_pred_np),
        'MAPE': np.mean(np.abs((y_true_np - y_pred_np) / y_true_np)) * 100
    }
    return metrics

# Create the directory for figures if it doesn't exist
figs_dir = 'figs'
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)
    
    
# Define start and end ranges for time filtering
start_range = 1e-5
end_range = 1e5
    
# Initialize loss lists before the training loop
total_losses = []
data_losses = []
physics_losses = []
regularization_losses = []

# Training loop
for epoch in tqdm(range(n_epochs), desc='Training'):
    idx = np.random.choice(t_data.shape[0], batch_size, replace=False)
    t_batch = torch.tensor(t_data[idx], dtype=torch.float32).view(-1, 1)
    y1_batch = torch.tensor(data_scaled[0, idx], dtype=torch.float32).unsqueeze(1)
    y2_batch = torch.tensor(data_scaled[1, idx], dtype=torch.float32).unsqueeze(1)
    y3_batch = torch.tensor(data_scaled[2, idx], dtype=torch.float32).unsqueeze(1)

    optimizer.zero_grad()
    y1_pred, y2_pred, y3_pred, k1_pred, k2_pred, k3_pred = main_model(t_batch)
    total_loss, data_loss, physics_loss, regularization_loss = custom_loss(main_model, t_batch, y1_batch, y2_batch, y3_batch)
    total_loss.backward()
    optimizer.step()
    
    # Inside the training loop, append each loss at the end of each epoch
    total_losses.append(total_loss.item())
    data_losses.append(data_loss.item())
    physics_losses.append(physics_loss.item())
    regularization_losses.append(regularization_loss.item())
    
    # Calculate additional metrics
    combined_true = torch.cat([y1_batch, y2_batch, y3_batch], dim=1)
    combined_pred = torch.cat([y1_pred, y2_pred, y3_pred], dim=1)
    metrics = calculate_metrics(combined_true, combined_pred)

    # Calculate gradient norm
    grad_norm = sum(p.grad.data.norm(2).item() for p in main_model.parameters())

    # Plotting and saving figures at specified epochs
    if epoch % 1000 == 0 and epoch != 0:  # adjust according to your preference for how often to plot
        # Forward pass with current parameters
        t_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)
        y1_pred, y2_pred, y3_pred, k1_pred, k2_pred, k3_pred = main_model.predict(t_tensor, yscale)
        
        # Convert tensors to numpy arrays for plotting
        t_np = t_tensor.numpy().flatten()
        y1_pred_np = y1_pred.numpy().flatten()
        y2_pred_np = y2_pred.numpy().flatten()
        y3_pred_np = y3_pred.numpy().flatten()
        
        k1_pred_np = k1_pred.numpy().flatten()
        k2_pred_np = k2_pred.numpy().flatten()
        k3_pred_np = k3_pred.numpy().flatten()
        
        
        # Filtering indices based on time range
        indices = np.where((t_np >= start_range) & (t_np <= end_range))[0]
        t_filtered = t_np[indices]


        # Actual data
        y1_data = sol.y[0, :]
        y2_data = sol.y[1, :]
        y3_data = sol.y[2, :]
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(t_np, y1_data, 'r', label="Actual Values")
        plt.plot(t_np, y1_pred_np, 'g--', label="PITLNN")
        plt.xlabel('Time')
        plt.ylabel('y1(t)')
        plt.title(f" ROBER - Species y1")
        plt.legend()
        plt.xscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'y1_epoch_{epoch}.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(t_np, y2_data, 'b', label="Actual Values")
        plt.plot(t_np, y2_pred_np, 'k--', label="PITLNN")
        plt.xlabel('Time')
        plt.ylabel('y2(t)')
        plt.title(f" ROBER - Species y2")
        plt.legend()
        plt.xscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'y2_epoch_{epoch}.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(t_np, y3_data, 'g', label="Actual Values")
        plt.plot(t_np, y3_pred_np, 'm--', label="PITLNN")
        plt.xlabel('Time')
        plt.ylabel('y3(t)')
        plt.title(f"ROBER - Species y3")
        plt.legend()
        plt.xscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'y3_epoch_{epoch}.png'))
        plt.close()
        
        # Plotting for k1_pred, k2_pred, k3_pred
        plt.figure(figsize=(10, 6))
        plt.plot(t_filtered, k1_pred_np[indices], 'r', label=f'Predicted k1 at Epoch {epoch}')
        plt.xlabel('Time')
        plt.ylabel('k1')
        plt.title('Prediction of Parameter k1 over Time')
        plt.xscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'k1_epoch_{epoch}.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(t_filtered, k2_pred_np[indices], 'b', label=f'Predicted k2 at Epoch {epoch}')
        plt.xlabel('Time')
        plt.ylabel('k2')
        plt.title('Prediction of Parameter k2 over Time')
        plt.xscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'k2_epoch_{epoch}.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(t_filtered, k3_pred_np[indices], 'g', label=f'Predicted k3 at Epoch {epoch}')
        plt.xlabel('Time')
        plt.ylabel('k3')
        plt.title('Prediction of Parameter k3 over Time')
        plt.xscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'k3_epoch_{epoch}.png'))
        plt.close()

        # Optional: Print update message
        print(f"Plots for epoch {epoch} saved.")

        # Logging to console
        print(f'Epoch {epoch}: Total Loss = {total_loss.item()}, Data Loss = {data_loss.item()}, Physics Loss = {physics_loss.item()}, Regularization Loss = {regularization_loss.item()}, Gradient Norm = {grad_norm}')
        print(f"Additional Metrics at Epoch {epoch}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.7f}")

        # Logging to CSV file
        with open(metrics_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, total_loss.item(), data_loss.item(), physics_loss.item(), regularization_loss.item(), grad_norm] + [metrics[metric] for metric in sorted(metrics)])
            
        # Save checkpoint every 1000 epochs
        checkpoint_filename = f'main_model_checkpoint_{epoch}.pth'
        torch.save(main_model.state_dict(), checkpoint_filename)

# Save the final model after all epochs are completed
torch.save(main_model.state_dict(), 'final_trained_main_model.pth')

# After training, plot the losses
plt.figure(figsize=(10, 6))
plt.plot(total_losses, label='Total Loss')
plt.plot(physics_losses, label='Physics Loss')
plt.plot(regularization_losses, label='Regularization Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('losses_over_epochs.png')
plt.show()
