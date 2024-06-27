### import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
import csv
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score



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
        self.r1_net = self.initialize_NN1(r1_layers)
        self.r2_net = self.initialize_NN2(r2_layers)
        self.r3_net = self.initialize_NN3(r3_layers)
        self.r5_net = self.initialize_NN5(r5_layers)
        self.r6_net = self.initialize_NN6(r6_layers)
        self.r10_net = self.initialize_NN10(r10_layers)
        
    def initialize_NN(self, layers):
        modules = []
        for i in range(len(layers)-2):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(nn.GELU())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        nn.init.xavier_uniform_(modules[-1].weight)
        nn.init.zeros_(modules[-1].bias)
        return nn.Sequential(*modules)
    
    def initialize_NN1(self, r1_layers):
        modules = []
        for i in range(len(r1_layers)-2):
            modules.append(nn.Linear(r1_layers[i], r1_layers[i+1]))
            modules.append(nn.GELU())
        modules.append(nn.Linear(r1_layers[-2], r1_layers[-1]))
        nn.init.xavier_uniform_(modules[-1].weight)
        nn.init.zeros_(modules[-1].bias)
        return nn.Sequential(*modules)
    
    def initialize_NN2(self, r2_layers):
        modules = []
        for i in range(len(r2_layers)-2):
            modules.append(nn.Linear(r2_layers[i], r2_layers[i+1]))
            modules.append(nn.GELU())
        modules.append(nn.Linear(r2_layers[-2], r2_layers[-1]))
        nn.init.xavier_uniform_(modules[-1].weight)
        nn.init.zeros_(modules[-1].bias)
        return nn.Sequential(*modules)
    
    def initialize_NN3(self, r3_layers):
        modules = []
        for i in range(len(r3_layers)-2):
            modules.append(nn.Linear(r3_layers[i], r3_layers[i+1]))
            modules.append(nn.GELU())
        modules.append(nn.Linear(r3_layers[-2], r3_layers[-1]))
        nn.init.xavier_uniform_(modules[-1].weight)
        nn.init.zeros_(modules[-1].bias)
        return nn.Sequential(*modules)
    
    def initialize_NN5(self, r5_layers):
        modules = []
        for i in range(len(r5_layers)-2):
            modules.append(nn.Linear(r5_layers[i], r5_layers[i+1]))
            modules.append(nn.GELU())
        modules.append(nn.Linear(r5_layers[-2], r5_layers[-1]))
        nn.init.xavier_uniform_(modules[-1].weight)
        nn.init.zeros_(modules[-1].bias)
        return nn.Sequential(*modules)
    
    def initialize_NN6(self, r6_layers):
        modules = []
        for i in range(len(r6_layers)-2):
            modules.append(nn.Linear(r6_layers[i], r6_layers[i+1]))
            modules.append(nn.GELU())
        modules.append(nn.Linear(r6_layers[-2], r6_layers[-1]))
        nn.init.xavier_uniform_(modules[-1].weight)
        nn.init.zeros_(modules[-1].bias)
        return nn.Sequential(*modules)
    
    def initialize_NN10(self, r10_layers):
        modules = []
        for i in range(len(r10_layers)-2):
            modules.append(nn.Linear(r10_layers[i], r10_layers[i+1]))
            modules.append(nn.GELU())
        modules.append(nn.Linear(r10_layers[-2], r10_layers[-1]))
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
    
    def predict(self, t, yscale):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            y1_pred, y2_pred, y3_pred, y4_pred, y5_pred, y6_pred, y7_pred, y8_pred, r1_pred, r2_pred, r3_pred, r5_pred, r6_pred, r10_pred = self(t)
        # Denormalize predictions
        y1_pred *= yscale[0]
        y2_pred *= yscale[1]
        y3_pred *= yscale[2]
        y4_pred *= yscale[3]
        y5_pred *= yscale[4]
        y6_pred *= yscale[5]
        y7_pred *= yscale[6]
        y8_pred *= yscale[7]
        return y1_pred, y2_pred, y3_pred, y4_pred, y5_pred, y6_pred, y7_pred, y8_pred, r1_pred, r2_pred, r3_pred, r5_pred, r6_pred, r10_pred


# Load model weights
def load_model_weights(model, filename):
    model.load_state_dict(torch.load(filename))

# Custom loss function with regularization
def custom_loss(model, t_data, y1_data, y2_data, y3_data, y4_data, y5_data, y6_data, y7_data, y8_data, phy_coeff=0.00000001, reg_coeff=0.00000001):
    y1_pred, y2_pred, y3_pred, y4_pred, y5_pred, y6_pred, y7_pred, y8_pred, r1, r2, r3, r5, r6, r10 = model(t_data)
    l1 = y1_pred - (-r1 * y1_pred + r2 * y2_pred + r3 * y3_pred + 0.0007)
    l2 = y2_pred - (r1 * y1_pred - r5 * y2_pred)
    l3 = y3_pred - (-r6 * y3_pred + r2 * y4_pred + 0.035 * y5_pred)
    l4 = y4_pred - (r3 * y2_pred + r1 * y3_pred - r10 * y4_pred)
    l5 = y5_pred - (-1.745 * y5_pred + r2 * y6_pred + r2 * y7_pred)
    l6 = y6_pred - (-280 * y6_pred * y8_pred + 0.69 * y4_pred + r1 * y5_pred - r2 * y6_pred + 0.69 * y7_pred)
    l7 = y7_pred - (280 * y6_pred * y8_pred - 1.81 * y7_pred)
    l8 = y8_pred - (-280 * y6_pred * y8_pred + 1.81 * y7_pred)
    data_loss = torch.mean((y1_pred - y1_data)**2 + (y2_pred - y2_data)**2 + (y3_pred - y3_data)**2 + (y4_pred - y4_data)**2 + (y5_pred - y5_data)**2 + (y6_pred - y6_data)**2 + (y7_pred - y7_data)**2 + (y8_pred - y8_data)**2)
    physics_loss = phy_coeff * torch.mean((l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8)**2)
    regularization_loss = reg_coeff * sum(p.pow(2.0).sum() for p in model.parameters())
    total_loss = data_loss + physics_loss + regularization_loss
    return total_loss, data_loss, physics_loss, regularization_loss

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

# Load weights from a pretrained model
pretrained_filenames = ['model_r1_1.72_r2_0.4_r3_8.3_r5_8.7_r6_10.0_r10_1.1.pth']  # Example filenames
for filename in pretrained_filenames:
    load_model_weights(main_model, filename)

# Initial settings
ntotal = 1000
batch_size = 50



u0 = [1, 0, 0, 0, 0, 0, 0, 0.0057]
tspan = (0, 321.8122)
r = [1.71, 0.43, 8.32, 8.75, 10.03, 1.12]
r1, r2, r3, r5, r6, r10 = r
tsteps = np.linspace(tspan[0], tspan[1], num=ntotal)
sol = solve_ivp(lambda t, y: hires_odes(t, y, r1, r2, r3, r5, r6, r10), tspan, u0, method='BDF', t_eval=tsteps)

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
optimizer = optim.AdamW(params, lr=0.0001, betas=(0.99, 0.9999), weight_decay=1e-5)

# Training parameters
n_epochs = 100000
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
        'RMSE': mean_squared_error(y_true_np, y_pred_np, squared=False),
        'Explained Variance': explained_variance_score(y_true_np, y_pred_np),
        'MAPE': np.mean(np.abs((y_true_np - y_pred_np) / y_true_np)) * 100
    }
    return metrics

# Create the directory for figures if it doesn't exist
figs_dir = 'figs'
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)   

    
# Define start and end ranges for time filtering
start_range = 0
end_range = 321.8122    
    

# Initialize loss lists before the training loop
total_losses = []
data_losses = []
physics_losses = []
regularization_losses = []
    
    
# Training loop
for epoch in tqdm(range(n_epochs), desc='Training'):
    idx = np.random.choice(t_data.shape[0], batch_size, replace=False)
    t_batch = torch.tensor(t_data[idx], dtype=torch.float32).view(-1, 1)  # Reshape t to [batch_size, 1]
    y1_batch = torch.tensor(data_scaled[0, idx], dtype=torch.float32).unsqueeze(1)
    y2_batch = torch.tensor(data_scaled[1, idx], dtype=torch.float32).unsqueeze(1)
    y3_batch = torch.tensor(data_scaled[2, idx], dtype=torch.float32).unsqueeze(1)
    y4_batch = torch.tensor(data_scaled[3, idx], dtype=torch.float32).unsqueeze(1)
    y5_batch = torch.tensor(data_scaled[4, idx], dtype=torch.float32).unsqueeze(1)
    y6_batch = torch.tensor(data_scaled[5, idx], dtype=torch.float32).unsqueeze(1)
    y7_batch = torch.tensor(data_scaled[6, idx], dtype=torch.float32).unsqueeze(1)
    y8_batch = torch.tensor(data_scaled[7, idx], dtype=torch.float32).unsqueeze(1)



    optimizer.zero_grad()
    y1_pred, y2_pred, y3_pred, y4_pred, y5_pred, y6_pred, y7_pred, y8_pred, r1_pred, r2_pred, r3_pred, r5_pred, r6_pred, r10_pred = main_model(t_batch)
    total_loss, data_loss, physics_loss, regularization_loss = custom_loss(
        main_model, t_batch, y1_batch, y2_batch, y3_batch, y4_batch, y5_batch, y6_batch, y7_batch, y8_batch
    )
    total_loss.backward() # Use the total_loss tensor for the backward pass
    optimizer.step()
    
    # Inside the training loop, append each loss at the end of each epoch
    total_losses.append(total_loss.item())
    data_losses.append(data_loss.item())
    physics_losses.append(physics_loss.item())
    regularization_losses.append(regularization_loss.item())
    
    
    # Calculate additional metrics
    combined_true = torch.cat([y1_batch, y2_batch, y3_batch, y4_batch, y5_batch, y6_batch, y7_batch, y8_batch], dim=1)
    combined_pred = torch.cat([y1_pred, y2_pred, y3_pred, y4_pred, y5_pred, y6_pred, y7_pred, y8_pred], dim=1)
    metrics = calculate_metrics(combined_true, combined_pred)

    # Calculate gradient norm
    grad_norm = sum(p.grad.data.norm(2).item() for p in main_model.parameters())


    # Plotting and saving figures at specified epochs
    if epoch % 200 == 0 and epoch != 0:  # adjust according to your preference for how often to plot
        # Forward pass with current parameters
        t_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)
        y1_pred, y2_pred, y3_pred, y4_pred, y5_pred, y6_pred, y7_pred, y8_pred, r1_pred, r2_pred, r3_pred, r5_pred, r6_pred, r10_pred = main_model.predict(t_tensor, yscale)
        
        # Convert tensors to numpy arrays for plotting
        t_np = t_tensor.numpy().flatten()
        y1_pred_np = y1_pred.numpy().flatten()
        y2_pred_np = y2_pred.numpy().flatten()
        y3_pred_np = y3_pred.numpy().flatten()
        y4_pred_np = y4_pred.numpy().flatten()
        y5_pred_np = y5_pred.numpy().flatten()
        y6_pred_np = y6_pred.numpy().flatten()
        y7_pred_np = y7_pred.numpy().flatten()
        y8_pred_np = y8_pred.numpy().flatten()
        
        
        r1_pred_np = r1_pred.numpy().flatten()
        r2_pred_np = r2_pred.numpy().flatten()
        r3_pred_np = r3_pred.numpy().flatten()
        r5_pred_np = r5_pred.numpy().flatten()
        r6_pred_np = r6_pred.numpy().flatten()
        r10_pred_np = r10_pred.numpy().flatten()
        
        
        # Filtering indices based on time range
        indices = np.where((t_np >= start_range) & (t_np <= end_range))[0]
        t_filtered = t_np[indices]


        # Actual data
        y1_data = sol.y[0, :]
        y2_data = sol.y[1, :]
        y3_data = sol.y[2, :]
        y4_data = sol.y[3, :]
        y5_data = sol.y[4, :]
        y6_data = sol.y[5, :]
        y7_data = sol.y[6, :]
        y8_data = sol.y[7, :]
        
        
         # Plotting
        plt.figure(figsize=(6, 5))
        plt.plot(t_np, y1_data, 'r', label="Actual y1")
        plt.plot(t_np, y1_pred_np, 'g--', label="Predicted y1")
        plt.xlabel('Time')
        plt.ylabel('Species y1')
        plt.title(f" HIRES - Species y1")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'y1_epoch_{epoch}.png'))
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.plot(t_np, y2_data, 'b', label="Actual y2")
        plt.plot(t_np, y2_pred_np, 'k--', label="Predicted y2")
        plt.xlabel('Time')
        plt.ylabel('Species y2')
        plt.title(f" HIRES - Species y2")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'y2_epoch_{epoch}.png'))
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.plot(t_np, y3_data, 'g', label="Actual y3")
        plt.plot(t_np, y3_pred_np, 'm--', label="Predicted y3")
        plt.xlabel('Time')
        plt.ylabel('Species y3')
        plt.title(f"HIRES - Species y3")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'y3_epoch_{epoch}.png'))
        plt.close()
        
        plt.figure(figsize=(12, 5))
        plt.plot(t_np, y4_data, 'm', label="Actual y4")
        plt.plot(t_np, y4_pred_np, 'b--', label="Predicted y4")
        plt.xlabel('Time')
        plt.ylabel('Species y4')
        plt.title(f"HIRES - Species y4")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'y4_epoch_{epoch}.png'))
        plt.close()
        
        plt.figure(figsize=(6, 5))
        plt.plot(t_np, y5_data, 'y', label="Actual y5")
        plt.plot(t_np, y5_pred_np, 'c--', label="Predicted y5")
        plt.xlabel('Time')
        plt.ylabel('Species y5')
        plt.title(f"HIRES - Species y5")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'y5_epoch_{epoch}.png'))
        plt.close()
        
        plt.figure(figsize=(6, 5))
        plt.plot(t_np, y6_data, 'm', label="Actual y6")
        plt.plot(t_np, y6_pred_np, 'y--', label="Predicted y6")
        plt.xlabel('Time')
        plt.ylabel('Species y6')
        plt.title(f"HIRES - Species y6")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'y6_epoch_{epoch}.png'))
        plt.close()
        
        plt.figure(figsize=(6, 5))
        plt.plot(t_np, y7_data, 'c', label="Actual y7")
        plt.plot(t_np, y7_pred_np, 'y--', label="Predicted y7")
        plt.xlabel('Time')
        plt.ylabel('Species y7')
        plt.title(f"HIRES - Species y7")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'y7_epoch_{epoch}.png'))
        plt.close()
        
        plt.figure(figsize=(6, 5))
        plt.plot(t_np, y8_data, 'k', label="Actual y8")
        plt.plot(t_np, y8_pred_np, 'y--', label="Predicted y8")
        plt.xlabel('Time')
        plt.ylabel('Species y8')
        plt.title(f"HIRES - Species y8")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'y8_epoch_{epoch}.png'))
        plt.close()
        
        
        plt.figure(figsize=(6, 5))
        plt.plot(t_filtered, r1_pred_np[indices]/10, 'r', label=f'Predicted r1 at Epoch {epoch}')
        plt.xlabel('Time')
        plt.ylabel('r1')
        plt.title('Prediction of Parameter r1 over Time')
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'r1_epoch_{epoch}.png'))
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.plot(t_filtered, r2_pred_np[indices], 'b', label=f'Predicted r2 at Epoch {epoch}')
        plt.xlabel('Time')
        plt.ylabel('r2')
        plt.title('Prediction of Parameter r2 over Time')
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'r2_epoch_{epoch}.png'))
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.plot(t_filtered, r3_pred_np[indices], 'g', label=f'Predicted r3 at Epoch {epoch}')
        plt.xlabel('Time')
        plt.ylabel('r3')
        plt.title('Prediction of Parameter r3 over Time')
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'r3_epoch_{epoch}.png'))
        plt.close()
        
        plt.figure(figsize=(6, 5))
        plt.plot(t_filtered, r5_pred_np[indices]*(-176), 'k', label=f'Predicted r5 at Epoch {epoch}')
        plt.xlabel('Time')
        plt.ylabel('r5')
        plt.title('Prediction of Parameter r5 over Time')
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'r5_epoch_{epoch}.png'))
        plt.close()
        
        plt.figure(figsize=(6, 5))
        plt.plot(t_filtered, r6_pred_np[indices]*(-234), 'm', label=f'Predicted r6 at Epoch {epoch}')
        plt.xlabel('Time')
        plt.ylabel('r6')
        plt.title('Prediction of Parameter r6 over Time')
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'r6_epoch_{epoch}.png'))
        plt.close()
        
        plt.figure(figsize=(6, 5))
        plt.plot(t_filtered, r10_pred_np[indices], 'c', label=f'Predicted r10 at Epoch {epoch}')
        plt.xlabel('Time')
        plt.ylabel('r10')
        plt.title('Prediction of Parameter r10 over Time')
        plt.grid(True)
        plt.savefig(os.path.join(figs_dir, f'r10_epoch_{epoch}.png'))
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
            
        # Save checkpoint every each epochs
        checkpoint_filename = f'main_model_checkpoint_{epoch}.pth'
        torch.save(main_model.state_dict(), checkpoint_filename)

# Save the final model after all epochs are completed
torch.save(main_model.state_dict(), 'final_trained_main_model.pth')

# After training, plot the losses
plt.figure(figsize=(6, 5))
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