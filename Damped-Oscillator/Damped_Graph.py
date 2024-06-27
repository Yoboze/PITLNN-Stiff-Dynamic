import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score

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

# Define the directory for figures and checkpoints
dir_path = os.getcwd()
figs_dir = os.path.join(dir_path, "figs")
checkpoint_dir = os.path.join(dir_path)
os.makedirs(figs_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Solve the ODE system with specific parameters to generate actual data
def ode_system(t, y, c, k):
    z1, z2 = y
    return [z2, -c*z1 - k*z2]

t_span = [0, 10]
y_init = [1.0, 1.0]
t_eval = np.linspace(t_span[0], t_span[1], 1000)
sol = solve_ivp(lambda t, y: ode_system(t, y, 1001, 1000), t_span, y_init, t_eval=t_eval, method='BDF')

# Prepare tensor for model prediction
t_tensor = torch.tensor(sol.t, dtype=torch.float32).view(-1, 1)

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

# Iterate over each checkpoint, load the model, predict, and plot
for epoch in range(1000, 50000, 1000):
    checkpoint_filename = f'main_model_checkpoint_{epoch}.pth'
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint_filename)))

    with torch.no_grad():
        z1_pred, z2_pred, c_pred, k_pred = model(t_tensor)

    # Combine predictions and actual data for metrics calculation
    combined_true = torch.tensor(np.vstack((sol.y[0], sol.y[1])).T, dtype=torch.float32)
    combined_pred = torch.cat((z1_pred, z2_pred), dim=1)
    metrics = calculate_metrics(combined_true, combined_pred)

    # Print metrics
    print(f"Metrics at Epoch {epoch}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.7f}")

    # Plotting
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.plot(sol.t, sol.y[0], 'r', label='Actual z1')
    plt.plot(sol.t, z1_pred.numpy().flatten(), 'b--', label='PITLNN z1')
    plt.title(f'Comparison of Actual and PITLNN z1')
    plt.xlabel('Time')
    plt.ylabel('z1')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(sol.t, sol.y[1], 'g', label='Actual z2')
    plt.plot(sol.t, z2_pred.numpy().flatten(), 'k--', label='PITLNN z2')
    plt.title(f'Comparison of Actual and PITLNN z2')
    plt.xlabel('Time')
    plt.ylabel('z2')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(sol.t, c_pred.numpy().flatten(), "r", label='C(t)')
    plt.xlabel('Time')
    plt.ylabel('Parameter c')
    plt.title(f'Time-Varying Parameter C')
    plt.grid(True)
    plt.legend()


    plt.subplot(2, 2, 4)
    plt.plot(sol.t, k_pred.numpy().flatten(), "k", label='K(t)')
    plt.xlabel('Time')
    plt.ylabel('Parameter k')
    plt.title(f'Time-Varying Parameter K')
    plt.grid(True)
    plt.legend()


    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, f'Complete_Epoch_{epoch}.png'))
    plt.close()

# Evaluate the final model
final_model_filename = 'final_trained_main_model.pth'
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, final_model_filename)))

with torch.no_grad():
    z1_pred, z2_pred, c_pred, k_pred = model(t_tensor)

# Combine predictions and actual data for final metrics calculation
combined_true = torch.tensor(np.vstack((sol.y[0], sol.y[1])).T, dtype=torch.float32)
combined_pred = torch.cat((z1_pred, z2_pred), dim=1)
final_metrics = calculate_metrics(combined_true, combined_pred)

# Print final metrics
print("Final Model Metrics:")
for metric, value in final_metrics.items():
    print(f"{metric}: {value:.7f}")

# Plotting for final model
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.plot(sol.t, sol.y[0], 'r', label='Actual z1')
plt.plot(sol.t, z1_pred.numpy().flatten(), 'b--', label='PITLNN z1')
plt.title('Comparison of Actual and Predicted z1 for Final Model')
plt.xlabel('Time')
plt.ylabel('z1')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(sol.t, sol.y[1], 'g', label='Actual z2')
plt.plot(sol.t, z2_pred.numpy().flatten(), 'k--', label='PITLNN z2')
plt.title('Comparison of Actual and Predicted z2 for Final Model')
plt.xlabel('Time')
plt.ylabel('z2')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(sol.t, c_pred.numpy().flatten(), "r")
plt.xlabel('Time')
plt.ylabel('Scaled Parameter c')
plt.title(f'Scaled c Predictions for Final Model')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(sol.t, k_pred.numpy().flatten(), "k")
plt.xlabel('Time')
plt.ylabel('Scaled Parameter k')
plt.title(f'Scaled k Predictions for Final Model')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(figs_dir, 'Comparison_Final_Model.png'))
plt.show()
