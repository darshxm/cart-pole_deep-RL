import gymnasium as gym
import torch
import torch.nn as nn
import os

# Define the QNetwork model architecture
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)   # Use LayerNorm instead of BatchNorm
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Check if a GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model weights onto the GPU if available
def load_model(path, state_dim, action_dim):
    model = QNetwork(state_dim, action_dim).to(device)  # Move model to GPU
    model.load_state_dict(torch.load(path, map_location=device))  # Map to the chosen device
    model.eval()  # Set to evaluation mode
    return model

# Initialize environment and model
env = gym.make('CartPole-v1', render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Load the trained policy model
model_path = os.getcwd() + "\\cartpole_policy_net_v2.pth"  # Update this if the path is different
policy_net = load_model(model_path, state_dim, action_dim)

# Run an episode with the trained model
state = env.reset()[0]
done = False
total_reward = 0

while not done:
    env.render()  # Display the environment
    
    # Convert state to tensor, move to GPU if available
    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
    
    # Get action from policy network
    with torch.no_grad():
        action = policy_net(state_tensor).argmax().item()  # Choose the action with the highest Q-value
    
    # Take the chosen action in the environment
    state, reward, done, _, _ = env.step(action)
    total_reward += reward

print(f"Total Reward from the Episode: {total_reward}")

env.close()
