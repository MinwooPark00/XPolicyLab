import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset, load_from_disk

from human_plan.utils.mano.forward import (
    mano_forward_retarget,
    # mano_forward_retarget_isaaclab
)

from human_plan.dataset_preprocessing.utils.mano_utils import (
    mano_to_inspire_mapping
)

# Dataset Class for Hugging Face Dataset
class HandDataset(Dataset):
    def __init__(self, hf_dataset):
        self.left_mano_parameters = torch.tensor(hf_dataset["current_left_mano_parameters"], dtype=torch.float32)[:, :]
        self.right_mano_parameters = torch.tensor(hf_dataset["current_right_mano_parameters"], dtype=torch.float32)[:, :]
        print(self.left_mano_parameters.shape)
        print(self.right_mano_parameters.shape)
        self.left_key_points = mano_forward_retarget(
            self.left_mano_parameters, is_right=False
        )[:, mano_to_inspire_mapping][:, 1:].detach()

        self.right_key_points = mano_forward_retarget(
            self.right_mano_parameters, is_right=True
        )[:, mano_to_inspire_mapping][:, 1:].detach()

        print(self.left_key_points.shape)
        print(self.right_key_points.shape)

        self.left_key_points = self.left_key_points.reshape(-1, 15)
        self.right_key_points = self.right_key_points.reshape(-1, 15)

        self.left_hand_dof_index = torch.tensor([26, 36, 27, 37, 28, 38, 29, 39, 30, 40, 46, 48])
        self.right_hand_dof_index = torch.tensor([31, 41, 32, 42, 33, 43, 34, 44, 35, 45, 47, 49])
        self.actions = torch.tensor(hf_dataset["action"], dtype=torch.float32)
        print(self.actions.shape)
        self.actions = torch.concat([
            self.actions[:, self.left_hand_dof_index],
            self.actions[:, self.right_hand_dof_index],
        ], dim=-1)
        print(self.actions.shape)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        # Concatenate left and right mano parameters as input
        inputs = torch.cat([self.left_key_points[idx], self.right_key_points[idx]])
        targets = self.actions[idx]  # Targets are the actions
        return inputs, targets

# Define the Network
class HandActuationNet(nn.Module):
    def __init__(self, input_dim=30, output_dim=12):
        super(HandActuationNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Training Function
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        count = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            # inputs 
            # print(inputs.shape, targets.shape)
            outputs = model(inputs)
            criterion = nn.MSELoss()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # print(loss)
            # if count % 50 == 0:
            #     print(f"Epoch {epoch}, {count}, {loss.item()}")
            count += 1
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    print("Training Complete!")

# Inference Function
def infer(model, left_mano, right_mano):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.cat([
            torch.tensor(left_mano, dtype=torch.float32), 
            torch.tensor(right_mano, dtype=torch.float32)
        ], dim=-1).reshape(-1, 30)
        assert left_mano.shape[-1] == 15
        assert right_mano.shape[-1] == 15
        predictions = model(input_tensor)
    return predictions.squeeze().detach().cpu().numpy()

# Main Script
if __name__ == "__main__":
    # Load the Hugging Face dataset
    hf_dataset = load_from_disk(
        "/home/xiaolwang/new_home/datasets/Manipulation-Summer/otv_isaaclab_hf_v3/HF_hand_V1_train"
    )  # Update with the actual dataset path or name

    # Use the train split for training and validation
    dataset = HandDataset(hf_dataset)

    # Split into train and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

    # Initialize Model, Train, and Save
    model = HandActuationNet(input_dim=30, output_dim=12 * 2)  # 15 dims for each hand
    train_model(model, train_loader, val_loader, epochs=2000, lr=0.001)
    torch.save(model.state_dict(), "hand_actuation_net.pth")

    # Load Model and Perform Inference
    model.load_state_dict(torch.load("hand_actuation_net.pth"))

    # Test Inference on a new sample
    test_left_mano = [0.1] * 15  # Replace with actual test data
    test_right_mano = [0.2] * 15  # Replace with actual test data
    prediction = infer(model, test_left_mano, test_right_mano)
    print("Prediction:", prediction)
