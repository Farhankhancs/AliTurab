import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
file_path = r"dataset.csv"
df = pd.read_csv(file_path)

# Separate features and target
X = df.iloc[:, :-1].values  # All columns except the last
y = df.iloc[:, -1].values  # Target column (0 or 1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  # BCELoss requires float labels

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define Transformer Model
class TransformerBinaryClassifier(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=4, num_layers=2):
        super(TransformerBinaryClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)  # Project input to d_model
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=128),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, 1)  # Output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)  # Linear projection
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return self.sigmoid(x)


# Initialize Model, Loss, and Optimizer
input_dim = X.shape[1]
model = TransformerBinaryClassifier(input_dim)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")


# Evaluation Function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X).squeeze()
            preds = (outputs > 0.5).long()  # Convert probabilities to binary labels
            all_preds.extend(preds.numpy())
            all_labels.extend(batch_y.numpy())

    # Compute classification metrics for each class
    class_report = classification_report(all_labels, all_preds, target_names=["Left Move (0)", "Right Move (1)"],
                                         output_dict=True)

    # Convert to DataFrame for better formatting
    results_table = pd.DataFrame(class_report).transpose()

    return results_table


# Get classification results
results_table = evaluate_model(model, test_loader)
print("\nClassification Report:\n", results_table)


# Inference Function
def predict(model, new_data):
    model.eval()
    with torch.no_grad():
        data_tensor = torch.tensor(new_data, dtype=torch.float32)
        prediction = model(data_tensor).squeeze().round().item()
    return int(prediction)


# Example usage
sample_input = X[0].reshape(1, -1)  # Reshape for single prediction
print("\nPredicted Move:", predict(model, sample_input))  # 0 (left) or 1 (right)
