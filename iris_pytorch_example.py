import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchviz import make_dot  # Import torchviz

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for batching
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Define the model (Simple Fully Connected Neural Network)
class IrisNN(nn.Module):
    def __init__(self):
        super(IrisNN, self).__init__()
        self.fc1 = nn.Linear(4, 64)  # 4 input features (Iris dataset has 4 features)
        self.relu = nn.ReLU() # ReLU activation function
        self.fc2 = nn.Linear(64, 3)  # 3 output classes (Iris dataset has 3 classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x) # Apply ReLU activation
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = IrisNN()
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001) # oh, model parameters ARE tensors, but with requires_grad=True

# Single forward pass for visualization
sample_input = X_train_tensor[:1]  # A single data point for the graph
sample_output = model(sample_input)  # Forward pass

# Visualize the computational graph
graph = make_dot(sample_output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
graph.render("iris_model_graph", format="png", cleanup=True)  # Save graph as PNG

# Training the model
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad() # reset the gradients
        outputs = model(inputs)
        loss = criterion(outputs, torch.argmax(labels, dim=1))  # CrossEntropyLoss expects class indices
        loss.backward() # TBackward (Tensor Backward)
        optimizer.step() # Accumulate the gradients
        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')

# Testing the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        # returns an array of classes with highest probability
        # Classes in THIS example are 0, 1, 2
        # 0 = Setosa, 1 = Versicolor, 2 = Virginica 
        # the size of the array is the same as the batch size
        predicted = torch.argmax(outputs, dim=1)  # Get the iris class (species) with highest probability
        true_labels = torch.argmax(labels, dim=1)  # Get the true iris class (species)
        correct += (predicted == true_labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')
