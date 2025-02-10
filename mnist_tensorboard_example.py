import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# 🚀 Initialize TensorBoard
writer = SummaryWriter("runs/MNIST")

# 🔹 Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])  # Define transformations
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)  # Load training dataset
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)  # Load test dataset

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)  # DataLoader for training data
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)  # DataLoader for test data

# 🔹 Define a simple CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Conv layer 1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Conv layer 2
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Fully connected layer 1
        self.fc2 = nn.Linear(128, 10)  # Output layer

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Apply ReLU to conv1
        x = torch.max_pool2d(x, 2)  # Apply max pooling
        x = torch.relu(self.conv2(x))  # Apply ReLU to conv2
        x = torch.max_pool2d(x, 2)  # Apply max pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))  # Apply ReLU to fc1
        x = self.fc2(x)  # Output layer (no activation)
        return x

# 🔹 Model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model = CNN().to(device)  # Move model to device
criterion = nn.CrossEntropyLoss()  # Define loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Define optimizer

# 🔹 Log model graph to TensorBoard
dummy_input = torch.randn(1, 1, 28, 28).to(device)  # Create a dummy input
writer.add_graph(model, dummy_input)  # Add model graph to TensorBoard

# 🔹 Training loop
num_epochs = 5  # Number of epochs
for epoch in range(num_epochs):
    total_loss = 0  # Initialize total loss
    correct = 0  # Initialize correct predictions
    total = 0  # Initialize total samples

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to device

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        total_loss += loss.item()  # Accumulate loss
        _, predicted = torch.max(outputs, 1)  # Get predictions
        correct += (predicted == labels).sum().item()  # Count correct predictions
        total += labels.size(0)  # Count total samples

    avg_loss = total_loss / len(train_loader)  # Compute average loss
    accuracy = correct / total * 100  # Compute accuracy

    # ✅ Log loss and accuracy to TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)  # Log training loss
    writer.add_scalar('Accuracy/train', accuracy, epoch)  # Log training accuracy

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')  # Print epoch stats

# 🔹 Log test images
test_images, test_labels = next(iter(test_loader))  # Get a batch of test images
grid = torchvision.utils.make_grid(test_images[:16], nrow=4, normalize=True)  # Create a grid of images
writer.add_image('MNIST Sample Images', grid)  # Log images to TensorBoard

# ✅ Close TensorBoard writer
writer.close()  # Close the writer
