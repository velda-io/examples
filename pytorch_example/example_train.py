#!/usr/bin/env python 

# This is a simple training script with PyTorch for a CNN model
# To train with GPU, run:
#    vrun -P gpu-t4-1 ./example_train.py
# 
print("Starting the job")
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Simple CNN model optimized for T4
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Reduced channels to fit T4 memory
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Adaptive pooling to handle any input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Data loading with optimizations for T4
def get_data_loaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                           num_workers=2, pin_memory=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
    
    return trainloader, testloader

# Training function
def train_model(model, trainloader, testloader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    model.train()
    
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}, '
                      f'Acc: {100. * correct / total:.2f}%')
                running_loss = 0.0
        
        scheduler.step()
        
        # Validation
        val_acc = evaluate_model(model, testloader)
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s')
        print(f'Validation Accuracy: {val_acc:.2f}%\n')

# Evaluation function
def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    model.train()
    return 100. * correct / total

# Main training script
def main():
    # Initialize model
    model = SimpleCNN(num_classes=10).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model has {total_params:,} parameters')
    
    # Get data loaders (smaller batch size for T4)
    trainloader, testloader = get_data_loaders(batch_size=128)
    
    # Train the model
    train_model(model, trainloader, testloader, num_epochs=10)
    
    # Final evaluation
    final_acc = evaluate_model(model, testloader)
    print(f'Final Test Accuracy: {final_acc:.2f}%')
    
    # Save the model
    torch.save(model.state_dict(), 'simple_cnn_cifar10.pth')
    print('Model saved as simple_cnn_cifar10.pth')

if __name__ == '__main__':
    main()