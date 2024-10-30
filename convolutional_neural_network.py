import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, num_filters=16, kernel_size=3, padding=1):
        """
        Initializes the CNN.
        Parameters:
            num_classes (int): Number of output classes.
            num_filters (int): Number of filters in the first convolutional layer. 
                The second layer will have 2x this number.
            kernel_size (int): Size of the convolving kernel.
            padding (int): Zero-padding added to both sides of the input.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size, padding=padding)
        
        #dimensions of our filtered img after CONV1
        chans, width, height = num_filters, 32, 32
        res = (width - kernel_size + 2 * padding) // 1 + 1
        
        #dimensions of our filtered img after POOL1
        chans, width, height = num_filters, res, res
        res = (width - kernel_size + 2 * padding) // 1 + 1
        
        #dimensions of our filtered img after CONV2
        chans, width, height = num_filters * 2, res, res
        res = (width - kernel_size + 2 * padding) // 1 + 1
        
        #dimensions after final pooling layer
        chans, width, height = num_filters * 2, res, res
        res = (width - kernel_size + 2 * padding) // 1 + 1
        
        #need to calculate the number of input features for the fully connected layer
        #the number of features is going to be the number of channels * width * height
        #of the last convolutional layer
        self.fc1 = nn.Linear(2 * num_filters * res * res, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        
    def forward(self, x):
        """
        Performs forward pass of the input.
        Parameters:
            x (Tensor): Input tensor.
        Returns:
            out (Tensor): Output tensor.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define transformations for the train set
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Define transformations for the validation set
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transforms)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, lr=0.01, momentum=0.9):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    model.train()  # Set the model to training mode
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def validate(model, val_loader):
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print('Validation accuracy: {:.2f}%'.format((accuracy) * 100))
    return accuracy

def visulization(accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(accuracies)
    plt.title('Validation accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    
def task1(num_filters=16):
    print('Task 1 with different number of filters')
    num_filters_list = [16, 32] # todo: try different number of filters
    for num_filters in num_filters_list:
        model = SimpleCNN(num_filters=num_filters) 
        model = model.to(device)
        accuracies = []
        for epoch in range(10):  # Loop over the dataset multiple times
            print('Epoch {}/{}'.format(epoch+1, 10))
            train(model, train_loader)
            accuracy = validate(model, val_loader)
            accuracies.append(accuracy)
        # visulization(accuracies) # You can use this function to visulize the trend of accuracy change
        
        
def task2():
    print('Task 2 with different kernel sizes')
    kernel_sizes = [3, 5] # todo: try different kernel sizes
    for kernel_size in kernel_sizes:
        model = SimpleCNN(kernel_size=kernel_size) 
        model = model.to(device)
        accuracies = []
        for epoch in range(10):  # Loop over the dataset multiple times
            print('Epoch {}/{}'.format(epoch+1, 10))
            train(model, train_loader)
            accuracy = validate(model, val_loader)
            accuracies.append(accuracy)
        # visulization(accuracies) # You can use this function to visulize the trend of accuracy change
        
def task3():
    print('Task 3 with different padding')
    padding_list = [0, 1, 2] # todo: try different padding
    for padding in padding_list:
        model = SimpleCNN(padding=padding, kernel_size=4) 
        model = model.to(device)
        accuracies = []
        for epoch in range(10):  # Loop over the dataset multiple times
            print('Epoch {}/{}'.format(epoch+1, 10))
            train(model, train_loader)
            accuracy = validate(model, val_loader)
            accuracies.append(accuracy)
        # visulization(accuracies) # You can use this function to visulize the trend of accuracy change

if __name__ == "__main__":
    task1()
    task2()
    task3()
