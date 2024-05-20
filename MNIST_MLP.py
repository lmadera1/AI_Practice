import numpy
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

#Number of epochs to train for
num_epochs = 15

#Used to normalize and rescale data from MNIST download
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])


#Load MNIST data from torchvision

#Train Data
train_set = datasets.MNIST('train_set', download=True, train=True, transform=transform)
#Batch test data into groups of 64
trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
#Iterator for batch of images and labels
train_iter = iter(trainloader)

#Test Data
test_set = datasets.MNIST('test_set', download=True, train=False, transform=transform)
#Batch test data into groups of 64
testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
#Iterator for batch of images and labels
test_iter = iter(testloader)

#Model definition
model = nn.Sequential(
    nn.Linear(784, 500),
    nn.ReLU(),
    nn.Linear(500, 100),
    nn.ReLU(),
    nn.Linear(100, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.LogSoftmax(dim=1)
)

#Loss Function
loss_fn = nn.CrossEntropyLoss()

#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Train
for images, labels in train_iter:
    images = images.view(images.shape[0], -1)
    y_pred = model(images)
    loss = loss_fn(y_pred, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#Evaluate
total_imgs = 0
total_correct = 0
for images, labels in test_iter:
    for i in range(len(images)):
        total_imgs += 1
        with torch.no_grad():
            logits = model(images[i].view(1, 784))
        probs = torch.exp(logits)
        probs = list(probs.numpy()[0])
        index = probs.index(max(probs))
        if (index == labels[i]):
            total_correct += 1
        

print(f"Accuracy: {total_correct / total_imgs * 100}%")