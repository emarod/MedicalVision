import medmnist
from medmnist import INFO, Evaluator
from medmnist import OCTMNIST
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from model import Net

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

train_dataset = OCTMNIST(split="train", download=True, transform=data_transform)
val_dataset = OCTMNIST(split="val", download=True, transform=data_transform)

BATCH_SIZE = 128

train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

info = INFO['octmnist']
n_channels = info['n_channels']     # Será 1 porque es escala de grises
n_classes = len(info['label'])      # OCTMNIST tiene 4 clases (Clasificación Multiclase)
print(f"Canales: {n_channels}, Clases a predecir: {n_classes}")

lr = 0.01
model = Net(in_channels=n_channels, num_classes=n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)


# train
print("training...")
NUM_EPOCHS = 3
for epoch in range(NUM_EPOCHS):
  train_correct = 0
  train_total = 0
  test_correct = 0
  test_total = 0

  model.train()
  for inputs, targets in tqdm(train_loader):
    # forward + backward + optimize
    optimizer.zero_grad()
    outputs = model(inputs)
    
    targets = targets.squeeze().long()
    loss = criterion(outputs, targets)
    
    loss.backward()
    optimizer.step()


MODEL_PATH = "octmnist_cnn_weights.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"Pesos guardados en: {MODEL_PATH}")