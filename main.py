import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from data import MNISTDataset
from utils import collate_fn 
from LeNet import LeNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#---------------------------------------------#
# Load MNIST Dataset
train = MNISTDataset(
    image_path=r'\mnist\train-images.idx3-ubyte',
    label_path=r'\mnist\train-labels.idx1-ubyte'
)

test = MNISTDataset(
    image_path=r'\mnist\t10k-images.idx3-ubyte',
    label_path=r'\mnist\t10k-labels.idx1-ubyte'
)

#---------------------------------------------#
# Split train into train and dev
train_size = int(0.9 * len(train))
dev_size = len(train) - train_size

train_dataset, dev_dataset = random_split(train, [train_size, dev_size])

#---------------------------------------------#
# Create DataLoaders
train_dataloader = DataLoader(
    dataset=train_dataset, 
    batch_size=64, 
    shuffle=True, 
    collate_fn=collate_fn
)

dev_dataloader = DataLoader(
    dataset=dev_dataset,
    batch_size=1, 
    shuffle=False, 
    collate_fn=collate_fn
)

test_dataloader = DataLoader(
    dataset=test,
    batch_size=1, 
    shuffle=True, 
    collate_fn=collate_fn
)

#---------------------------------------------#

model = LeNet()


model.to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)


#---------------------------------------------#
# Training loop
# Cài early Stopping
patience = 3   
best_loss = np.inf   
trigger_times = 0   
early_stop = False 
previous_model_path = 'previous_model.pth' 

for epoch in range(10):  # Bạn có thể đặt số epoch lớn hơn nếu dùng early stopping
    print(f'Epoch {epoch + 1}')
    
    # Training phase
    model.train()  # Set model to training mode
    for images, labels in tqdm(train_dataloader, desc='Train'):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation phase on dev set
    model.eval()  
    dev_loss = 0  
    dev_predicted = []
    dev_gts = []
    
    with torch.no_grad():
        for images, labels in tqdm(dev_dataloader, desc='Dev'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            dev_loss += loss.item()
            
            dev_predicted.append(outputs.argmax(dim=1).item())
            dev_gts.append(labels.item())
    
    dev_loss /= len(dev_dataloader)
    
    dev_predicted = np.array(dev_predicted)
    dev_gts = np.array(dev_gts)
    
    dev_precision = precision_score(dev_gts, dev_predicted, average='weighted', zero_division=0)
    dev_recall = recall_score(dev_gts, dev_predicted, average='weighted')
    dev_f1 = f1_score(dev_gts, dev_predicted, average='weighted')
    dev_accuracy = accuracy_score(dev_gts, dev_predicted)
    
    print(f'Dev Loss: {dev_loss:.4f}')
    print(f'Dev Precision: {dev_precision:.4f}')
    print(f'Dev Recall: {dev_recall:.4f}')
    print(f'Dev F1 Score: {dev_f1:.4f}')
    print(f'Dev Accuracy: {dev_accuracy:.4f}')
    
    if dev_loss < best_loss:
        best_loss = dev_loss
        trigger_times = 0
    else:
        trigger_times += 1
        print(f'No improvement in validation loss. Patience: {trigger_times}/{patience}')
        
        if trigger_times >= patience:
            print("Early stopping triggered!")
            early_stop = True
            break

    if early_stop:
        break

#---------------------------------------------#
# Testing loop
model.eval()
test_predicted = []
test_gts = []

with torch.no_grad():
    for images, labels in tqdm(test_dataloader, desc='Test'):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        test_predicted.append(outputs.argmax(dim=1).item())
        test_gts.append(labels.item())


test_predicted = np.array(test_predicted)
test_gts = np.array(test_gts)

test_precision = precision_score(test_gts, test_predicted, average='weighted', zero_division=0)
test_recall = recall_score(test_gts, test_predicted, average='weighted')
test_f1 = f1_score(test_gts, test_predicted, average='weighted')
test_accuracy = accuracy_score(test_gts, test_predicted)

print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test F1 Score: {test_f1:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
