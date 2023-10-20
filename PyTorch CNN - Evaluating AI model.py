import cv2, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_score, confusion_matrix
from torchvision import transforms
from PIL import Image
import numpy as np

# Define the CNN architecture (same as before)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 54 * 54, 128)
        #self.fc2 = nn.Linear(128, 2)
        self.fc2 = nn.Linear(128, 3) #ChatGPT recommended last output layer include no. of classes of training dataset
    def forward(self, x):
        x = self.conv1(x)
        #x = nn.ReLU()(x) #ChatGPT recommends removing ReLU
        x = self.pool(x)
        x = self.conv2(x)
        #x = nn.ReLU()(x) #ChatGPT recommends removing ReLU
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        #x = nn.ReLU()(x) #ChatGPT recommends removing ReLU
        x = self.fc2(x)
        return x

# Load the trained model
model_path = '/Users/Busaidi/Desktop/ML Models/cnn_july.pt'
model = CNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load validation dataset (val_dataset)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Set batch_size accordingly

true_labels = []
predicted_labels = []
predicted_probabilities = []

# Generate predictions
with torch.no_grad():
    for inputs, labels in val_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predicted_prob = torch.softmax(outputs, dim=1)[:, 1]  # Probability for positive class

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())
        predicted_probabilities.extend(predicted_prob.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
roc_auc = roc_auc_score(true_labels, predicted_probabilities)
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Print or display metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC-AUC:", roc_auc)
print("Confusion Matrix:")
print(conf_matrix)
