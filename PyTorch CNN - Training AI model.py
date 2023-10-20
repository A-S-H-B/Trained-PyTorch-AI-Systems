import matplotlib.pyplot as plt
import torch, cv2
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#Successfully trained PyTorch CNN Image Classifier

# Define the CNN architecture
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

#To define mean and std for normalisation
mean = (0.5, 0.5, 0.5)
std = (0.2, 0.2, 0.2)

# Define the data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.9),
    #transforms.Resize(224),
    transforms.RandomRotation(60),
    #transforms.RandomCrop(),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,))
    #transforms.RandomPerspective(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

# Load the dataset
dataset = datasets.ImageFolder('/Users/Busaidi/Desktop/ML Balanced', transform=transform) #Gives IndexError: Target 2 is out of bounds
#dataset = datasets.ImageFolder("/Users/Busaidi/Downloads/FastAI_ImageNet_v2/train") #Gives batch PIL Image error
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

print("Number of classes in train_dataset:", len(train_dataset.dataset.classes))
print("Class labels in train_dataset:", train_dataset.dataset.classes)

# Create dataloaders for training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

def show(dataset):
    classes = ("Ammar", "Father", "Amroo")
    dataiter = iter(dataset)
    images, labels = dataiter.next()
    fig, axes = plt.subplots(figsize=(10, 4), ncols=5)
    for i in range(5):
        ax = axes[i]
        ax.imshow(images[i].permute(1, 2, 0))
        ax.title.set_text(' '.join('%5s' % classes[labels[i]]))
    plt.show()

#show(val_dataset)

# Define the model and training parameters
model = CNN().to(device="cpu")
model.eval()
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))
#criterion = RMSELoss()
criterion = nn.CrossEntropyLoss() #Better for classification which is computer vision
optimizer = optim.Adam(model.parameters(), lr=0.0001) #Opimisers (eg gradient descent, ADAM)

# Train the model on the training dataset
epochs = 11 #Affects overfitting/underfitting
for epoch in range(epochs):
    print("This epoch number is: " + str(epoch))
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device="cpu")
        target = target.to(torch.long)
        output = model(data) #Output are predictions
        #output = output.flatten()
        # Backdrop and Adam optimisation
        #print("Tensor of Output:")
        #print(output.type())
        #print(output.size())
        #print(output)
        #target = target.unsqueeze(1).float() #THIS LINE MADE THE CODE MF-ING WORK !!
        #print("Tensor of Target:")
        #print(target.type())
        #print(target.size())
        #print(target)
        loss = criterion(output, target)
        #loss = (output - target) ** 2
        loss.backward()
        optimizer.step()
    print("Finished one epoch of training model")
    print("")

# Save the trained model
torch.save(model.state_dict(), '/Users/Busaidi/Desktop/ML Models/cnn_july.pt')
print("Finished saving model")

# Load the saved model
print("Loading model now")
model = CNN()
model.load_state_dict(torch.load('/Users/Busaidi/Desktop/ML Models/cnn_july.pt'))
model.eval()
print("Finished loading model")

# Overlay model with live webcam feed
print("Opening AI on webcam now")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Preprocess frame
    frame = cv2.resize(frame, (224, 224))
    frame = transform(frame).unsqueeze(0)
    output = model(frame)
    prediction = torch.argmax(output).item()
    print('Prediction:', prediction)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()