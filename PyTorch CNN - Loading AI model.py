import torch, cv2
import torch.nn as nn
import torchvision.transforms as transforms

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        # Defining model architecture here
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 54 * 54, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        # Defining forward pass logic here
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

# Load the saved model
print("Loading model now")
model = CNN()
model.to("cpu")
model.load_state_dict(torch.load('/Users/Busaidi/Desktop/ML Models/cnn_july.pt', map_location='cpu'))
model.eval()
print("Finished loading model")

# Overlay model with live webcam feed
print("Opening AI on webcam now")
cap = cv2.VideoCapture(0)
while True:
    # Capture frame-to-frame
    ret, frame = cap.read()
    if not ret:
        break
    # Frame preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(frame)
    input_batch = input_tensor.unsqueeze(0)
    # Forward pass through the model
    with torch.no_grad():
        output = model(input_batch)
        print(output)
        print(output.type)
        print(output.shape)
    print(output)
    print(output.type)
    print(output.shape)

    # MISSING CODE HERE FOR PROCESSING OUTPUT AND GETTING BOUNDING BOXES AND CONFIDENCE SCORES
    # Define the confidence threshold for object detection
    confidence_threshold = 0.5
    # Obtain the predicted bounding boxes and confidence scores
    output = output.squeeze(0)
    pred_boxes = output['boxes'].detach().numpy()
    pred_scores = output['scores'].detach().numpy()
    # Apply non-maximum suppression to filter out overlapping bounding boxes
    indices = cv2.cnn.NMSBoxes(pred_boxes, pred_scores, confidence_threshold, 0.4)
    # Filter out the bounding boxes and confidence scores based on the indices
    boxes = pred_boxes[indices[:, 0]]
    scores = pred_scores[indices[:, 0]]

    # Overlay bounding boxes and confidence scores on the frame
    for box, confidence in zip(boxes, scores):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # Display the frame
    cv2.imshow('Webcam', frame)
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ## Frame preprocessing
    #frame = cv2.resize(frame, (224, 224))
    #frame = transform(frame).unsqueeze(0)
    #output = model(frame)
    #prediction = torch.argmax(output).item()
    #print('Prediction:', prediction)

cap.release()
cv2.destroyAllWindows()