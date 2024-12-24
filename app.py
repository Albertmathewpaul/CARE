import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Streamlit App Title
st.title("CARE: Comprehensive AI Retinal Expert")

# File Uploader for Fundus Image
uploaded_file = st.file_uploader("Upload an Eye Fundus Image", type=["jpg", "png", "jpeg"])

# Define the Model Architecture
class ClassCNN(nn.Module):
    def __init__(self):
        super(ClassCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 3)  # 3 classes: Mild, Medium, Severe

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the Model and Weights
model = ClassCNN()
model.load_state_dict(torch.load("classification_model.pth", map_location=torch.device('cpu')))
model.eval()

# Image Preprocessing Function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),         # Convert to Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize (standard ImageNet mean/std)
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Perform Classification
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")  # Open and ensure it's RGB
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        with st.spinner("Classifying..."):
            # Preprocess the image
            input_tensor = preprocess_image(image)

            # Predict with the model
            output = model(input_tensor)
            _, predicted_class = torch.max(output, 1)

            # Map class index to human-readable labels
            class_labels = ["Cataract", "Normal", "glaucoma","diabetic retinopathy"]
            result = class_labels[predicted_class.item()]

            # Display the result
            st.success(f"Result: {result}")
