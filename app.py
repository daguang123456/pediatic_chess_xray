import streamlit as st
# from img_classification import teachable_machine_classification
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

# def pil_loader(path):
#     with open(path, 'rb') as f:
#         with Image.open(f) as img:
#             return img.convert('RGB')

# Define model architecture
class PneumoniaClassifier(nn.Module):
    def __init__(self):
        super(PneumoniaClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(16, 30, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(30)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(30, 60, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(60)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(60 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)
        
        x = x.view(-1, 60 * 28 * 28)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Instantiate model and optimizer
model = PneumoniaClassifier()

# Load the trained model
model = torch.load("pediatic_pneumonia_classifier2.pt")

# Define the data transform for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


st.title("儿科肺炎 X 线图像分类")
# st.header("脑肿瘤MRI分类示例")
st.text("上传儿科肺MRI的图像，将图像分类为肺炎或无肺炎")

uploaded_file = st.file_uploader("选择儿科肺MRI的图像...", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='上传了核磁共振成像。', use_column_width=True)
    st.write("")
    st.write("分类中...")

    # Load the input image and apply the transform
    # image1 = Image.open(uploaded_file)
    image1 = Image.open(uploaded_file).convert('RGB')
    image1 = transform(image1)
    image1 = image1.unsqueeze(0)

    # Make a prediction on the input image
    with torch.no_grad():
        outputs = model(image1)
        print("outputs", outputs)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted)
        
    # Print the predicted class
    if predicted.item() == 0:
        st.write("图像被归类为正常")
    else:
        st.write("图像被归类为肺炎")



