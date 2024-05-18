from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import streamlit as st
def set_background_image():
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("https://img.upanh.tv/2024/05/18/111e99ded6dd68631.jpg");
        background-size: cover
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

hide_streamlit_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    header {visibility: hidden;}
                    </style>
                    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
set_background_image()
st.set_page_config(layout="wide")
st.header("Sinh trắc học vân tay")
st.write(
    "Sinh trắc học vân tay là nghiên cứu về các đặc điểm vân tay để xác định tính cách và tương lai của một người.")
# You can add content related to palmistry here

classes = ['Hình cung', 'Vòng tròn hướng tâm', 'Vòng lặp Ulnar', 'Vòm lều', 'Vòng xoáy']


class FingerprintCNN(nn.Module):
    def __init__(self):
        super(FingerprintCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, len(classes))

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        x = self.pool(nn.functional.relu(self.conv5(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_fin = FingerprintCNN()
model_fin.load_state_dict(torch.load(r'fingerprint.pth', map_location=device))
model_fin.eval()



# Hàm để đọc nội dung từ tệp văn bản
def read_file_content(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


# Function to predict label for input image
def predict_label(img):
    # img = cv2.imread(image_path)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (128, 128))  # Resize the image to 128x128
    img.reshape(-1, 128, 128, 3)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model_fin(img)
        _, predicted = torch.max(outputs, 1)
    predicted_class = classes[predicted.item()]
    return predicted_class


uploaded_file = st.file_uploader("Nhập ảnh vân tay của bạn", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', width=200, use_column_width=False)

    # Start prediction when "Start" button is clicked
    if st.button('Start'):
        # Save the uploaded file locally
        # with open(uploaded_file.name, "wb") as f:
        #     f.write(uploaded_file.getbuffer())
        image = Image.open(uploaded_file)
        # Predict label
        predicted_label = predict_label(image)

        # Display prediction result

        filename = f"Label_sinh_trac/{predicted_label}.txt"

        # Đọc nội dung từ tệp văn bản tương ứng
        content = read_file_content(filename)

        # Hiển thị nhãn dự đoán
        st.subheader("Loại Dấu Vân Tay:")
        st.markdown(
            f"<p style='text-align:center; font-size:60px; color:blue'><strong>{predicted_label}</strong></p>",
            unsafe_allow_html=True)

        # Hiển thị nội dung từ tệp văn bản
        st.markdown("**Thông tin chi tiết:**")
        st.text_area(" ", content, height=300)