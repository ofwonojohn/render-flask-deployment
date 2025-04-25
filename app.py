import os
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CNNModel, self).__init__()
        self.base_model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc_layers(x)
        return self.classifier(x)

# Load the model once globally
model = CNNModel()
model_path = "cnn_enhanced.pth"

try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

# Preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Prediction logic
def predict(image: Image.Image) -> str:
    try:
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).item()
        return "Pothole Detected" if prediction == 1 else "No Pothole"
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error during prediction"

# Flask Routes
@app.route('/predict', methods=['POST'])
def predict_pothole():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    try:
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        result = predict(image)
        return jsonify({'filename': file.filename, 'prediction': result})
    except Exception as e:
        return jsonify({'error': f"Failed to process image: {str(e)}"}), 500

@app.route('/')
def home():
    return render_template('index.html')

# App Entry Point

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
