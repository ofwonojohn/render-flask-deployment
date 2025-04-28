import os
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import base64

app = Flask(__name__)
CORS(app)

# 1. Define the CNN Model
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

# 2. GradCAM Class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor):
        output = self.model(input_tensor)
        self.model.zero_grad()
        pred_class = output.argmax(dim=1)
        one_hot = torch.zeros_like(output)
        one_hot[0][pred_class] = 1
        output.backward(gradient=one_hot)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.squeeze(0)
        
        for i in range(activations.size(0)):
            activations[i] *= pooled_gradients[i]
        
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap

# 3. Load the Model
model = CNNModel()
model_path = "cnn_enhanced.pth"

try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print(f"✅ Model loaded successfully from {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 4. Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Utility: Convert NumPy array to Base64
def encode_image_to_base64(img_np):
    _, buffer = cv2.imencode('.jpg', img_np)
    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

# 6. Prediction and GradCAM
def predict_and_explain(image: Image.Image):
    try:
        # Original size image
        original_image = np.array(image.resize((224, 224)))

        # Preprocess
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(image_tensor)
        
        prediction = torch.argmax(output, dim=1).item()
        label = "Pothole Detected" if prediction == 1 else "No Pothole Detected"

        # Grad-CAM
        target_layer = model.base_model[2]
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam.generate(image_tensor)

        # Prepare GradCAM visualization
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Superimpose GradCAM on original image
        superimposed_img = cv2.addWeighted(original_image, 1, heatmap, 0.4, 0)

        # Encode both images
        original_base64 = encode_image_to_base64(original_image)
        gradcam_base64 = encode_image_to_base64(superimposed_img)

        return label, original_base64, gradcam_base64

    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", None, None

# 7. Flask Routes
@app.route('/predict', methods=['POST'])
def predict_pothole():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    try:
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        label, original_img, gradcam_img = predict_and_explain(image)

        if original_img is None:
            return jsonify({'error': 'Prediction failed'}), 500

        return jsonify({
            'filename': file.filename,
            'prediction': label,
            'original_image_base64': original_img,
            'gradcam_image_base64': gradcam_img
        })

    except Exception as e:
        return jsonify({'error': f"Failed to process image: {str(e)}"}), 500

@app.route('/')
def home():
    return render_template('index.html')

# 8. Run App
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
