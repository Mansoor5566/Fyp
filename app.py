import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
import gdown
import os
import urllib.request


# ----------------- Page Config & Style -----------------
st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="centered")

st.markdown("""
    <style>
        body { background-color: #f7fdfc; }
        .main { background-color: #ffffff; padding: 2rem; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1, h3 { font-family: 'Segoe UI', sans-serif; }
        .center-text { text-align: center; }
    </style>
""", unsafe_allow_html=True)

# ----------------- Header -----------------
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("<h1 class='center-text' style='color:#0e6ba8;'>👁️‍🗨️ Diabetic Retinopathy Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='center-text' style='font-size:18px;'>Upload a fundus image below to detect the presence and stage of diabetic retinopathy.</p>", unsafe_allow_html=True)

# ----------------- Class Labels -----------------
labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
encoder = LabelEncoder()
encoder.fit(labels)

# ----------------- Load Model -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = Swin_T_Weights.DEFAULT
model = swin_t(weights=weights)

for param in model.parameters():
    param.requires_grad = False

    



# Download if not exists
MODEL_URL = "https://drive.google.com/uc?id=19pXpKWUQnh0HxzuHODOXysFtJ_W46bcV"
MODEL_PATH = "swin_model.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔄 Downloading model... Please wait."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded!")

# Replace classification head
model.head = nn.Sequential(
    nn.Linear(model.head.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(labels))
)
model = model.to(device)

try:
    state_dict = torch.load("swin_model.pth", map_location=device, weights_only=False)  
    for key in list(state_dict.keys()):
        if key.startswith("head.") and key not in model.state_dict():
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
except Exception as e:
    st.error(f"❌ Error loading model weights: {e}")
    st.stop()


# ----------------- File Upload -----------------
uploaded_file = st.file_uploader("📤 Upload a fundus image", type=["jpg", "jpeg", "png"])

# ----------------- Grad-CAM Function -----------------
def generate_gradcam(pil_img, model, input_tensor, class_index):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    img_np = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
    img_np = np.clip(img_np, 0, 1)

    try:
        # Attempt to select a valid layer for Swin-T
        target_layers = [model.features[-1][-1].norm1]
    except Exception:
        try:
            target_layers = [model.features[-1].blocks[-1].norm1]
        except Exception:
            return None  # Couldn't find target layer

    try:
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_index)])
        cam_image = show_cam_on_image(img_np, grayscale_cam[0], use_rgb=True)
        return cam_image
    except Exception as e:
        print(f"Grad-CAM internal error: {e}")
        return None


# ----------------- Inference -----------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        # Show uploaded image centered
        resized = image.resize((300, 300))
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(resized, caption="👁 Uploaded Image", use_container_width=True)

        # Preprocess
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
        ])
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            pred_label = encoder.inverse_transform([pred.item()])[0]

        # Show prediction
        st.markdown(f"""
        <div class='center-text'>
            <h3 style='color: #1e88e5;'>🔬 Predicted Condition: <span style='color: #d84315;'>{pred_label}</span></h3>
            <p style='font-size:18px;'>🔎 Confidence Score: <strong style='color: #388e3c;'>{conf.item() * 100:.2f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Grad-CAM
        gradcam_img = generate_gradcam(image, model, input_tensor, pred.item())
        if gradcam_img is not None:
            st.image(gradcam_img, caption="🔥 Grad-CAM Heatmap", use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# Close main box
st.markdown("</div>", unsafe_allow_html=True)
