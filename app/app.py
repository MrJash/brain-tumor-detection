import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS
st.markdown("""
<style>
    /* Result banner */
    .result-banner {
        padding: 1.25rem 1.5rem;
        border-radius: 6px;
        text-align: center;
        margin: 1rem 0;
    }
    .result-banner.danger {
        background: #dc2626;
    }
    .result-banner.safe {
        background: #16a34a;
    }
    .result-banner h2 {
        color: #ffffff !important;
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin: 0;
        padding: 0;
    }
    .result-banner p {
        color: rgba(255,255,255,0.85);
        margin: 0.25rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Probability row */
    .prob-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0.4rem 0;
        font-size: 0.9rem;
    }
    .prob-row .label {
        width: 110px;
        text-align: right;
        font-weight: 500;
        color: #9ca3af;
        flex-shrink: 0;
    }
    .prob-row .label.active {
        color: #e2e8f0;
        font-weight: 700;
    }
    .prob-row .bar-bg {
        flex: 1;
        height: 22px;
        background: rgba(255,255,255,0.08);
        border-radius: 4px;
        overflow: hidden;
    }
    .prob-row .bar-fill {
        height: 100%;
        background: #3b82f6;
        border-radius: 4px;
        min-width: 2px;
    }
    .prob-row .bar-fill.top {
        background: #3b82f6;
    }
    .prob-row .pct {
        width: 52px;
        text-align: right;
        font-variant-numeric: tabular-nums;
        color: #9ca3af;
        flex-shrink: 0;
    }
    .prob-row .pct.active {
        color: #e2e8f0;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model and metadata"""
    model_path = Path("models/tumor_model.pth")
    
    if not model_path.exists():
        st.error(f"❌ Model not found at {model_path}")
        st.error("Please train the model first by running the training notebook.")
        st.stop()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    class_names = checkpoint.get('class_names', ['glioma', 'meningioma', 'notumor', 'pituitary'])
    num_classes = len(class_names)
    
    # Initialize model
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, class_names, checkpoint


@st.cache_data
def load_metrics():
    """Load training metrics if available"""
    metrics_path = Path("outputs/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None


def get_image_transform():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def predict_image(model, image, class_names, transform):
    """Make prediction on uploaded image"""
    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    predicted_label = class_names[predicted_class.item()]
    confidence_score = confidence.item() * 100
    
    # Get all class probabilities
    all_probs = {class_names[i]: probabilities[0][i].item() * 100 
                 for i in range(len(class_names))}
    
    return predicted_label, confidence_score, all_probs


def get_confidence_class(confidence):
    """Get CSS class based on confidence level"""
    if confidence >= 80:
        return "confidence-high"
    elif confidence >= 60:
        return "confidence-medium"
    else:
        return "confidence-low"


def display_result(predicted_label, confidence, all_probs):
    """Display prediction results"""
    is_tumor = predicted_label.lower() != 'notumor'
    
    # Result banner - solid color, white text, fully readable
    if is_tumor:
        display_name = predicted_label.capitalize()
        st.markdown(f"""
            <div class="result-banner danger">
                <h2>TUMOR DETECTED</h2>
                <p>Type: {display_name} &mdash; Confidence: {confidence:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-banner safe">
                <h2>NO TUMOR DETECTED</h2>
                <p>Confidence: {confidence:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Probabilities
    st.caption("Class probabilities")
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    top_class = sorted_probs[0][0]
    
    bars_html = ""
    for name, prob in sorted_probs:
        is_top = name == top_class
        label_cls = "label active" if is_top else "label"
        fill_cls = "bar-fill top" if is_top else "bar-fill"
        pct_cls = "pct active" if is_top else "pct"
        display = name.capitalize()
        bars_html += f"""
        <div class="prob-row">
            <span class="{label_cls}">{display}</span>
            <div class="bar-bg"><div class="{fill_cls}" style="width:{prob:.1f}%"></div></div>
            <span class="{pct_cls}">{prob:.1f}%</span>
        </div>"""
    
    st.markdown(bars_html, unsafe_allow_html=True)


def main():
    # Header
    st.title("Brain Tumor Detection")
    st.caption("MRI scan classification using ResNet18 -- local GPU training")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("Classifies brain MRI scans into four categories using a ResNet18 model trained with transfer learning.")
        st.markdown("""
        **Glioma** -- brain / spinal cord  
        **Meningioma** -- meninges  
        **Pituitary** -- pituitary gland  
        **No Tumor** -- healthy
        """)
        
        st.divider()
        st.header("Model")
        try:
            model, class_names, checkpoint = load_model()
            metrics = load_metrics()
            
            c1, c2 = st.columns(2)
            c1.metric("Val Accuracy", f"{checkpoint.get('val_acc', 0)*100:.1f}%")
            if metrics:
                c2.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0)*100:.1f}%")
            st.metric("Best Epoch", checkpoint.get('epoch', 'N/A'))
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()
        
        st.divider()
        st.caption("For educational purposes only. Not a substitute for professional medical diagnosis.")
    
    # Main content
    st.divider()
    
    uploaded_file = st.file_uploader(
        "Upload MRI scan",
        type=['jpg', 'jpeg', 'png'],
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        col_img, col_res = st.columns([1, 1], gap="large")
        
        with col_img:
            st.image(image, width="stretch")
        
        with col_res:
            if st.button("Analyze", type="primary", width="stretch"):
                with st.spinner("Running model..."):
                    try:
                        transform = get_image_transform()
                        predicted_label, confidence, all_probs = predict_image(
                            model, image, class_names, transform
                        )
                        display_result(predicted_label, confidence, all_probs)
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
    else:
        st.info("Upload a brain MRI image to get started.")


if __name__ == "__main__":
    main()
