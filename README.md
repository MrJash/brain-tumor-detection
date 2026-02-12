# 🧠 Brain Tumor Detection & Classification

A comprehensive deep learning project for detecting and classifying brain tumors from MRI scans using PyTorch, OpenCV, and Streamlit. Features **local GPU training** with CUDA acceleration and an interactive web-based inference dashboard.

## 📋 Overview

This system classifies brain MRI scans into **4 categories**:

- **Glioma** - Tumors that occur in the brain and spinal cord
- **Meningioma** - Tumors arising from the meninges
- **Pituitary** - Tumors in the pituitary gland
- **No Tumor** - Healthy brain scans

### Key Features

✅ **Local GPU Training** - Automatic CUDA detection and GPU acceleration  
✅ **Transfer Learning** - Pretrained ResNet18 model for efficient training  
✅ **Real-time Inference** - Interactive Streamlit dashboard  
✅ **Comprehensive Metrics** - Accuracy, confusion matrix, classification reports  
✅ **Reproducible Results** - Fixed random seeds and version control  

---

## 🏗️ Project Structure

```
brain-tumor-detection2/
├── data/                   # ⚠️ NOT INCLUDED (add your own dataset)
│   ├── train/              # Training images (~5,700 images)
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── test/               # Test/validation images
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
├── notebooks/
│   └── training.ipynb      # Main training notebook
├── models/
│   └── tumor_model.pth     # Trained model (generated after training)
├── app/
│   └── app.py              # Streamlit inference dashboard
├── outputs/
│   ├── metrics.json        # Training metrics (generated)
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
├── requirements.txt        # Python dependencies
├── setup_venv.bat          # Virtual environment setup (Windows)
├── run_app.bat             # Quick launch script
└── README.md
```

---

## ⚠️ Important: Dataset Setup

The `data/` directory is **NOT included** in this repository due to its large size (~2GB+). You need to:

1. **Obtain the dataset** from a brain tumor MRI source (e.g., Kaggle: "Brain Tumor MRI Dataset")
2. **Create the directory structure**:
   ```
   data/
   ├── train/
   │   ├── glioma/
   │   ├── meningioma/
   │   ├── notumor/
   │   └── pituitary/
   └── test/
       ├── glioma/
       ├── meningioma/
       ├── notumor/
       └── pituitary/
   ```
3. **Place your images** in the appropriate class folders (JPG/PNG format)

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11** (recommended)
- **NVIDIA GPU** with CUDA support (optional but recommended for training)
- **~2GB free disk space** (for dependencies and model)

### 1. Setup Virtual Environment

Run the setup script to create a virtual environment and install dependencies:

```bash
# Windows
setup_venv.bat

# Manual setup (all platforms)
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Verify GPU Setup (Optional)

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### 3. Prepare Your Dataset

Before training, ensure your dataset is organized in the `data/` directory following the structure above. Download the brain tumor MRI dataset and place images in their respective class folders.

### 4. Train the Model

Open and run the training notebook:

```bash
# Activate virtual environment first
venv\Scripts\activate

# Start Jupyter
jupyter notebook notebooks/training.ipynb
```

**Run all cells** in the notebook. Training takes approximately:
- **With GPU**: 10-20 minutes (10 epochs)
- **Without GPU**: 1-2 hours (10 epochs)

The notebook will:
1. Load data from `data/train` and `data/test`
2. Train ResNet18 model on GPU (if available)
3. Save best model to `models/tumor_model.pth`
4. Generate metrics and visualizations in `outputs/`

### 5. Run the Dashboard

After training, launch the Streamlit inference app:

```bash
# Quick launch (Windows)
run_app.bat

# Manual launch
venv\Scripts\activate
streamlit run app/app.py
```

Open your browser to `http://localhost:8501`

---

## 📊 Model Architecture

**Base Model:** ResNet18 (pretrained on ImageNet)

**Modifications:**
- Final FC layer replaced for 4-class classification
- Input: 224×224×3 RGB images
- Output: 4 class probabilities (softmax)

**Training Configuration:**
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)
- Scheduler: ReduceLROnPlateau
- Epochs: 10 (configurable)
- Batch Size: 32
- Data Augmentation: Random flip, rotation, color jitter

---

## 📈 Expected Performance

Based on the dataset structure:

| Metric | Expected Value |
|--------|---------------|
| Training Accuracy | 95-98% |
| Validation Accuracy | 90-95% |
| Test Accuracy | 88-93% |

*Actual results may vary based on random initialization and GPU configuration.*

---

## 🖥️ Using the Streamlit Dashboard

1. **Upload Image**: Click "Browse files" and select an MRI scan (JPG/PNG)
2. **Analyze**: Click "Analyze Image" button
3. **View Results**:
   - **Tumor Status**: YES/NO indicator
   - **Tumor Type**: Glioma, Meningioma, or Pituitary (if detected)
   - **Confidence Score**: Model confidence percentage
   - **Class Probabilities**: Probability breakdown for all classes

### Sample Workflow

```
Upload MRI → Click Analyze → View Prediction
                            ↓
                    🔴 TUMOR DETECTED: YES
                    Tumor Type: GLIOMA
                    Confidence: 94.32%
```

---

## 🔧 Technical Details

### GPU Requirements

- **NVIDIA GPU** with CUDA 11.8+ support
- **Minimum 2GB VRAM** (4GB+ recommended)
- **Driver**: Latest NVIDIA drivers

### Dataset Format

⚠️ **The dataset is NOT included in this repository.** You must provide your own dataset.

Images must follow this structure:
```
data/
  train/
    <class_name>/
      image1.jpg
      image2.jpg
```

Class names are **automatically inferred** from folder names: `glioma`, `meningioma`, `notumor`, `pituitary`

### Model Artifacts

After training, the following files are generated:

| File | Description |
|------|-------------|
| `models/tumor_model.pth` | Trained model weights + metadata |
| `outputs/metrics.json` | Training/validation metrics |
| `outputs/training_history.png` | Loss and accuracy curves |
| `outputs/confusion_matrix.png` | Classification confusion matrix |
| `outputs/sample_predictions.png` | Visualization of predictions |

---

## 📦 Dependencies

Core libraries (see `requirements.txt`):

- **torch** >= 2.0.0 (with CUDA support)
- **torchvision** >= 0.15.0
- **opencv-python** >= 4.8.0
- **streamlit** >= 1.28.0
- **scikit-learn** >= 1.3.0
- **matplotlib**, **seaborn**, **pandas**, **numpy**

---

## 🛠️ Troubleshooting

### GPU Not Detected

```python
# Check CUDA availability
import torch
print(torch.cuda.is_available())  # Should return True

# Install correct PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Model Not Found Error

Ensure you've run the training notebook completely:
1. Open `notebooks/training.ipynb`
2. Run all cells (Kernel → Restart & Run All)
3. Verify `models/tumor_model.pth` exists

### Streamlit Port Conflict

```bash
# Use different port
streamlit run app/app.py --server.port 8502
```

---

## 📝 Training Customization

Edit hyperparameters in `notebooks/training.ipynb`:

```python
CONFIG = {
    'batch_size': 32,        # Increase if GPU has more memory
    'num_epochs': 10,        # More epochs = better accuracy
    'learning_rate': 0.001,  # Learning rate
    'img_size': 224,         # Input image size
}
```

---

## ⚠️ Important Notes

### Training Rules

✅ **DO**: Train in `notebooks/training.ipynb`  
✅ **DO**: Use GPU for faster training  
✅ **DO**: Save model to `models/tumor_model.pth`  

❌ **DON'T**: Train inside Streamlit app  
❌ **DON'T**: Use Google Colab or cloud training  
❌ **DON'T**: Modify data directory structure  

### Inference Rules

- Streamlit app is **inference-only** (no training logic)
- Model is loaded once at startup (cached)
- Each prediction takes ~2 seconds (CPU) or <1 second (GPU)

---

## 🎯 Use Cases

- **Educational**: Learn PyTorch, transfer learning, and deep learning workflows
- **Research**: Baseline for medical image classification projects
- **Prototyping**: Rapid prototyping of MRI analysis systems

---

## 📄 License & Disclaimer

This project is for **educational purposes only**.

⚕️ **Medical Disclaimer**: This tool is NOT a medical device and should NOT be used for clinical diagnosis. Always consult qualified healthcare professionals for medical advice.

---

## 🤝 Contributing

Suggestions for improvement:

1. Add more data augmentation techniques
2. Implement ensemble models
3. Add GradCAM visualizations for interpretability
4. Support for more tumor types
5. Export to ONNX for production deployment

---

## 📧 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure dataset structure matches the expected format

---

## 🏆 Acknowledgments

- **PyTorch** team for excellent deep learning framework
- **Streamlit** for interactive dashboards
- **torchvision** for pretrained ResNet models
- Medical imaging community for dataset contributions

---

**Built with ❤️ using PyTorch, OpenCV, and Streamlit**

*Last updated: February 2026*
