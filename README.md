# 🪼 Jellyfish Species Classifier Web App

A Streamlit web application for classifying 6 different jellyfish species using transfer learning models (ResNet50V2 and EfficientNetB0).

## Features

- 📸 **Image Upload**: Upload jellyfish images for real-time classification
- 🎯 **AI-Powered Predictions**: Uses state-of-the-art transfer learning models
- 📊 **Confidence Visualization**: Interactive charts showing prediction confidence
- 🔬 **Model Comparison**: View performance metrics from Part 1 vs Part 2
- 🎨 **Modern UI**: Clean, responsive interface with custom styling

## Supported Species

1. Moon Jellyfish
2. Barrel Jellyfish
3. Blue Jellyfish
4. Compass Jellyfish
5. Lions Mane Jellyfish
6. Mauve Stinger Jellyfish

## Installation

### Prerequisites

- Python 3.9 or higher
- TensorFlow-compatible system (GPU optional but recommended)

### Setup

1. **Clone or navigate to the project directory**
   ```bash
   cd "C:\TP\YEAR 2 SEM 2\DLOR\DLOR_ASSIGNMENT\DLOR_JELLYFISH_DATASET"
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure required files are present**
   - `jellyfish_final_model.keras` (trained model from Part 2)
   - `part2_summary.json` (optional, for model metadata)
   - `app.py` (Streamlit application)

## Running the Application

### Option 1: Command Line
```bash
streamlit run app.py
```

### Option 2: With Custom Port
```bash
streamlit run app.py --server.port 8501
```

### Option 3: Open in Browser Automatically
```bash
streamlit run app.py --server.headless false
```

## Usage

1. **Launch the app** using one of the commands above
2. **Open your browser** to `http://localhost:8501` (usually opens automatically)
3. **Upload an image** of a jellyfish using the file uploader
4. **Click "Classify Jellyfish"** to get predictions
5. **View results** including:
   - Predicted species
   - Confidence percentage
   - Probability distribution for all classes
   - Model performance metrics

## Model Information

### Architecture
- **Part 1 Baseline**: Custom 4-Block CNN
- **Part 2 Models**: 
  - ResNet50V2 with transfer learning
  - EfficientNetB0 with transfer learning

### Performance
- Part 1 Baseline: ~61-64% accuracy
- Part 2 Best Model: Improved accuracy through:
  - Transfer learning from ImageNet
  - Systematic hyperparameter tuning (Keras-Tuner)
  - Enhanced validation split (20%)
  - Random Search (ResNet) and Grid Search (EfficientNet)

## Tips for Best Results

- Use **clear, well-lit images**
- Ensure the **jellyfish is centered**
- Avoid **heavily cropped or blurry images**
- Images should show **distinct features** of the jellyfish

## Project Structure

```
DLOR_JELLYFISH_DATASET/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── jellyfish_final_model.keras    # Best trained model (Part 2)
├── jellyfish_baseline_model.keras # Baseline model (Part 1)
│
├── part2_summary.json             # Model metadata and metrics
├── baseline_metrics.json          # Baseline performance metrics
│
├── DLOR_Part1Final.ipynb          # Part 1: Baseline CNN notebook
├── DLOR_Part2.ipynb               # Part 2: Transfer learning notebook
│
└── [Train/valid/test folders]     # Dataset directories
```

## Troubleshooting

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

### Model Not Found Error
Ensure `jellyfish_final_model.keras` exists in the same directory as `app.py`

### Memory Issues
If running on limited RAM, reduce batch processing or use CPU mode:
```bash
export CUDA_VISIBLE_DEVICES=""  # Linux/Mac
set CUDA_VISIBLE_DEVICES=       # Windows CMD
streamlit run app.py
```

## Technical Details

### Preprocessing Pipeline
- Input size: 128x128 pixels
- Normalization: [0, 1] range
- Color space: RGB (3 channels)
- Data augmentation (training only):
  - Random horizontal/vertical flip
  - Random rotation (30%)
  - Random zoom (20%)
  - Random contrast/brightness (20%)

### Model Architecture
**Custom Classification Head:**
- GlobalAveragePooling2D
- Dense(256) + BatchNorm + ReLU
- Dropout (tuned rate)
- Dense(6) + Softmax

## Credits

**Course**: Deep Learning and Object Recognition (DLOR)  
**Assignment**: Part 2 - Transfer Learning & Hyperparameter Tuning  
**Date**: February 2026

## License

Educational project for academic purposes.
