"""
Jellyfish Classification Web Application
Part 2: Transfer Learning Model Deployment

This application uses the best-performing model from Part 2 to classify
6 different species of jellyfish in real-time.
"""

import streamlit as st
import numpy as np
from PIL import Image
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import datetime

# Try to import TensorFlow with helpful error message
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError as e:
    TF_AVAILABLE = False
    TF_ERROR = str(e)

# Page configuration
st.set_page_config(
    page_title="Jellyfish Classifier",
    page_icon="🪼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and metadata
@st.cache_resource
def load_model_and_metadata():
    """Load the trained model from Part 2"""
    import os
    try:
        # Load Part 2 model ONLY - use .h5 format for TF version compatibility
        # Try multiple path resolution strategies
        try:
            base_path = Path(__file__).parent.resolve()
        except:
            base_path = Path(os.getcwd())
        
        # Try .h5 format first (compatible with TF 2.10 and TF 2.20)
        model_path = base_path / 'jellyfish_final_model.h5'
        
        if not model_path.exists():
            # Fallback to .keras format
            model_path = base_path / 'jellyfish_final_model.keras'
        
        if not model_path.exists():
            # Try the directory where this script actually is
            alternate_path = Path(r"C:\TP\YEAR 2 SEM 2\DLOR\DLOR_ASSIGNMENT\DLOR_JELLYFISH_DATASET\jellyfish_final_model.h5")
            if alternate_path.exists():
                model_path = alternate_path
        
        if not model_path.exists():
            st.error(f"❌ Model file not found!")
            st.error(f"   Tried: {model_path}")
            st.error("   Please run cell 23 in DLOR_Part2.ipynb to save the model as .h5 format!")
            st.info(f"   Current directory: {os.getcwd()}")
            st.info(f"   Base path: {base_path}")
            return None, None, None
        
        # Load the Part 2 transfer learning model
        model = keras.models.load_model(str(model_path))
        st.success(f"✅ Loaded Part 2 Model: {model_path.name} ({model.count_params():,} params)")
        
        # Load Part 2 summary
        metadata = None
        summary_path = base_path / 'part2_summary.json'
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                metadata = json.load(f)
        
        # Class names (must match training order from notebook)
        class_names = [
            'Moon Jellyfish',
            'Barrel Jellyfish',
            'Blue Jellyfish',
            'Compass Jellyfish',
            'Lions Mane Jellyfish',
            'Mauve Stinger Jellyfish'
        ]
        
        return model, metadata, class_names
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        st.info("💡 Solution: Run cell 30 in DLOR_Part2.ipynb to re-save the model properly")
        return None, None, None

# Preprocessing function
def preprocess_image(image, target_size=(128, 128)):
    """Preprocess image for model prediction"""
    # Resize image
    img = image.resize(target_size)
    
    # Convert to array
    img_array = np.array(img)
    
    # Ensure RGB (3 channels)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    # Normalize to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Prediction function
def predict(model, image, class_names):
    """Make prediction on image"""
    # Preprocess
    processed_img = preprocess_image(image)
    
    # Predict
    predictions = model.predict(processed_img, verbose=0)
    
    # Get top class
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Get all probabilities
    probabilities = {class_names[i]: float(predictions[0][i] * 100) 
                    for i in range(len(class_names))}
    
    return predicted_class, confidence, probabilities

# Main app
def main():
    # Check TensorFlow availability first
    if not TF_AVAILABLE:
        st.error("🚨 TensorFlow is not installed in this environment!")
        st.markdown("""
        ### How to Fix:
        
        **Option 1: Use your conda TensorFlow environment**
        ```bash
        conda activate tensorflow-gpu
        pip install streamlit plotly
        streamlit run app.py
        ```
        
        **Option 2: Install TensorFlow in current environment**
        ```bash
        pip install tensorflow
        streamlit run app.py
        ```
        
        **Option 3: Use the correct Python environment**
        ```bash
        # Find your conda environments
        conda env list
        
        # Activate TensorFlow environment
        conda activate tensorflow-gpu
        
        # Install Streamlit in that environment
        pip install streamlit plotly
        
        # Run the app
        streamlit run app.py
        ```
        """)
        st.stop()
    
    # Load model
    model, metadata, class_names = load_model_and_metadata()
    
    if model is None:
        st.error("⚠️ Could not load model. Please ensure 'jellyfish_final_model.keras' exists in the directory.")
        return
    
    # Header
    st.markdown('<h1 class="main-header">🪼 Jellyfish Species Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image to identify the jellyfish species using AI</p>', unsafe_allow_html=True)
    
    # Initialize session state for history
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    
    # Sidebar
    with st.sidebar:
        st.header("🐙 Supported Species")
        for i, species in enumerate(class_names, 1):
            st.write(f"{i}. {species}")
        
        st.markdown("---")
        st.caption("🔬 DLOR Assignment - Part 2")
        st.caption("Transfer Learning Model")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Jellyfish Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a jellyfish"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Predict button
            if st.button("🔍 Classify Jellyfish", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, probabilities = predict(model, image, class_names)
                    
                    # Store in session state
                    st.session_state['prediction'] = predicted_class
                    st.session_state['confidence'] = confidence
                    st.session_state['probabilities'] = probabilities
                    
                    # Add to history
                    history_entry = {
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'species': predicted_class,
                        'confidence': confidence,
                        'image_name': uploaded_file.name
                    }
                    st.session_state['history'].insert(0, history_entry)  # Add to beginning
                    
                    # Keep only last 10 predictions
                    if len(st.session_state['history']) > 10:
                        st.session_state['history'] = st.session_state['history'][:10]
    
    with col2:
        st.subheader("🎯 Classification Results")
        
        if 'prediction' in st.session_state:
            # Main prediction
            st.markdown(
                f'<div class="prediction-box">🪼 {st.session_state["prediction"]}</div>',
                unsafe_allow_html=True
            )
            
            # Confidence
            st.markdown(
                f'<div class="confidence-box">'
                f'<b>Confidence:</b> {st.session_state["confidence"]:.2f}%</div>',
                unsafe_allow_html=True
            )
            
            # Confidence bar
            fig = go.Figure(go.Bar(
                x=[st.session_state['confidence']],
                y=['Confidence'],
                orientation='h',
                marker=dict(
                    color='#667eea',
                    line=dict(color='#764ba2', width=2)
                ),
                text=[f"{st.session_state['confidence']:.1f}%"],
                textposition='inside',
                textfont=dict(size=16, color='white')
            ))
            
            fig.update_layout(
                xaxis=dict(range=[0, 100], showticklabels=False),
                yaxis=dict(showticklabels=False),
                height=100,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # All probabilities
            st.markdown("---")
            st.subheader("📊 All Class Probabilities")
            
            # Sort probabilities
            sorted_probs = sorted(
                st.session_state['probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Create bar chart
            prob_df = pd.DataFrame(sorted_probs, columns=['Species', 'Probability'])
            
            fig2 = px.bar(
                prob_df,
                x='Probability',
                y='Species',
                orientation='h',
                color='Probability',
                color_continuous_scale='viridis',
                text='Probability'
            )
            
            fig2.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside'
            )
            
            fig2.update_layout(
                xaxis_title="Confidence (%)",
                yaxis_title="",
                showlegend=False,
                height=400,
                xaxis=dict(range=[0, 105])
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
        else:
            st.info("👆 Upload an image and click 'Classify Jellyfish' to see results")
            
            # Show example
            st.markdown("### 💡 Tips for Best Results:")
            st.markdown("""
            - Use clear, well-lit images
            - Ensure jellyfish is centered
            - Avoid heavily cropped images
            - Image should show distinct features
            """)
    
    # Footer section - Classification History
    st.markdown("---")
    st.header("📜 Classification History")
    
    if st.session_state['history']:
        # Display history as a table
        st.write(f"Showing last {len(st.session_state['history'])} predictions:")
        
        # Create DataFrame for history
        history_df = pd.DataFrame(st.session_state['history'])
        history_df.columns = ['Timestamp', 'Predicted Species', 'Confidence (%)', 'Image File']
        
        # Format confidence to 2 decimal places
        history_df['Confidence (%)'] = history_df['Confidence (%)'].apply(lambda x: f"{x:.2f}%")
        
        # Display as styled table
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Visualization of history confidence
        if len(st.session_state['history']) > 1:
            st.subheader("📊 Confidence Trend")
            
            # Prepare data for chart
            chart_data = pd.DataFrame({
                'Prediction #': [f"#{i+1}" for i in range(len(st.session_state['history']))],
                'Confidence': [entry['confidence'] for entry in st.session_state['history']],
                'Species': [entry['species'] for entry in st.session_state['history']]
            })
            
            # Reverse to show oldest to newest
            chart_data = chart_data.iloc[::-1].reset_index(drop=True)
            
            fig_history = px.line(
                chart_data,
                x='Prediction #',
                y='Confidence',
                markers=True,
                text='Species',
                title='Confidence Levels Across Predictions'
            )
            
            fig_history.update_traces(
                textposition='top center',
                line=dict(color='#667eea', width=3),
                marker=dict(size=10, color='#764ba2')
            )
            
            fig_history.update_layout(
                yaxis_title="Confidence (%)",
                xaxis_title="",
                height=400,
                yaxis=dict(range=[0, 105]),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_history, use_container_width=True)
        
        # Clear history button
        if st.button("🗑️ Clear History", use_container_width=False):
            st.session_state['history'] = []
            st.rerun()
    else:
        st.info("👆 No classification history yet. Upload and classify images to build your history!")

if __name__ == "__main__":
    main()
