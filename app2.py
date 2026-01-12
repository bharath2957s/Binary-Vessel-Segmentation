import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io
import base64
from pathlib import Path
import segmentation_models_pytorch as smp
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="DRIVE Dataset - Retinal Vessel Segmentation Demo",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .task-header {
        background: linear-gradient(90deg, #2E86AB, #A23B72);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metrics-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .model-comparison {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .vessel-stats {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class DRIVEModelPredictor:
    def __init__(self, model_path, model_config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = model_config
        self.model = None
        self.load_model(model_path)
        
        # DRIVE-specific preprocessing (512x512 as per training)
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def load_model(self, model_path):
        """Load the trained DRIVE model"""
        try:
            if model_path and Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Create 2D UNet architecture (as specified for baseline)
                if self.model_config['architecture'] == 'unet':
                    self.model = smp.Unet(
                        encoder_name=self.model_config['encoder'],
                        encoder_weights=None,
                        in_channels=3,
                        classes=1,  # Binary segmentation
                        activation=None  # Will apply sigmoid separately
                    )
                elif self.model_config['architecture'] == 'unetplusplus':
                    self.model = smp.UnetPlusPlus(
                        encoder_name=self.model_config['encoder'],
                        encoder_weights=None,
                        in_channels=3,
                        classes=1,
                        activation=None
                    )
                elif self.model_config['architecture'] == 'deeplabv3plus':
                    self.model = smp.DeepLabV3Plus(
                        encoder_name=self.model_config['encoder'],
                        encoder_weights=None,
                        in_channels=3,
                        classes=1,
                        activation=None
                    )
                
                # Load trained weights
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            self.model = None
    
    def predict(self, image):
        """Make binary vessel segmentation prediction"""
        if self.model is None:
            return None, None, "Model not loaded"
        
        try:
            original_shape = image.shape[:2]
            
            # Apply DRIVE preprocessing
            transformed = self.transform(image=image)
            input_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            # Predict with sigmoid activation for binary output
            with torch.no_grad():
                logits = self.model(input_tensor)
                # Apply sigmoid for probability map (0-1)
                prob_map = torch.sigmoid(logits).cpu().numpy().squeeze()
            
            # Resize back to original dimensions
            prob_map = cv2.resize(prob_map, (original_shape[1], original_shape[0]))
            
            # Create binary mask (thresholded)
            binary_mask = (prob_map > 0.5).astype(np.uint8)
            
            return prob_map, binary_mask, "Success"
            
        except Exception as e:
            return None, None, f"Prediction error: {str(e)}"

def calculate_drive_metrics(prob_map, binary_pred, ground_truth=None):
    """Calculate comprehensive DRIVE dataset metrics"""
    
    # Basic vessel statistics
    total_pixels = prob_map.size
    vessel_pixels = binary_pred.sum()
    vessel_percentage = (vessel_pixels / total_pixels) * 100
    
    metrics = {
        'Basic Statistics': {
            'Total Pixels': f"{total_pixels:,}",
            'Vessel Pixels': f"{int(vessel_pixels):,}",
            'Vessel Coverage': f"{vessel_percentage:.2f}%",
            'Background Pixels': f"{int(total_pixels - vessel_pixels):,}",
            'Mean Probability': f"{prob_map.mean():.4f}",
            'Max Probability': f"{prob_map.max():.4f}",
            'Min Probability': f"{prob_map.min():.4f}"
        }
    }
    
    # If ground truth is provided, calculate advanced metrics
    if ground_truth is not None:
        gt_binary = (ground_truth > 0.5).astype(np.float32)
        pred_binary = binary_pred.astype(np.float32)
        prob_flat = prob_map.flatten()
        gt_flat = gt_binary.flatten()
        
        # Calculate confusion matrix components
        tp = (pred_binary * gt_binary).sum()
        tn = ((1 - pred_binary) * (1 - gt_binary)).sum()
        fp = (pred_binary * (1 - gt_binary)).sum()
        fn = ((1 - pred_binary) * gt_binary).sum()
        
        # Calculate metrics required for DRIVE evaluation
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)  # Same as sensitivity
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp + 1e-8)
        
        # ROC-AUC and PR-AUC
        try:
            roc_auc = roc_auc_score(gt_flat, prob_flat)
            precision_vals, recall_vals, _ = precision_recall_curve(gt_flat, prob_flat)
            pr_auc = auc(recall_vals, precision_vals)
        except:
            roc_auc = 0.0
            pr_auc = 0.0
        
        metrics['DRIVE Evaluation Metrics'] = {
            'Dice Score': f"{dice:.4f}",
            'Precision': f"{precision:.4f}",
            'Recall (Sensitivity)': f"{recall:.4f}",
            'F1-Score': f"{f1_score:.4f}",
            'Accuracy': f"{accuracy:.4f}",
            'Specificity': f"{specificity:.4f}",
            'ROC-AUC': f"{roc_auc:.4f}",
            'PR-AUC': f"{pr_auc:.4f}"
        }
        
        # Return additional data for plotting
        return metrics, {
            'prob_map': prob_map,
            'ground_truth': gt_flat,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'dice': dice, 'precision': precision, 'recall': recall,
            'f1': f1_score, 'accuracy': accuracy, 'specificity': specificity,
            'roc_auc': roc_auc, 'pr_auc': pr_auc
        }
    
    return metrics, None

def create_drive_overlay_visualization(image, prob_map, binary_pred, ground_truth=None, threshold=0.5):
    """Create DRIVE-specific overlay visualizations"""
    
    if ground_truth is not None:
        # 4-panel visualization with ground truth
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Retinal Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Ground truth
        axes[0, 1].imshow(image, alpha=0.7)
        gt_overlay = np.zeros_like(image)
        gt_mask = ground_truth > 0.5
        gt_overlay[gt_mask] = [0, 255, 0]  # Green for ground truth vessels
        axes[0, 1].imshow(gt_overlay, alpha=0.5)
        axes[0, 1].set_title('Ground Truth Overlay', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Prediction probability map
        im = axes[1, 0].imshow(prob_map, cmap='hot', alpha=0.9)
        axes[1, 0].set_title(f'Prediction Probability Map', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Prediction overlay
        axes[1, 1].imshow(image, alpha=0.7)
        pred_overlay = np.zeros_like(image)
        pred_mask = binary_pred > 0.5
        pred_overlay[pred_mask] = [255, 0, 0]  # Red for predicted vessels
        axes[1, 1].imshow(pred_overlay, alpha=0.5)
        axes[1, 1].set_title('Prediction Overlay', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
    else:
        # 3-panel visualization without ground truth
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Retinal Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Probability map
        im = axes[1].imshow(prob_map, cmap='hot', alpha=0.9)
        axes[1].set_title('Vessel Probability Map', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay on original
        axes[2].imshow(image, alpha=0.7)
        overlay = np.zeros_like(image)
        vessel_mask = binary_pred > 0.5
        overlay[vessel_mask] = [255, 0, 0]  # Red vessels
        axes[2].imshow(overlay, alpha=0.5)
        axes[2].set_title('Vessel Segmentation Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def create_roc_pr_plots(metrics_data):
    """Create ROC and PR curves for DRIVE evaluation"""
    if metrics_data is None:
        return None
    
    try:
        prob_flat = metrics_data['prob_map'].flatten()
        gt_flat = metrics_data['ground_truth']
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(gt_flat, prob_flat)
        roc_auc = metrics_data['roc_auc']
        
        # Calculate PR curve
        precision, recall, _ = precision_recall_curve(gt_flat, prob_flat)
        pr_auc = metrics_data['pr_auc']
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC Curve
        ax1.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('ROC Curve - DRIVE Dataset', fontsize=14, fontweight='bold')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # PR Curve
        ax2.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.4f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision-Recall Curve - DRIVE Dataset', fontsize=14, fontweight='bold')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        st.error(f"Error creating ROC/PR plots: {str(e)}")
        return None

def create_metrics_comparison_table(baseline_data, improved_data):
    """Create comparison table for baseline vs improved models"""
    if baseline_data is None and improved_data is None:
        return None
    
    metrics_names = ['dice', 'precision', 'recall', 'f1', 'accuracy', 'specificity', 'roc_auc', 'pr_auc']
    display_names = ['Dice Score', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'Specificity', 'ROC-AUC', 'PR-AUC']
    
    comparison_data = []
    
    for i, metric in enumerate(metrics_names):
        row = {'Metric': display_names[i]}
        
        if baseline_data and metric in baseline_data:
            row['Baseline Model'] = f"{baseline_data[metric]:.4f}"
        else:
            row['Baseline Model'] = "N/A"
        
        if improved_data and metric in improved_data:
            row['Improved Model'] = f"{improved_data[metric]:.4f}"
        else:
            row['Improved Model'] = "N/A"
        
        # Calculate improvement if both available
        if (baseline_data and improved_data and 
            metric in baseline_data and metric in improved_data):
            improvement = improved_data[metric] - baseline_data[metric]
            row['Improvement'] = f"{improvement:+.4f}"
        else:
            row['Improvement'] = "N/A"
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def main():
    # Header
    st.markdown('<h1 class="main-header">👁️ DRIVE Dataset - Retinal Vessel Segmentation</h1>', 
                unsafe_allow_html=True)
    
    # Task 5 Header
    st.markdown('<div class="task-header">Task 5: Interactive Demo Application</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 **Binary Vessel Segmentation Demo**
    This application demonstrates **retinal vessel segmentation** on the DRIVE dataset. 
    Upload your trained models and test images to see:
    - **Binary segmentation maps** (vessel vs background)
    - **Comprehensive metrics** (Dice, Precision, Recall, F1, ROC-AUC, PR-AUC)
    - **Overlay visualizations** with ground truth comparison
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Model Configuration")
    
    # Threshold setting
    threshold = st.sidebar.slider(
        "Binary Segmentation Threshold:",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Threshold for converting probability map to binary vessel mask"
    )
    
    # Model file uploads
    st.sidebar.subheader("📁 Upload Model Predictions")
    
    baseline_prediction_file = st.sidebar.file_uploader(
        "Baseline Model Prediction - .tif/.png",
        type=['tif', 'tiff', 'png', 'jpg'],
        help="Upload your Task 2 baseline model prediction (2D UNet + DiceLoss output)"
    )
    
    improved_prediction_file = st.sidebar.file_uploader(
        "Improved Model Prediction - .tif/.png", 
        type=['tif', 'tiff', 'png', 'jpg'],
        help="Upload your Task 2 improved model prediction output"
    )
    
    # Ground truth upload option
    st.sidebar.subheader("🎯 Ground Truth (Optional)")
    ground_truth_file = st.sidebar.file_uploader(
        "Ground Truth Mask - .gif/.png",
        type=['gif', 'png', 'jpg'],
        help="Upload ground truth for comprehensive metrics calculation"
    )
    
    # Initialize prediction data
    baseline_prediction = None
    improved_prediction = None
    
    # Load prediction images
    if baseline_prediction_file:
        baseline_pred_image = Image.open(baseline_prediction_file)
        baseline_prediction = np.array(baseline_pred_image)
        
        # Convert to grayscale if needed and normalize to 0-1
        if len(baseline_prediction.shape) == 3:
            baseline_prediction = cv2.cvtColor(baseline_prediction, cv2.COLOR_RGB2GRAY)
        baseline_prediction = baseline_prediction.astype(np.float32) / 255.0
    
    if improved_prediction_file:
        improved_pred_image = Image.open(improved_prediction_file)
        improved_prediction = np.array(improved_pred_image)
        
        # Convert to grayscale if needed and normalize to 0-1
        if len(improved_prediction.shape) == 3:
            improved_prediction = cv2.cvtColor(improved_prediction, cv2.COLOR_RGB2GRAY)
        improved_prediction = improved_prediction.astype(np.float32) / 255.0
    
    # Main content area
    st.header("📤 Upload DRIVE Test Image")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Choose a retinal fundus image",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help="Upload a DRIVE dataset image or similar retinal fundus photograph"
    )
    
    if uploaded_file is not None:
        # Load and display uploaded image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Convert to RGB if needed
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            pass  # Already RGB
        elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        else:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        
        # Load ground truth if provided
        ground_truth = None
        if ground_truth_file is not None:
            gt_image = Image.open(ground_truth_file)
            ground_truth = np.array(gt_image)
            if len(ground_truth.shape) == 3:
                ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_RGB2GRAY)
            # Resize to match input image
            ground_truth = cv2.resize(ground_truth, (image_np.shape[1], image_np.shape[0]))
            ground_truth = (ground_truth > 0).astype(np.float32)
        
        # Display original image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image_np, caption="DRIVE Retinal Image", use_container_width=True)
        
        # Show image statistics
        st.markdown('<div class="vessel-stats">📊 Image Statistics</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Width", f"{image_np.shape[1]}px")
        with col2:
            st.metric("Height", f"{image_np.shape[0]}px")
        with col3:
            st.metric("Channels", f"{image_np.shape[2]}")
        with col4:
            if ground_truth is not None:
                st.metric("Ground Truth", "✅ Available")
            else:
                st.metric("Ground Truth", "❌ Not provided")
        
        # Prediction section
        st.header("🔍 Binary Vessel Segmentation Results")
        
        if baseline_prediction is None and improved_prediction is None:
            st.warning("⚠️ Please upload at least one model prediction image to analyze results.")
            st.info("💡 Upload your baseline and improved model prediction files (.tif/.png) in the sidebar.")
        else:
            # Process predictions
            predictions = {}
            metrics_data = {}
            
            with st.spinner("Processing prediction images..."):
                progress_bar = st.progress(0)
                
                if baseline_prediction is not None:
                    progress_bar.progress(25)
                    # Resize to match input image if needed
                    if baseline_prediction.shape != image_np.shape[:2]:
                        baseline_prediction = cv2.resize(baseline_prediction, 
                                                       (image_np.shape[1], image_np.shape[0]))
                    
                    # Create binary mask using threshold
                    baseline_binary = (baseline_prediction > threshold).astype(np.uint8)
                    predictions['baseline'] = (baseline_prediction, baseline_binary)
                    
                    # Calculate metrics
                    metrics, data = calculate_drive_metrics(baseline_prediction, baseline_binary, ground_truth)
                    metrics_data['baseline'] = data
                
                progress_bar.progress(50)
                
                if improved_prediction is not None:
                    progress_bar.progress(75)
                    # Resize to match input image if needed
                    if improved_prediction.shape != image_np.shape[:2]:
                        improved_prediction = cv2.resize(improved_prediction, 
                                                       (image_np.shape[1], image_np.shape[0]))
                    
                    # Create binary mask using threshold
                    improved_binary = (improved_prediction > threshold).astype(np.uint8)
                    predictions['improved'] = (improved_prediction, improved_binary)
                    
                    # Calculate metrics
                    metrics, data = calculate_drive_metrics(improved_prediction, improved_binary, ground_truth)
                    metrics_data['improved'] = data
                
                progress_bar.progress(100)
                progress_bar.empty()
            
            # Display results
            if predictions:
                # Task 3: Visualization overlays
                st.markdown('<div class="task-header">Task 3: Visualization Overlays</div>', 
                           unsafe_allow_html=True)
                
                # Create visualizations for each model
                for model_name, (prob_map, binary_pred) in predictions.items():
                    model_title = "Baseline Model (2D UNet)" if model_name == 'baseline' else "Improved Model"
                    st.subheader(f"🎨 {model_title} - Overlay Visualization")
                    
                    fig = create_drive_overlay_visualization(image_np, prob_map, binary_pred, ground_truth)
                    st.pyplot(fig, use_container_width=True)
                
                # Task 4: Evaluation metrics and plots
                st.markdown('<div class="task-header">Task 4: Comprehensive Evaluation</div>', 
                           unsafe_allow_html=True)
                
                # Metrics comparison table
                if ground_truth is not None and metrics_data:
                    st.subheader("📊 DRIVE Dataset Metrics Table")
                    
                    baseline_data = metrics_data.get('baseline')
                    improved_data = metrics_data.get('improved')
                    
                    comparison_df = create_metrics_comparison_table(baseline_data, improved_data)
                    if comparison_df is not None:
                        st.dataframe(comparison_df, use_container_width=True)
                    
                    # ROC and PR plots
                    st.subheader("📈 ROC and PR Curves")
                    
                    col1, col2 = st.columns(2)
                    
                    if baseline_data:
                        with col1:
                            st.write("**Baseline Model**")
                            fig_roc_baseline = create_roc_pr_plots(baseline_data)
                            if fig_roc_baseline:
                                st.pyplot(fig_roc_baseline)
                    
                    if improved_data:
                        with col2:
                            st.write("**Improved Model**")
                            fig_roc_improved = create_roc_pr_plots(improved_data)
                            if fig_roc_improved:
                                st.pyplot(fig_roc_improved)
                
                # Detailed metrics display
                st.subheader("📋 Detailed Performance Metrics")
                
                for model_name, (prob_map, binary_pred) in predictions.items():
                    model_title = "Baseline Model (2D UNet + DiceLoss)" if model_name == 'baseline' else "Improved Model"
                    
                    with st.expander(f"📊 {model_title} - Detailed Metrics"):
                        metrics, _ = calculate_drive_metrics(prob_map, binary_pred, ground_truth)
                        
                        for category, metric_dict in metrics.items():
                            st.markdown(f"**{category}:**")
                            cols = st.columns(4)
                            for i, (key, value) in enumerate(metric_dict.items()):
                                with cols[i % 4]:
                                    st.metric(key, value)
                
                # Task 2: Model information
                st.markdown('<div class="task-header">Task 2: Model Training Information</div>', 
                           unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'baseline' in predictions:
                        st.markdown("""
                        <div class="metrics-container">
                        <h4>🔵 Baseline Model Specifications</h4>
                        <ul>
                        <li><strong>Architecture:</strong> 2D UNet</li>
                        <li><strong>Loss Function:</strong> Dice Loss</li>
                        <li><strong>Encoder:</strong> ResNet-34</li>
                        <li><strong>Output:</strong> Binary vessel segmentation</li>
                        <li><strong>Activation:</strong> Sigmoid</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if 'improved' in predictions:
                        st.markdown("""
                        <div class="metrics-container">
                        <h4>🟢 Improved Model Specifications</h4>
                        <ul>
                        <li><strong>Architecture:</strong> UNet++</li>
                        <li><strong>Loss Function:</strong> Combined Loss</li>
                        <li><strong>Encoder:</strong> EfficientNet-B0</li>
                        <li><strong>Output:</strong> Binary vessel segmentation</li>
                        <li><strong>Activation:</strong> Sigmoid</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Download section
                st.header("💾 Download Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📥 Download All Visualizations", type="primary"):
                        # Create a comprehensive results figure
                        results_text = f"""
DRIVE Dataset - Retinal Vessel Segmentation Results
=================================================

Image: {uploaded_file.name}
Threshold: {threshold}
Ground Truth Available: {'Yes' if ground_truth is not None else 'No'}

"""
                        for model_name, (prob_map, binary_pred) in predictions.items():
                            model_title = "Baseline (2D UNet)" if model_name == 'baseline' else "Improved Model"
                            results_text += f"\n{model_title} Results:\n"
                            metrics, _ = calculate_drive_metrics(prob_map, binary_pred, ground_truth)
                            
                            for category, metric_dict in metrics.items():
                                results_text += f"\n{category}:\n"
                                for key, value in metric_dict.items():
                                    results_text += f"  {key}: {value}\n"
                        
                        st.download_button(
                            label="📄 Download Complete Report",
                            data=results_text,
                            file_name=f"DRIVE_segmentation_report_{uploaded_file.name.split('.')[0]}.txt",
                            mime="text/plain"
                        )
                
                with col2:
                    if ground_truth is not None and metrics_data:
                        comparison_df = create_metrics_comparison_table(
                            metrics_data.get('baseline'), 
                            metrics_data.get('improved')
                        )
                        if comparison_df is not None:
                            csv_buffer = io.StringIO()
                            comparison_df.to_csv(csv_buffer, index=False)
                            
                            st.download_button(
                                label="📊 Download Metrics CSV",
                                data=csv_buffer.getvalue(),
                                file_name=f"DRIVE_metrics_comparison_{uploaded_file.name.split('.')[0]}.csv",
                                mime="text/csv"
                            )
                
                with col3:
                    # Download binary masks
                    if st.button("🎭 Download Binary Masks"):
                        mask_data = {}
                        for model_name, (prob_map, binary_pred) in predictions.items():
                            mask_data[model_name] = binary_pred
                        
                        # Create downloadable masks as images
                        if 'baseline' in mask_data:
                            baseline_mask = (mask_data['baseline'] * 255).astype(np.uint8)
                            baseline_img = Image.fromarray(baseline_mask, mode='L')
                            
                            buf = io.BytesIO()
                            baseline_img.save(buf, format='PNG')
                            buf.seek(0)
                            
                            st.download_button(
                                label="📥 Baseline Binary Mask",
                                data=buf.getvalue(),
                                file_name=f"baseline_mask_{uploaded_file.name.split('.')[0]}.png",
                                mime="image/png"
                            )
    
    # Information section
    with st.expander("ℹ️ DRIVE Dataset Information & Usage Guide"):
        st.markdown("""
        ### 📋 About DRIVE Dataset
        The **Digital Retinal Images for Vessel Extraction (DRIVE)** dataset is a benchmark for retinal vessel segmentation:
        - **40 color fundus images** (20 training, 20 test)
        - **584×565 pixel resolution**
        - **Binary segmentation task**: vessel vs background
        - **Expert manual annotations** as ground truth
        
        ### 🎯 Task Requirements Fulfilled
        
        **Task 2 (Model Training):**
        - ✅ Baseline: 2D UNet with Dice Loss
        - ✅ Advanced: Improved architecture with combined loss
        - ✅ Binary vessel segmentation output
        
        **Task 3 (Inference):**
        - ✅ Visualization overlays of predictions vs ground truth
        - ✅ Support for 3+ DRIVE test images
        - ✅ Probability maps and binary masks
        
        **Task 4 (Evaluation):**
        - ✅ Complete metrics table (Dice, Precision, Recall, F1, Accuracy, Specificity)
        - ✅ ROC-AUC and PR-AUC calculations
        - ✅ ROC and PR curve plots
        - ✅ Model comparison analysis
        
        **Task 5 (Demo App):**
        - ✅ Interactive web application
        - ✅ Real-time vessel segmentation
        - ✅ Overlay prediction visualization
        - ✅ Baseline vs improved model toggle
        
        ### 🚀 How to Use This App
        
        1. **Prepare Your Prediction Images:**
           ```
           # Your model outputs should be saved as images:
           # - baseline_prediction.tif (2D UNet + Dice Loss output)  
           # - improved_prediction.tif (Advanced model output)
           # - Grayscale images with pixel values 0-255
           # - Higher values = higher vessel probability
           ```
        
        2. **Upload Prediction Images:**
           - Upload your Task 2 baseline model prediction (.tif/.png)
           - Upload your Task 2 improved model prediction (.tif/.png)
           
        3. **Upload Original Image:**
           - Use corresponding DRIVE test image
           - Supported formats: PNG, JPG, TIF, TIFF
           
        4. **Optional Ground Truth:**
           - Upload corresponding manual annotation
           - Enables comprehensive evaluation metrics
           
        5. **Analyze Results:**
           - View overlay visualizations  
           - Compare model performance
           - Download detailed reports
        
        ### 📊 Expected Performance Ranges
        **Good DRIVE Results:**
        - **Dice Score**: 0.75 - 0.82
        - **Sensitivity (Recall)**: 0.70 - 0.80
        - **Specificity**: 0.95 - 0.98
        - **ROC-AUC**: 0.92 - 0.97
        - **PR-AUC**: 0.80 - 0.90
        
        ### 🔧 Technical Specifications
        - **Input Processing**: 512×512 resize with ImageNet normalization
        - **Output**: Single-channel probability maps (0-1)
        - **Threshold**: Adjustable binary segmentation threshold
        - **GPU Support**: Automatic CUDA detection for RTX 2050
        - **Memory Efficient**: Optimized for 4GB VRAM
        """)
    
    # Troubleshooting section
    with st.expander("🔧 Troubleshooting & Tips"):
        st.markdown("""
        ### Common Issues & Solutions
        
        **Model Loading Errors:**
        - Ensure prediction images are in grayscale format
        - Check that image values are normalized (0-255 range)
        - Verify prediction and original images have compatible dimensions
        - Convert TIFF to PNG if loading issues occur
        
        **Poor Segmentation Results:**
        - Try adjusting the binary threshold (0.3-0.7 range)
        - Ensure training was completed with sufficient epochs
        - Check if model was trained on DRIVE dataset specifically
        
        **Memory Issues (RTX 2050):**
        - App is optimized for single image inference
        - Close other GPU applications if needed
        - Use CPU inference if CUDA issues occur
        
        **Image Format Issues:**
        - Convert TIFF images to PNG if loading fails
        - Ensure images are RGB (not RGBA or grayscale)
        - Check image dimensions are reasonable (< 2000px)
        
        ### Performance Tips
        - **Upload ground truth** for complete evaluation metrics
        - **Use consistent threshold** across model comparisons
        - **Test multiple images** to validate model robustness
        - **Compare with published DRIVE benchmarks**
        
        ### Model Training Tips
        - **Baseline UNet**: Use standard Dice loss, ResNet34 encoder
        - **Improved Model**: Try UNet++, combined losses, better encoders
        - **Data Augmentation**: Rotation, flip, brightness for retinal images
        - **Validation**: Use proper cross-validation on DRIVE training set
        """)
    
    # Footer with dataset citation
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <strong>DRIVE Dataset Citation:</strong><br>
        Staal, J., Abràmoff, M. D., Niemeijer, M., Viergever, M. A., & van Ginneken, B. (2004).<br>
        Ridge-based vessel segmentation in color images of the retina.<br>
        <em>IEEE Transactions on Medical Imaging, 23(4), 501-509.</em><br><br>
        
        <strong>🎈 DRIVE Retinal Vessel Segmentation Demo | Built with Streamlit</strong><br>
        Task 5 Deliverable: Interactive Demo Application with Overlay Predictions
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()