import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import DRIVEDataset, get_transforms
from model import get_model
from train import calculate_metrics

def evaluate_model(model_path, data_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = get_model(config['model_name'], config['encoder']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test dataset
    test_dataset = DRIVEDataset(
        data_root, 
        split='test', 
        transform=get_transforms('test')
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    all_metrics = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            filename = batch['filename'][0]
            
            outputs = model(images)
            
            metrics = calculate_metrics(outputs, masks)
            all_metrics.append(metrics)
            
            # Visualize results
            if i < 5:  # Show first 5 results
                visualize_prediction(images[0], masks[0], outputs[0], filename, i)
    
    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    print("Test Results:")
    for key, value in avg_metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")
    
    return avg_metrics

def visualize_prediction(image, target, prediction, filename, idx):
    # Convert tensors to numpy
    image = image.cpu().numpy().transpose(1, 2, 0)
    target = target.cpu().numpy().squeeze()
    prediction = torch.sigmoid(prediction).cpu().numpy().squeeze()
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(target, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title('Prediction (Probability)')
    axes[2].axis('off')
    
    axes[3].imshow(prediction > 0.5, cmap='gray')
    axes[3].set_title('Prediction (Thresholded)')
    axes[3].axis('off')
    
    plt.suptitle(f'Results for {filename}')
    plt.savefig(f'results/prediction_{idx}.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    evaluate_model('models/best_model.pth', 'data/DRIVE')           