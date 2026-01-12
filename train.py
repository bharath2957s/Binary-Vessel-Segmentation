import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import DRIVEDataset, get_transforms
from model import get_model

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, inputs, targets):
        return self.alpha * self.bce(inputs, targets) + (1 - self.alpha) * self.dice(inputs, targets)

def calculate_metrics(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred) > threshold
    target = target > 0.5
    
    pred = pred.float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = intersection / (union + 1e-8)
    dice = 2 * intersection / (pred.sum() + target.sum() + 1e-8)
    
    tp = (pred * target).sum()
    tn = ((1-pred) * (1-target)).sum()
    fp = (pred * (1-target)).sum()
    fn = ((1-pred) * target).sum()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    
    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'accuracy': accuracy.item(),
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item()
    }

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    metrics = {'iou': 0, 'dice': 0, 'accuracy': 0, 'sensitivity': 0, 'specificity': 0}
    
    for batch in tqdm(dataloader, desc='Training'):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        batch_metrics = calculate_metrics(outputs, masks)
        for key in metrics:
            metrics[key] += batch_metrics[key]
    
    # Average metrics
    num_batches = len(dataloader)
    total_loss /= num_batches
    for key in metrics:
        metrics[key] /= num_batches
    
    return total_loss, metrics

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    metrics = {'iou': 0, 'dice': 0, 'accuracy': 0, 'sensitivity': 0, 'specificity': 0}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            batch_metrics = calculate_metrics(outputs, masks)
            for key in metrics:
                metrics[key] += batch_metrics[key]
    
    num_batches = len(dataloader)
    total_loss /= num_batches
    for key in metrics:
        metrics[key] /= num_batches
    
    return total_loss, metrics

def main():
    # Configuration
    config = {
        'data_root': 'data/DRIVE',
        'model_name': 'unetplusplus',
        'encoder': 'efficientnet-b0',
        'batch_size': 4,  # Reduced for RTX 2050
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'patience': 15,
        'save_dir': 'models'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = DRIVEDataset(
        config['data_root'], 
        split='training', 
        transform=get_transforms('training')
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2
    )
    
    # Model, loss, optimizer
    model = get_model(config['model_name'], config['encoder']).to(device)
    criterion = CombinedLoss()
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    
    # Training loop
    best_dice = 0
    patience_counter = 0
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Metrics: IoU={train_metrics['iou']:.4f}, Dice={train_metrics['dice']:.4f}")
        
        scheduler.step(train_loss)
        
        # Save best model
        if train_metrics['dice'] > best_dice:
            best_dice = train_metrics['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': best_dice,
                'config': config
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break

if __name__ == '__main__':
    main()