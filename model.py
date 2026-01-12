import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class VesselSegmentationModel(nn.Module):
    def __init__(self, encoder_name='efficientnet-b0', classes=1):
        super().__init__()
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=classes,
            activation=None
        )
        
    def forward(self, x):
        return self.model(x)

def get_model(model_name='unetplusplus', encoder='efficientnet-b0'):
    """Get segmentation model"""
    if model_name == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None
        )
    elif model_name == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None
        )
    elif model_name == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None
        )
    
    return model