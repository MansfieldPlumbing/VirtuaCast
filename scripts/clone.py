# clone.py
# This script exports the PyTorch model to ONNX in an unoptimized "training" mode.
# This prevents operator fusion (e.g., Conv+BatchNorm) and creates a graph that
# is a near 1-to-1 clone of the PyTorch architecture, making it ideal for debugging
# and blueprint comparison.

import torch
import torch.nn as nn
import os
import sys
from collections import OrderedDict

# --- Configuration ---
MODEL_FILENAME = "..\models\student_128.pth"
# Use a new name for the unoptimized ONNX file to avoid confusion
ONNX_CLONE_FILENAME = "..\models\student_128_clone.onnx"
INPUT_IMAGE_SIZE = 128
EMBEDDING_SIZE = 512

# ==============================================================================
# === MODEL ARCHITECTURE (Identical to convert.py)                         ===
# ==============================================================================

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(OrderedDict([
            ('0', nn.Linear(channel, channel // reduction, bias=False)),
            ('1', nn.ReLU(inplace=True)),
            ('2', nn.Linear(channel // reduction, channel, bias=False)),
            ('3', nn.Sigmoid())
        ]))
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Bottleneck(nn.Module):
    def __init__(self, channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Module()
        self.conv1.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.conv1.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.norm1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Module()
        self.conv2.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.conv2.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.norm2 = nn.BatchNorm2d(channels)
        self.se = SELayer(channels)

    def forward(self, x):
        identity = x
        out = self.conv1.depthwise(x)
        out = self.conv1.pointwise(out)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2.depthwise(out)
        out = self.conv2.pointwise(out)
        out = self.norm2(out)
        out = self.se(out)
        out += identity
        out = self.relu(out)
        return out

class FaceSwapModel(nn.Module):
    def __init__(self, in_channels=3, embedding_size=512):
        super(FaceSwapModel, self).__init__()
        
        self.enc1 = nn.Sequential(OrderedDict([('0', nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)), ('1', nn.BatchNorm2d(64))]))
        self.enc2 = nn.Sequential(OrderedDict([('0', nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True)), ('1', nn.BatchNorm2d(128))]))
        self.enc3 = nn.Sequential(OrderedDict([('0', nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True)), ('1', nn.BatchNorm2d(256))]))
        self.enc4 = nn.Sequential(OrderedDict([('0', nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=True)), ('1', nn.BatchNorm2d(512))]))
        self.enc5 = nn.Sequential(OrderedDict([('0', nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=True)), ('1', nn.BatchNorm2d(1024))]))
        
        self.bottleneck = nn.Sequential(*(Bottleneck(1024) for _ in range(4)))

        self.conditioning_network = nn.Module()
        self.conditioning_network.fc1 = nn.Linear(embedding_size, 512, bias=True)
        self.conditioning_network.fc2 = nn.Linear(512, 16384, bias=True)
        
        self.dec1 = nn.Module(); self.dec1.conv = nn.Conv2d(1024 + 512, 512, 3, 1, 1, bias=True); self.dec1.norm = nn.BatchNorm2d(512)
        self.dec2 = nn.Module(); self.dec2.conv = nn.Conv2d(512 + 256, 256, 3, 1, 1, bias=True); self.dec2.norm = nn.BatchNorm2d(256)
        self.dec3 = nn.Module(); self.dec3.conv = nn.Conv2d(256 + 128, 128, 3, 1, 1, bias=True); self.dec3.norm = nn.BatchNorm2d(128)
        self.dec4 = nn.Module(); self.dec4.conv = nn.Conv2d(128 + 64, 64, 3, 1, 1, bias=True); self.dec4.norm = nn.BatchNorm2d(64)
        
        self.output_conv = nn.Sequential(OrderedDict([
            ('0', nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)),
            ('1', nn.ReLU(inplace=True)),
            ('2', nn.Conv2d(64, in_channels, kernel_size=1, padding=0, bias=True))
        ]))

        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, input_image, input_embedding):
        enc1_out = self.relu(self.enc1(input_image))
        enc2_out = self.relu(self.enc2(enc1_out))
        enc3_out = self.relu(self.enc3(enc2_out))
        enc4_out = self.relu(self.enc4(enc3_out))
        enc5_out = self.relu(self.enc5(enc4_out))
        
        bottleneck_out = self.bottleneck(enc5_out)
        
        cond = self.relu(self.conditioning_network.fc1(input_embedding))
        cond = self.conditioning_network.fc2(cond)
        cond = cond.view(-1, 1024, 4, 4)
        cond = self.upsample(cond)

        conditioned = bottleneck_out + cond
        
        dec1_in = torch.cat([self.upsample(conditioned), enc4_out], dim=1); dec1_out = self.relu(self.dec1.norm(self.dec1.conv(dec1_in)))
        dec2_in = torch.cat([self.upsample(dec1_out), enc3_out], dim=1);  dec2_out = self.relu(self.dec2.norm(self.dec2.conv(dec2_in)))
        dec3_in = torch.cat([self.upsample(dec2_out), enc2_out], dim=1);   dec3_out = self.relu(self.dec3.norm(self.dec3.conv(dec3_in)))
        dec4_in = torch.cat([self.upsample(dec3_out), enc1_out], dim=1);   dec4_out = self.relu(self.dec4.norm(self.dec4.conv(dec4_in)))
        
        return self.output_conv(dec4_out)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FILENAME)
    onnx_path = os.path.join(script_dir, ONNX_CLONE_FILENAME)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'"); sys.exit(1)

    print(f"Loading state dictionary from '{model_path}'...")
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    print("Initializing model architecture...")
    model = FaceSwapModel()

    print("Loading state dictionary with non-strict checking...")
    model.load_state_dict(state_dict, strict=False)
    
    # --- KEY CHANGE FOR DEBUGGING ---
    # 1. Put the model in TRAINING mode instead of EVAL mode.
    # This prevents PyTorch from treating BatchNorm as a simple foldable operation.
    model.train()

    dummy_image = torch.randn(1, 3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
    dummy_embedding = torch.randn(1, EMBEDDING_SIZE)

    print(f"Exporting UNOPTIMIZED model to ONNX at '{onnx_path}'...")
    try:
        torch.onnx.export(
            model,
            (dummy_image, dummy_embedding),
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=False,
            input_names=['target', 'source'],
            output_names=['output'],
            dynamic_axes={
                'target': {0: 'batch_size'},
                'source': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            # 2. Explicitly tell the exporter we are in training mode.
            # This ensures BatchNorm and other training-specific layers are
            # exported as separate, non-fused operators.
            training=torch.onnx.TrainingMode.TRAINING
        )
        print("\n" + "="*60)
        print(f"SUCCESS: Unoptimized model exported to '{onnx_path}'")
        print("="*60 + "\n")
    except Exception as e:
        print(f"Error exporting model: {e}"); sys.exit(1)

if __name__ == '__main__':
    main()