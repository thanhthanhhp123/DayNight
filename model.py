import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedGenerator(nn.Module):
    def __init__(self, in_channels=3):
        super(EnhancedGenerator, self).__init__()
        
        # Encoder blocks with skip connections
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        
        # Multi-head self-attention blocks
        self.attention_blocks = nn.Sequential(
            *[MultiHeadSelfAttention(256) for _ in range(9)]
        )
        
        # Decoder blocks with AdaIN and skip connections
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, in_channels, 7, padding=3),
            nn.Tanh()
        )
        
        # AdaIN layers
        self.adain = AdaptiveInstanceNorm2d(256)

    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Apply multi-head self-attention and AdaIN
        x = self.attention_blocks(e3)
        x = self.adain(x)
        
        # Decoder with skip connections
        x = self.dec3(x, e2)
        x = self.dec2(x, e1)
        x = self.dec1(x)
        
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels*2, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, skip):
        x = self.up_sample(x)
        x = torch.cat([x, skip], dim=1)  # Skip connection
        return self.conv(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.InstanceNorm2d(channels)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H*W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        x = (attn @ v).reshape(B, C, H, W)
        x = self.proj(x)
        return self.norm(x)

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)
        
        # Learnable parameters for style modulation
        self.alpha = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        
    def forward(self, x):
        normalized = self.instance_norm(x)
        return self.alpha * normalized + self.beta

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super(PatchGANDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            # Initial layer without normalization
            *discriminator_block(in_channels, ndf, normalize=False),
            # Increasing depth with normalization
            *discriminator_block(ndf, ndf * 2),
            *discriminator_block(ndf * 2, ndf * 4),
            *discriminator_block(ndf * 4, ndf * 8),
            # Final layer for PatchGAN output
            nn.Conv2d(ndf * 8, 1, 4, padding=1)
        )
        
    def forward(self, x):
        # Returns NxNxN patch outputs
        return self.model(x)

def init_weights(model):
    """Initialize network weights using Xavier initialization"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            nn.init.xavier_normal_(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    
    model.apply(init_func)