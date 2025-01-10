import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image
import os
from datasets import DayNightDataset
from model import EnhancedGenerator, PatchGANDiscriminator, init_weights
import torch.optim as optim
import itertools
from torch.cuda.amp import autocast, GradScaler

class GANLoss(nn.Module):
    def __init__(self, gan_mode='lsgan'):
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        
    def __call__(self, prediction, target_is_real):
        target_tensor = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.loss(prediction, target_tensor)


def train_model(config):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    netG_A2B = EnhancedGenerator(in_channels=3).to(device)
    netG_B2A = EnhancedGenerator(in_channels=3).to(device)
    netD_A = PatchGANDiscriminator(in_channels=3).to(device)
    netD_B = PatchGANDiscriminator(in_channels=3).to(device)
    
    # Initialize weights
    init_weights(netG_A2B)
    init_weights(netG_B2A)
    init_weights(netD_A)
    init_weights(netD_B)
    
    # Setup data loader
    dataset = DayNightDataset(config['day_dir'], config['night_dir'])
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Loss functions
    criterion_GAN = GANLoss(gan_mode='lsgan').to(device)
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    
    # Optimizers
    optimizer_G = optim.Adam(
        itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
        lr=config['lr'],
        betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        itertools.chain(netD_A.parameters(), netD_B.parameters()),
        lr=config['lr'],
        betas=(0.5, 0.999)
    )
    
    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_G, 
        T_max=config['num_epochs'],
        eta_min=config['lr'] * 0.1
    )
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_D,
        T_max=config['num_epochs'],
        eta_min=config['lr'] * 0.1
    )
    
    # Gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Progress bar
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
        
        for i, (real_A, real_B) in enumerate(pbar):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            # Generate identity mappings
            with autocast():
                same_B = netG_A2B(real_B)
                same_A = netG_B2A(real_A)
                
                # Translation
                fake_B = netG_A2B(real_A)
                fake_A = netG_B2A(real_B)
                
                # Cycle
                recovered_A = netG_B2A(fake_B)
                recovered_B = netG_A2B(fake_A)
                
                # Train Generators
                optimizer_G.zero_grad()
                
                # Identity loss
                loss_identity_A = criterion_identity(same_A, real_A) * config['lambda_identity']
                loss_identity_B = criterion_identity(same_B, real_B) * config['lambda_identity']
                
                # GAN loss
                loss_GAN_A2B = criterion_GAN(netD_B(fake_B), True)
                loss_GAN_B2A = criterion_GAN(netD_A(fake_A), True)
                
                # Cycle loss
                loss_cycle_A = criterion_cycle(recovered_A, real_A) * config['lambda_cycle']
                loss_cycle_B = criterion_cycle(recovered_B, real_B) * config['lambda_cycle']
                
                # Total generator loss
                loss_G = (loss_identity_A + loss_identity_B + 
                         loss_GAN_A2B + loss_GAN_B2A + 
                         loss_cycle_A + loss_cycle_B)
            
            # Update generators
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            
            # Train Discriminators
            with autocast():
                optimizer_D.zero_grad()
                
                # Real loss
                loss_D_real_A = criterion_GAN(netD_A(real_A), True)
                loss_D_real_B = criterion_GAN(netD_B(real_B), True)
                
                # Fake loss
                loss_D_fake_A = criterion_GAN(netD_A(fake_A.detach()), False)
                loss_D_fake_B = criterion_GAN(netD_B(fake_B.detach()), False)
                
                # Total discriminator loss
                loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
                loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
                loss_D = loss_D_A + loss_D_B
            
            # Update discriminators
            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            scaler.update()
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f'{loss_G.item():.4f}',
                'D_loss': f'{loss_D.item():.4f}'
            })
            
            # Save intermediate results
            if i % config['save_frequency'] == 0:
                save_image(real_A, fake_B, recovered_A, 
                          real_B, fake_A, recovered_B,
                          epoch, i, config['output_dir'])
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        # Save models
        if (epoch + 1) % config['checkpoint_frequency'] == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'netG_A2B_state_dict': netG_A2B.state_dict(),
                'netG_B2A_state_dict': netG_B2A.state_dict(),
                'netD_A_state_dict': netD_A.state_dict(),
                'netD_B_state_dict': netD_B.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, config['checkpoint_dir'], epoch + 1)

def save_image(real_A, fake_B, recovered_A, real_B, fake_A, recovered_B, epoch, batch, output_dir):
    """Lưu ảnh kết quả trong quá trình training"""
    from torchvision.utils import save_image
    
    # Tạo grid ảnh
    image_grid = torch.cat([
        torch.cat([real_A, fake_B, recovered_A], dim=3),
        torch.cat([real_B, fake_A, recovered_B], dim=3)
    ], dim=2)
    
    # Lưu ảnh
    save_image(image_grid, 
              os.path.join(output_dir, f'epoch_{epoch}_batch_{batch}.png'),
              normalize=True)

def save_checkpoint(state, checkpoint_dir, epoch):
    """Lưu checkpoint model"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))

# Configuration
config = {
    'day_dir': 'day_night_images/train/day',
    'night_dir': 'day_night_images/train/night',
    'batch_size': 4,
    'num_workers': 4,
    'lr': 2e-4,
    'num_epochs': 200,
    'lambda_identity': 5.0,
    'lambda_cycle': 10.0,
    'save_frequency': 100,
    'checkpoint_frequency': 5,
    'output_dir': 'outputs',
    'checkpoint_dir': 'checkpoints'
}

if __name__ == '__main__':
    train_model(config)