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
from model import AttentionGenerator, AttentionDiscriminator

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DayNightDataset(args.day_dir, args.night_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    G_day2night = AttentionGenerator().to(device)
    G_night2day = AttentionGenerator().to(device)
    D_day = AttentionDiscriminator().to(device)
    D_night = AttentionDiscriminator().to(device)

    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()    

    optimizer_G = torch.optim.Adam(
        list(G_day2night.parameters()) + list(G_night2day.parameters()),
        lr=args.lr, betas=(args.beta1, args.beta2)
    )

    optimizer_D = torch.optim.Adam(
        list(D_day.parameters()) + list(D_night.parameters()),
        lr=args.lr, betas=(args.beta1, args.beta2)
    )

    for epoch in tqdm(range(args.epochs)):
        for i, (real_day, real_night) in enumerate(dataloader):
            real_night = real_night.to(device)
            real_day = real_day.to(device)

            fake_night = G_day2night(real_day)
            fake_day = G_night2day(real_night)

            recovered_day = G_night2day(fake_night)
            recovered_night = G_day2night(fake_day)

            identity_day = G_night2day(real_day)
            identity_night = G_day2night(real_night)

            optimizer_G.zero_grad()

            loss_identity = (criterion_identity(identity_day, real_day) + 
                           criterion_identity(identity_night, real_night)) * 5.0
            
            loss_GAN = (criterion_GAN(D_night(fake_night), torch.ones_like(D_night(fake_night))) +
                       criterion_GAN(D_day(fake_day), torch.ones_like(D_day(fake_day))))
            loss_cycle = (criterion_cycle(recovered_day, real_day) + 
                         criterion_cycle(recovered_night, real_night)) * 10.0
            
            loss_G = loss_GAN + loss_cycle + loss_identity
            loss_G.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            loss_D_day = (criterion_GAN(D_day(real_day), torch.ones_like(D_day(real_day))) +
                         criterion_GAN(D_day(fake_day.detach()), torch.zeros_like(D_day(fake_day))))
            
            # Night discriminator
            loss_D_night = (criterion_GAN(D_night(real_night), torch.ones_like(D_night(real_night))) +
                           criterion_GAN(D_night(fake_night.detach()), torch.zeros_like(D_night(fake_night))))
            
            loss_D = (loss_D_day + loss_D_night) * 0.5
            loss_D.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Batch [{i}] "
                      f"Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")
                
        if (epoch + 1) % 10 == 0:
            torch.save({
                'G_day2night': G_day2night.state_dict(),
                'G_night2day': G_night2day.state_dict(),
                'D_day': D_day.state_dict(),
                'D_night': D_night.state_dict()
            }, f'checkpoint_epoch_{epoch+1}.pth')

def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    G_day2night = AttentionGenerator().to(device)
    G_night2day = AttentionGenerator().to(device)
    checkpoint = torch.load(args.checkpoint)
    G_day2night.load_state_dict(checkpoint['G_day2night'])
    G_night2day.load_state_dict(checkpoint['G_night2day'])

    G_day2night.eval()
    G_night2day.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    day_images = os.listdir(args.day_dir)
    night_images = os.listdir(args.night_dir)

    os.makedirs('results/day2night', exist_ok=True)
    os.makedirs('results/night2day', exist_ok=True)

    with torch.no_grad():
        for img_name in day_images:
            img_path = os.path.join(args.day_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)

            fake_night = G_day2night(img)
            save_image(fake_night, f'results/day2night/{img_name}')

        for img_name in night_images:
            img_path = os.path.join(args.night_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)

            fake_day = G_night2day(img)
            save_image(fake_day, f'results/night2day/{img_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--day_dir', type=str, required=True, default='day_night_images/training/day')
    parser.add_argument('--night_dir', type=str, required=True, default='day_night_images/training/night')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()
    train(args)
                