import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as v2
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from models.GAN import Generator, Discriminator
from engines import train_one_epoch_for_GAN, eval_one_epoch_for_GAN
from utils import save_model_GAN, log_tensorboard, get_tensorboard_writer, generate_fn
from losses import elbo_loss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', default='GAN', type=str)
    parser.add_argument('--data', default='data', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--in_channels', default=784, help = 'fashionMNIST resolution(H*W)', type=int)
    parser.add_argument('--hidden_dim', default=[1024, 512, 256, 1], help='Discriminator hidden dim',type=list)
    parser.add_argument('--latent_dim', default=20, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--checkpoint_dir', default='checkpoints/GAN', type=str)
    args = parser.parse_args()
    return args


def main(args):
    # -------------------------------------------------------------------------
    # Set Logger & Checkpoint Dirs
    # -------------------------------------------------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.checkpoint_dir, f'{args.title}.log')
    logging.basicConfig(
        filename=log_path,
        format='%(asctime)s - %(message)s',
        level=logging.INFO, 
    )
    writer = get_tensorboard_writer(args.checkpoint_dir)
    
    # -------------------------------------------------------------------------
    # Data Processing Pipeline
    # -------------------------------------------------------------------------
    train_transform = v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    train_data = FashionMNIST(
        root=args.data,
        train=True,
        download=True,
        transform=train_transform,
    )
    train_loader = DataLoader(
        dataset=train_data, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True,
    )

    val_transform = v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    val_data = FashionMNIST(
        root=args.data, 
        train=False,
        download=True, 
        transform=val_transform,
    )
    val_loader = DataLoader(
        dataset=val_data, 
        batch_size=args.batch_size,
        shuffle=False,
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    generator = Generator(latent_dim = args.latent_dim,
                          output_dim = args.in_channels
                          ).to(args.device)

    discriminator = Discriminator(in_channels = args.in_channels,).to(args.device)

    # -------------------------------------------------------------------------
    # Performance Metic, Loss Function, Optimizer
    # -------------------------------------------------------------------------    
    loss_fn = nn.BCELoss()
    
    G_optimizer = optim.AdamW(
        params=generator.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
    )

    D_optimizer = optim.AdamW(
        params=discriminator.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
    )

    # -------------------------------------------------------------------------
    # Run Main Loop
    # -------------------------------------------------------------------------
    best_val_loss = 1000.0
    best_val_epoch = 0

    for epoch in range(args.epochs):
        # Train one epoch
        train_summary = train_one_epoch_for_GAN(
            generator=generator,
            discriminator=discriminator,
            dataloader=train_loader,
            criterion=loss_fn,
            optG=G_optimizer,
            optD=D_optimizer,
            device=args.device,
            latent_dim=args.latent_dim,
        )

        # Evaluate one epoch (optional - G only)
        val_summary = eval_one_epoch_for_GAN(
            generator=generator,
            discriminator=discriminator,
            dataloader=val_loader,
            criterion=loss_fn,
            device=args.device,
            latent_dim=args.latent_dim,
        )

        # Save the last model
        checkpoint_path = f'{args.checkpoint_dir}/{args.title}_last.pt'
        save_model_GAN(
                path=checkpoint_path,
                generator=generator,
                discriminator=discriminator,
                optG=G_optimizer,
                optD=D_optimizer,
                epoch=epoch+1,
            )

        # Save the best model
        if val_summary['lossG'] < best_val_loss:
            best_val_epoch = epoch + 1
            best_val_loss = val_summary['lossG']
            checkpoint_path = f'{args.checkpoint_dir}/{args.title}_best.pt'
            save_model_GAN(
                path=checkpoint_path,
                generator=generator,
                discriminator=discriminator,
                optG=G_optimizer,
                optD=D_optimizer,
                epoch=epoch+1,
            )

        # Write tensorboard logs
        log_tensorboard(writer, epoch, train_summary, val_summary)
        
        # Write logs
        log = (f'epoch {epoch+1}, '
               + f'train_lossD: {train_summary["lossD"]:.4f}, '
               + f'train_lossG: {train_summary["lossG"]:.4f}, '
               + f'val_loss: {val_summary["lossG"]:.4f}, '
               + f'best_val_epoch: {best_val_epoch}, ')
        logging.info(log)
        print(log)
        if epoch % 5 == 0:
            n_samples = 4
            image_dims = (args.in_channels, 28, 28)
            samples = generate_fn(model = generator,
                                  model_type = args.title, 
                                  n_samples = n_samples, 
                                  image_dims = image_dims, 
                                  device = args.device, 
                                  latent_dim = args.latent_dim)
            # 이미지 저장
            grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
            save_path = os.path.join(args.checkpoint_dir, f"{args.title}_samples_{epoch}.png")
            torchvision.utils.save_image(grid, save_path)
    # Generate image 
    best_model_path = f'{args.checkpoint_dir}/{args.title}_best.pt'
    checkpoint = torch.load(best_model_path, map_location=args.device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()

    n_samples = 16
    image_dims = (args.in_channels, 28, 28)
    samples = generate_fn(model = generator,
                          model_type = args.title, 
                          n_samples = n_samples, 
                          image_dims = image_dims, 
                          device = args.device, 
                          latent_dim = args.latent_dim)

    # 이미지 저장
    grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
    save_path = os.path.join(args.checkpoint_dir, f"{args.title}_samples.png")
    torchvision.utils.save_image(grid, save_path)

    print(f"Sample image saved to {save_path}")


if __name__=="__main__":
    args = get_args()
    main(args)