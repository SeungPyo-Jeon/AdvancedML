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
from models.AR import PixelCNN
from engines import train_one_epoch_for_PixelCNN, eval_one_epoch_for_PixelCNN
from utils import save_model, log_tensorboard, get_tensorboard_writer, generate_fn, load_model
from losses import PixelCNN_loss
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', default='PixelCNN', type=str)
    parser.add_argument('--data', default='data', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--kernel_size', default=5, type=int)
    parser.add_argument('--output_dim', default = 1024, type=int)
    parser.add_argument('--n_bit', default = 8, type=int)
    parser.add_argument('--n_res_layers', default = 6, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--checkpoint_dir', default='checkpoints/PixelCNN', type=str)
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
        num_workers=10
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
        num_workers=10
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = PixelCNN(
        in_channels = args.in_channels,
        n_bits=args.n_bit,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        kernel_size=args.kernel_size,
        n_res_layers=args.n_res_layers,
        norm_layer=True
    ).to(args.device)

    # -------------------------------------------------------------------------
    # Performance Metic, Loss Function, Optimizer
    # -------------------------------------------------------------------------    
    loss_fn = PixelCNN_loss
    
    optimizer = torch.optim.RMSprop(
        params=model.parameters(),lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    # -------------------------------------------------------------------------
    # Run Main Loop
    # -------------------------------------------------------------------------
    best_val_loss = 1000.0
    best_val_epoch = 0
    load_model( f'checkpoints/PixelCNN_ver4/{args.title}_best.pt', model, optimizer, scheduler, args.device)
    for epoch in range(args.epochs):
        continue
        # Train one epoch
        train_summary = train_one_epoch_for_PixelCNN(
            model=model, 
            loader=train_loader, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            device=args.device,
        )

        # Evaluate one epoch (optional - G only)
        val_summary = eval_one_epoch_for_PixelCNN(
            model=model, 
            loader=val_loader, 
            loss_fn=loss_fn, 
            device=args.device,
        )

        # Save the last model
        checkpoint_path = f'{args.checkpoint_dir}/{args.title}_last.pt'
        save_model(checkpoint_path, model, optimizer, scheduler, epoch+1)


        # Save the best model
        if val_summary['loss'] < best_val_loss:
            best_val_epoch = epoch + 1
            best_val_accuracy = val_summary['loss']
            checkpoint_path = f'{args.checkpoint_dir}/{args.title}_best.pt'
            save_model(checkpoint_path, model, optimizer, scheduler, best_val_epoch)

        # Write tensorboard logs
        log_tensorboard(writer, epoch, train_summary, val_summary)

        # Write logs
        log = (f'epoch {epoch+1}, '
               + f'train_loss: {train_summary["loss"]:.4f}, '
               + f'val_loss: {val_summary["loss"]:.4f}, '
               + f'best_val_epoch: {best_val_epoch}, ')
        logging.info(log)
        print(log)
        if epoch % 1 == 0:
            n_samples = 4
            image_dims = (args.in_channels, 28, 28)
            preprocess_fn = lambda sample, n_bits: sample.to(torch.float32) / (2**n_bits - 1)
            samples = generate_fn(model, args.title, n_samples, image_dims, args.device, preprocess_fn, args.n_bit)
            # 이미지 저장
            grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
            save_path = os.path.join(args.checkpoint_dir, f"{args.title}_samples_{epoch}.png")
            torchvision.utils.save_image(grid, save_path)

    # Generate image 
    best_model_path = f'{args.checkpoint_dir}/{args.title}_best.pt'
    checkpoint = torch.load(best_model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    n_samples = 16
    image_dims = (args.in_channels, 28, 28)
    preprocess_fn = lambda sample, n_bits: sample.to(torch.float32) / (2**n_bits - 1)
    samples = generate_fn(model, args.title, n_samples, image_dims, args.device, preprocess_fn, args.n_bit)

    # 이미지 저장
    grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
    save_path = os.path.join(args.checkpoint_dir, f"{args.title}_samples.png")
    torchvision.utils.save_image(grid, save_path)

    print(f"Sample image saved to {save_path}")


if __name__=="__main__":
    args = get_args()
    main(args)