import torch
from torch.cuda.amp import autocast, GradScaler
from torchmetrics.aggregation import MeanMetric
from tqdm import tqdm

def train_one_epoch_for_VAE(model, loader, loss_fn, optimizer, scheduler, device):
    model.train()
    
    loss_epoch = MeanMetric()
    bce_epoch = MeanMetric()
    kl_epoch = MeanMetric()

    for inputs, _ in loader:
        inputs = inputs.to(device).view(inputs.size(0), -1)  # [B, 1, 28, 28] → [B, 784]

        optimizer.zero_grad()
        x_recon, mu, logvar = model(inputs)
        loss, bce, kl = loss_fn(inputs, x_recon, mu, logvar)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_epoch.update(loss.detach().cpu())
        bce_epoch.update(bce.detach().cpu())
        kl_epoch.update(kl.detach().cpu())

    summary = {
        'loss': loss_epoch.compute(),
        'kl': kl_epoch.compute(),
        'bce': bce_epoch.compute(),
    }
    return summary

def eval_one_epoch_for_VAE(model, loader, loss_fn, device):
    model.eval()

    loss_epoch = MeanMetric()
    bce_epoch = MeanMetric()
    kl_epoch = MeanMetric()
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device).view(inputs.size(0), -1)
            x_recon, mu, logvar = model(inputs)
            loss, bce, kl = loss_fn(inputs, x_recon, mu, logvar)

        loss_epoch.update(loss.detach().cpu())
        bce_epoch.update(bce.detach().cpu())
        kl_epoch.update(kl.detach().cpu())

    summary = {
        'loss': loss_epoch.compute(),
        'kl': kl_epoch.compute(),
        'bce': bce_epoch.compute(),
    }
    return summary


def train_one_epoch_for_GAN(generator, discriminator, dataloader, criterion, optG, optD, device, latent_dim=100):
    generator.train()
    discriminator.train()

    lossG_epoch = MeanMetric()
    lossD_epoch = MeanMetric()

    for real_images, _ in dataloader:
        real_images = real_images.to(device)  # (B, 1, 28, 28)
        batch_size = real_images.size(0)

        # === Train Discriminator ===
        # Step 1: Reset gradients
        optD.zero_grad()
        
        # Step 2: Compute loss for real images (label = 1)
        real_label = torch.ones(batch_size, device=device)
        loss_real_for_dis = criterion( 
            discriminator(real_images.view(batch_size,-1)),
            real_label)
        
        # Step 3: Generate fake images and labels
        noise = torch.randn( batch_size, latent_dim, device=device )
        fake_images = generator( noise )

        # Step 4: Compute loss for fake images (label = 0)
        fake_label = torch.zeros( batch_size, device=device)
        loss_fake_for_dis = criterion(  discriminator(fake_images.view(batch_size, -1)), fake_label )

        # Step 5: Combine losses and update D
        lossD = loss_real_for_dis + loss_fake_for_dis
        lossD.backward()
        optD.step()

        # Fill this

        # === Train Generator ===
        # Step 1: Reset gradients
        optG.zero_grad()

        # Step 2: Generate fake images
        noise2 = torch.randn( batch_size, latent_dim, device=device )
        gener_images = generator( noise2 )

        # Step 3: Compute loss for fooling discriminator (label = 1)
        tensor_label = torch.ones( batch_size, device=device)
        lossG = criterion(  discriminator( gener_images.view(batch_size, -1)) , tensor_label )

        # Step 4: Update G
        lossG.backward()
        optG.step()
        # Fill this 

        lossD_epoch.update(lossD.detach().cpu())
        lossG_epoch.update(lossG.detach().cpu())

    return {
        'lossD': lossD_epoch.compute(),
        'lossG': lossG_epoch.compute()
    }

def eval_one_epoch_for_GAN(generator, discriminator, dataloader, criterion, device, latent_dim=100):
    generator.eval()
    discriminator.eval()

    lossG_epoch = MeanMetric()
    with torch.no_grad():
        for real_images, _ in dataloader:
            batch_size = real_images.size(0)
            noise = torch.randn(batch_size, latent_dim, device=device) 
            
            fake_images = generator(noise)  # (B, 1, 28, 28)
            output = discriminator(fake_images.view(batch_size, -1)) 
            real_labels = torch.ones(batch_size, device=device)
            lossG = criterion(output, real_labels)
            lossG_epoch.update(lossG.detach().cpu())

    return {
        'lossG': lossG_epoch.compute()
    }

scaler = GradScaler()
def train_one_epoch_for_PixelCNN(model, loader, loss_fn, optimizer, device):
    model.train()
    loss_epoch = MeanMetric()

    for inputs, y in tqdm(loader):
        x = inputs.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(x)
            loss = loss_fn(logits, x, model.n_bits)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_epoch.update(loss.detach().cpu())
        #logits = model(x)
        #loss = loss_fn(logits, x, model.n_bits).mean(0)
        #loss.backward()
        #optimizer.step()
        #loss_epoch.update(loss.detach().cpu())

    summary = {
        'loss': loss_epoch.compute(),
    }
    return summary

def eval_one_epoch_for_PixelCNN(model, loader, loss_fn, device):
    model.eval()

    loss_epoch = MeanMetric()

    with torch.no_grad():
        for inputs, y in tqdm(loader):
            x = inputs.to(device)
            y = y.to(device)
            with autocast():
                logits = model(x)
                loss = loss_fn(logits, x, model.n_bits)
            #logits = model(x)
            #loss = loss_fn(logits, x, model.n_bits).mean(0)

        loss_epoch.update(loss.detach().cpu())

    summary = {
        'loss': loss_epoch.compute(),
    }
    return summary