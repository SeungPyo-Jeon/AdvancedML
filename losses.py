import torch
from torch.nn import functional as F

def elbo_loss(x, recon_x, mu, logvar):
    """
    Args:
        x : Generator output ; Shape: (B, C, H, W)
        recon_x : Reconstruction target ; Shape: (B, C, H, W)
        mu : Mean vector of the latent distribution ; Shape: (B, latent_dim)
        logvar : Log variance of the latent distribution. ; Shape: (B, latent_dim)

    Returns:
        BCE : Reconstruction error between recon_x and x
        KLD : Kullback–Leibler divergence.
            - Measures how much the learned latent distribution q(z|x) diverges from the prior p(z)
              (usually standard normal).
        total_loss : The total ELBO loss (negative ELBO).
    """ 
    B = x.size(0)
    # Fill this 
    BCE = F.mse_loss( recon_x, x, reduction='sum')/B
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    KLD = torch.mean(KLD)
    total_loss = BCE+ KLD

    return total_loss, BCE, KLD

def PixelCNN_loss(logits, targets, n_bits):
    """
    Args:
        logits: Model output logits for each quantized pixel level. ; Shape: (B, 2**n_bits, C, H, W)
        targets: Ground truth image tensor. ; Shape: (B, C, H, W)
        n_bits: Number of bits used for quantizing pixel values.

    Returns:
        loss: Scalar reconstruction loss (cross entropy averaged over all pixels).
    """
    # Fill this
    num_bins = 2 ** n_bits
    logits = logits.permute(0, 2, 3, 4, 1).reshape(-1, num_bins)
    targets = (targets * (num_bins - 1)).long().reshape(-1)
    loss = F.cross_entropy(logits, targets)
    #print( loss.shape, loss )
    return loss