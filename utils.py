import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm


def save_model(path, model, optimizer, scheduler, epoch):
    state_dict = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(state_dict, path)

def save_model_GAN(path, generator, discriminator, optG, optD, epoch):
    state_dict = {
        'epoch': epoch,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizerG': optG.state_dict(),
        'optimizerD': optD.state_dict(),
    }
    torch.save(state_dict, path)

def load_model(path, model, optimizer, scheduler, device):
    """
    저장된 체크포인트로부터 모델, 옵티마이저, 스케줄러 상태를 불러옵니다.
    """

    # 체크포인트 로드 (GPU에서 저장한 것을 CPU에서 불러올 경우를 대비해 map_location 사용)
    checkpoint = torch.load(path, map_location=device)
    
    # 모델, 옵티마이저, 스케줄러의 상태(state_dict)를 불러오기
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # 스케줄러가 있는 경우에만 불러오기
    if scheduler is not None and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
        
    # 마지막으로 저장된 epoch 다음부터 시작하도록 epoch 번호 가져오기
    start_epoch = checkpoint['epoch']
    
    print(f"** Checkpoint loaded from {path}. Resuming from epoch {start_epoch}.")
    
    return model, optimizer, scheduler, start_epoch

def get_tensorboard_writer(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def log_tensorboard(writer, epoch, train_summary, val_summary):
    for key in train_summary:
        writer.add_scalar(f'Loss/train/{key}', train_summary[key], epoch)

    for key in val_summary:
        writer.add_scalar(f'Loss/val/{key}', val_summary[key], epoch)


@torch.no_grad()
def generate_fn(model, model_type, n_samples, image_dims, device, preprocess_fn=None, n_bits=None, h=None, latent_dim=100):
    """
    Args:
        model: Generator
        model_type: str, 'pixelcnn', 'vae', 'gan'
        n_samples: a number of image
        image_dims: (C, H, W)
        device: torch.device
        preprocess_fn: PixelCNN normalize func
        n_bits: PixelCNN n bits
        h: conditional vector
        latent_dim: VAE/GAN latent dim
    """
    model.eval()

    if model_type.lower() == 'pixelcnn':
        assert n_bits is not None and preprocess_fn is not None, "PixelCNN requires n_bits and preprocess_fn"
        C, H, W = image_dims
        levels = 2 ** n_bits
        out = torch.zeros(n_samples, C, H, W, dtype=torch.long, device=device)

        with tqdm(total=(C * H * W), desc=f'PixelCNN Generating {n_samples} samples') as pbar:
            for y in range(H):
                for x in range(W):
                    for c in range(C):
                        logits = model(out.float() / (levels - 1), h)
                        probs = F.softmax(logits[:, :, c, y, x], dim=1)
                        sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
                        out[:, c, y, x] = sampled
                        pbar.update()
        return preprocess_fn(out, n_bits)

    elif model_type.lower() in ['vae', 'gan']:
        z = torch.randn(n_samples, latent_dim, device=device)
        if h is not None:
            samples = model(z, h) 
        else:
            samples = model(z)
        samples = samples.view(-1, 1, 28, 28)
        return samples

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'pixelcnn', 'vae', or 'gan'.")