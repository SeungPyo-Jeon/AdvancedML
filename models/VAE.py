import torch
import torch.nn.functional as F
from torch import autograd, nn, optim

class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        # Fill this 
        self.FFN = nn.Linear( input_dim, hidden_dim)
        self.relu = nn.ReLU( inplace=True)

        self.mu = nn.Linear( hidden_dim, latent_dim)
        self.logvar = nn.Linear( hidden_dim , latent_dim)
    def forward(self, x):
        # Fill this 
        x = self.FFN(x)
        x = self.relu(x)

        mu = self.mu(x)
        logvar = self.logvar(x)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super().__init__()
        # Fill this 
        self.FFN1 = nn.Linear( latent_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.FFN2 = nn.Linear( hidden_dim, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, z):
        # Fill this 
        z = self.FFN1( z ) 
        z = self.relu( z )
        z = self.FFN2( z )
        out = self.sig( z )

        return out

class VAE(nn.Module):
    def __init__(self, in_channels=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, in_channels)

    def reparameterize(self, mu, logvar):
        # Fill this 
        eta = torch.randn_like(mu)
        out = mu + torch.exp(logvar*0.5) * eta
        return out

    def forward(self, x):
        x = x.view(-1, 784)
        # Fill this 
        mu, logvar = self.encoder( x )
        z = self.reparameterize( mu, logvar )
        decoder_output = self.decoder( z )

        return decoder_output, mu, logvar