import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

seed = 99
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim[0])
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(mlp_dim[0], mlp_dim[1])
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(mlp_dim[1], input_dim)

    def forward(self, x):
        # [B, L, D] or [B, D, L]
        return self.fc3(self.tanh2(self.fc2(self.tanh1(self.fc1(x)))))

class FactorizedChannelMixing(nn.Module):
    def __init__(self, input_dim, factorized_dim) :
        super().__init__()

        for dim in factorized_dim:
            assert input_dim > dim
        self.channel_mixing = MLPBlock(input_dim, factorized_dim)

    def forward(self, x):

        return self.channel_mixing(x)

# FTCF Block
class FTCFBlock(nn.Module):
    def __init__(self, configs, tokens_dim, channels_dim, tokens_hidden_dim, channels_hidden_dim, fac_C, norm_flag):
        super().__init__()
        self.configs = configs
        self.tokens_mixing = MLPBlock(tokens_dim, tokens_hidden_dim)
        self.channels_mixing = FactorizedChannelMixing(channels_dim, channels_hidden_dim) if fac_C else None
        self.norm = nn.LayerNorm(channels_dim) if norm_flag else None

    def forward(self,x):
        # 1. token-mixing [B, D, #tokens]
        y = self.norm(x) if self.norm else x
        y = self.tokens_mixing(y.transpose(1, 2)).transpose(1, 2)   # TemporalMixing

        # 2. channel-mixing [B, #tokens, D]
        if self.channels_mixing:
            if self.configs.short_res:
                y += x  # short_res
                res = y
                y = self.norm(y) if self.norm else y
                y = res + self.channels_mixing(y)   # short_res
            else:
                y = self.channels_mixing(y)   # without short_res

        return y

# Encoder
class Encoder(nn.Module):
    def __init__(self, configs, sequence_length=30, input_dim=5, latent_dim=2):
        super(Encoder, self).__init__()
        self.configs = configs
        self.input_dim = input_dim
        self.hidden_num = [210, 90]   # MLP of time_step
        self.channels_hidden_dim = [2, 7] # MLP of channel

        self.mlp_blocks = nn.ModuleList([
            FTCFBlock(configs, sequence_length, input_dim, self.hidden_num, self.channels_hidden_dim, configs.fac_C, configs.norm) for _ in range(configs.e_layers)
        ])
        self.dropout = nn.Dropout(0.5)
        if self.configs.long_res:
            self.fc_mu = nn.Sequential(nn.Linear(sequence_length * input_dim * 2, sequence_length * input_dim // 2),
                                       nn.Tanh(), nn.Dropout(0.2),
                                       nn.Linear(sequence_length * input_dim // 2, latent_dim))
            self.fc_logvar = nn.Sequential(nn.Linear(sequence_length * input_dim * 2, sequence_length * input_dim // 2),
                                        nn.Tanh(), nn.Dropout(0.2),
                                        nn.Linear(sequence_length * input_dim // 2, latent_dim))
        else:
            self.fc_mu = nn.Sequential(nn.Linear(sequence_length * input_dim, sequence_length * input_dim // 2),
                                       nn.Tanh(), nn.Dropout(0.2),
                                       nn.Linear(sequence_length * input_dim // 2, latent_dim))
            self.fc_logvar = nn.Sequential(nn.Linear(sequence_length * input_dim, sequence_length * input_dim // 2),
                                        nn.Tanh(), nn.Dropout(0.2),
                                        nn.Linear(sequence_length * input_dim // 2, latent_dim))

    def forward(self, x):
        # shape(x) = (batch_size, seq_len=30, input_dim=14)
        # 1. Mixer
        mixer_x = x
        for block in self.mlp_blocks:
            mixer_x = block(mixer_x)

        # 2. residual
        res = x.transpose(1, 2).reshape(x.size(0), -1)

        # 4. Sample mu and var
        faltten_x = mixer_x.transpose(1, 2).reshape(mixer_x.size(0), -1)
        if self.configs.long_res:
            faltten_x = torch.cat([res, faltten_x], dim=-1) # concat LongRes
        faltten_x = self.dropout(faltten_x) # dp=0.5
        mu = self.fc_mu(faltten_x)  # shape(batch_size, latent_size)
        logvar = self.fc_logvar(faltten_x)  # shape(batch_size, latent_size)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(Decoder, self).__init__()
        self.regressor = nn.Sequential(*[nn.Linear(latent_dim, 200),
                                         nn.Tanh(),
                                         nn.Linear(200, 1)])

    def forward(self, z):
        out = self.regressor(z)
        return out

def reparameterize(mu, logvar):
    """
	Reparameterization trick to sample from N(mu, var) from
	N(0,1).
	:param mu: (Tensor) Mean of the latent Gaussian [B x D]
	:param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
	:return: (Tensor) [B x D]
	"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

class FFVE(nn.Module):
    def __init__(self, configs, sequence_length=30, input_dim=14, latent_dim=3):
        super(FFVE, self).__init__()
        self.encoder = Encoder(configs=configs, sequence_length=sequence_length, input_dim=input_dim,
                          hidden_dim=300, latent_dim=latent_dim).to(configs.device)
        self.decoder = Decoder(latent_dim=latent_dim).to(configs.device)
    def forward(self, x):
        # encoder
        mu, var = self.encoder(x)
        # reparameterize
        z = reparameterize(mu, var).float()
        # decoder
        out = self.decoder(z).view(-1)
        return out

def total_loss(out, tr_y, mu, var):
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    """
    kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1), dim=0)
    rmse_loss = torch.sqrt(F.mse_loss(out, tr_y) + 1e-6)
    loss = kl_loss + rmse_loss
    return loss, kl_loss, rmse_loss

def total_loss_forTest(out, tr_y, mu, var):
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    """
    kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1), dim=0)
    rmse_loss = F.mse_loss(out, tr_y) + 1e-6
    loss = kl_loss + rmse_loss
    return loss, kl_loss, rmse_loss

if __name__ == '__main__':
    device = 'cuda:0'
    input = torch.ones([128, 30, 5]).to(device)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    mu, var = encoder(input)
    z = reparameterize(mu, var)
    out = decoder(z)
