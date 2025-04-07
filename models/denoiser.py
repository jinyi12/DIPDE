import torch.nn as nn


class KappaDenoiser(nn.Module):
    def __init__(self, latent_dim=128):
        super(KappaDenoiser, self).__init__()

        # Encoder blocks
        self.enc1_down = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.enc1_relu = nn.ReLU(inplace=True)

        self.enc2_down = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enc2_relu = nn.ReLU(inplace=True)

        self.enc3_down = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc3_relu = nn.ReLU(inplace=True)

        self.enc_flat = nn.Flatten()
        # self.enc_linear = nn.Linear(4 * 4 * 128, latent_dim) # for 32 x 32 input
        self.enc_linear = nn.Linear(8 * 8 * 128, latent_dim)  # for 64 x 64 input
        # Decoder blocks
        # self.dec_linear = nn.Linear(latent_dim, 4 * 4 * 128)
        self.dec_linear = nn.Linear(latent_dim, 8 * 8 * 128)  # for 64 x 64 input
        self.dec_unflat = nn.Unflatten(1, (128, 8, 8))

        self.dec3_up = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec3_relu = nn.ReLU(inplace=True)

        self.dec2_up = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec2_relu = nn.ReLU(inplace=True)

        self.dec1_up = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        self.dec1_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Ensure x is of the form (B, 1, H, W)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        # Encoder
        enc = self.enc1_down(x)
        enc = self.enc1_relu(enc)

        enc = self.enc2_down(enc)
        enc = self.enc2_relu(enc)
        enc_skip2 = enc  # For later skip connection

        enc = self.enc3_down(enc)
        enc = self.enc3_relu(enc)
        enc_skip3 = enc  # For later skip connection

        enc = self.enc_flat(enc)
        enc = self.enc_linear(enc)

        # Decoder
        dec = self.dec_linear(enc)
        dec = self.dec_unflat(dec)

        dec = dec + enc_skip3  # Skip connection from encoder block 3
        dec = self.dec3_up(dec)
        dec = self.dec3_relu(dec)

        dec = dec + enc_skip2  # Skip connection from encoder block 2
        dec = self.dec2_up(dec)
        dec = self.dec2_relu(dec)

        dec = self.dec1_up(dec)
        dec = self.dec1_relu(dec)
        return dec
