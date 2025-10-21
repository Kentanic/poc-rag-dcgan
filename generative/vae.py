# generative/vae.py
import os, math, time
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms, utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Encoder(nn.Module):
    def _init_(self, z_dim=16):
        super()._init_()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(128*4*4, z_dim)
        self.fc_logvar = nn.Linear(128*4*4, z_dim)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def _init_(self, z_dim=16):
        super()._init_()
        self.fc = nn.Linear(z_dim, 128*4*4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 128, 4, 4)
        return self.deconv(h)

class VAE(nn.Module):
    def _init_(self, z_dim=16):
        super()._init_()
        self.enc = Encoder(z_dim)
        self.dec = Decoder(z_dim)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparam(mu, logvar)
        recon = self.dec(z)
        return recon, mu, logvar

def make_loader(data_dir, batch_size=128):
    tf = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ds = datasets.FashionMNIST(root=data_dir, train=True, transform=tf, download=True)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

def vae_loss(x, recon, mu, logvar):
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + 0.001*kld, recon_loss, kld

@torch.no_grad()
def sample_grid_vae(model, n=36, z_dim=16, out_path=None):
    model.eval()
    z = torch.randn(n, z_dim, device=DEVICE)
    imgs = model.dec(z).cpu()
    grid = utils.make_grid(imgs, nrow=int(math.sqrt(n)), normalize=True, value_range=(-1,1))
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        utils.save_image(grid, out_path)
    return grid

def train_vae(data_dir="data/gen_data", out_dir="models/vae", z_dim=16, epochs=3, batch_size=128, lr=2e-4):
    os.makedirs(out_dir, exist_ok=True)
    model = VAE(z_dim=z_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = make_loader(data_dir, batch_size=batch_size)

    for ep in range(1, epochs+1):
        t0 = time.time()
        for x, _ in loader:
            x = x.to(DEVICE)
            recon, mu, logvar = model(x)
            loss, rec, kld = vae_loss(x, recon, mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
        torch.save(model.state_dict(), os.path.join(out_dir, "vae.pt"))
        sample_grid_vae(model, n=36, z_dim=z_dim, out_path=os.path.join(out_dir, f"samples_ep{ep}.png"))
        print(f"[VAE] epoch {ep}/{epochs} loss={loss.item():.3f} rec={rec.item():.3f} kld={kld.item():.3f} time={time.time()-t0:.1f}s")
    return os.path.join(out_dir, "vae.pt")