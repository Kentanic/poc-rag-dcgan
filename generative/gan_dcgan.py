# generative/gan_dcgan.py
import os, io, zipfile, math, time
from typing import List, Tuple, Optional
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms, utils

# ---------- device ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ---------------- DCGAN models ----------------
class Generator(nn.Module):
    def __init__(self, nz: int = 32, ngf: int = 64, nc: int = 1):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, ndf: int = 64, nc: int = 1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),  # -> [B,1,H,W]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x)            # [B, 1, H, W]
        out = out.mean(dim=(2, 3))    # GAP -> [B, 1]
        return out.squeeze(1)         # [B]


def weights_init(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias.data)

# ---------------- datasets ----------------
def _fashion_loader(batch_size: int = 128, img_size: int = 32, max_samples: int = 2048) -> DataLoader:
    tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    root = "data/torch_data"
    ds_full = datasets.FashionMNIST(root=root, train=True, download=True, transform=tfm)
    if max_samples and max_samples < len(ds_full):
        ds = Subset(ds_full, list(range(max_samples)))
    else:
        ds = ds_full
    bs = min(batch_size, 128)
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)


class ImageFolderGray(Dataset):
    def __init__(self, folder: str, img_size: int = 32):
        self.paths = []
        p = Path(folder)
        if p.exists():
            for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                self.paths.extend(sorted(p.glob(f"*{ext}")))
        self.tfm = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tfm(img)


def _custom_loader(folder: str, batch_size: int = 128, img_size: int = 32) -> DataLoader:
    ds = ImageFolderGray(folder, img_size=img_size)
    if len(ds) == 0:
        raise RuntimeError(f"No images found in {folder}")
    bs = min(batch_size, max(8, len(ds) // 2))
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)

# ---------------- ckpt IO ----------------
def _ckpt_dir() -> str:
    d = "data/generated/dcgan"
    os.makedirs(d, exist_ok=True)
    return d

def ckpt_path(tag: str = "latest") -> str:
    return os.path.join(_ckpt_dir(), f"dcgan_{tag}.pt")

def save_ckpt(G: Generator, D: Discriminator, tag: str = "latest"):
    torch.save({"G": G.state_dict(), "D": D.state_dict()}, ckpt_path(tag))

def load_ckpt(nz: int = 32, nc: int = 1, tag: str = "latest") -> Tuple[Generator, Discriminator]:
    G = Generator(nz=nz, nc=nc).to(DEVICE)
    D = Discriminator(nc=nc).to(DEVICE)
    G.apply(weights_init)
    D.apply(weights_init)
    p = ckpt_path(tag)
    if os.path.exists(p):
        ck = torch.load(p, map_location=DEVICE)
        G.load_state_dict(ck["G"])
        D.load_state_dict(ck["D"])
    return G.eval(), D.eval()

# ---------------- training ----------------
@torch.no_grad()
def _grid_from_gen(G: Generator, n: int = 36, nz: int = 32, seed: int = 0) -> Image.Image:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    z = torch.randn(n, nz, 1, 1, generator=g, device=DEVICE)
    fake = G(z).cpu()
    grid = utils.make_grid(fake, nrow=int(math.sqrt(n)), normalize=True, value_range=(-1, 1))
    return transforms.ToPILImage()(grid)

def train_dcgan(
    dataset: str = "fashion-mnist",
    custom_folder: Optional[str] = None,
    epochs: int = 1,
    lr: float = 2e-4,
    nz: int = 32,
    img_size: int = 32,
    max_batches: int = 20,          # <<< cap batches per epoch for speed
    log_every: int = 10,
) -> str:
    # data
    if dataset == "custom" and custom_folder:
        loader = _custom_loader(custom_folder, img_size=img_size)
    else:
        loader = _fashion_loader(img_size=img_size, max_samples=max_batches*128)

    # models
    G = Generator(nz=nz).to(DEVICE)
    D = Discriminator().to(DEVICE)
    G.apply(weights_init)
    D.apply(weights_init)

    opt_g = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    print(f"[DCGAN] device={DEVICE}, epochs={epochs}, nz={nz}, img_size={img_size}, "
          f"batches/epoch≈{min(max_batches, len(loader))}")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        for b, batch in enumerate(loader):
            if b >= max_batches:
                break
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(DEVICE)
            bs = x.size(0)
            real = torch.ones(bs, device=DEVICE)
            fake = torch.zeros(bs, device=DEVICE)

            # D step
            with torch.no_grad():
                z = torch.randn(bs, nz, 1, 1, device=DEVICE)
                xf = G(z)
            dr = D(x)
            df = D(xf.detach())
            loss_d = bce(dr, real) + bce(df, fake)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # G step
            z = torch.randn(bs, nz, 1, 1, device=DEVICE)
            xf = G(z)
            df = D(xf)
            loss_g = bce(df, real)
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            if (b + 1) % log_every == 0:
                print(f"[ep {ep}] batch {b+1}/{min(max_batches, len(loader))} "
                      f"loss_d={loss_d.item():.3f} loss_g={loss_g.item():.3f}")

        # save ckpt + quick preview each epoch
        save_ckpt(G, D, tag="latest")
        preview = _grid_from_gen(G, n=36, nz=nz, seed=ep)
        preview_path = os.path.join(_ckpt_dir(), "preview.png")
        preview.save(preview_path)
        print(f"[ep {ep}] saved ckpt + preview ({int((time.time()-t0)*1000)} ms) -> {preview_path}")

    return ckpt_path("latest")

# ---------------- generation API for UI ----------------
@torch.no_grad()
def sample_grid(n: int = 36, seed: int = 0, nz: int = 32) -> Image.Image:
    G, _ = load_ckpt(nz=nz, tag="latest")
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    z = torch.randn(n, nz, 1, 1, generator=g, device=DEVICE)
    fake = G(z).cpu()
    grid = utils.make_grid(fake, nrow=int(math.sqrt(n)), normalize=True, value_range=(-1, 1))
    return transforms.ToPILImage()(grid)

@torch.no_grad()
def generate_images(n: int = 36, seed: int = 0, nz: int = 32) -> Tuple[List[Image.Image], bytes]:
    G, _ = load_ckpt(nz=nz, tag="latest")
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    z = torch.randn(n, nz, 1, 1, generator=g, device=DEVICE)
    fake = G(z).cpu()

    ims: List[Image.Image] = []
    for i in range(n):
        im = transforms.ToPILImage()(utils.make_grid(fake[i], normalize=True, value_range=(-1, 1)))
        ims.append(im)

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, im in enumerate(ims):
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            buf.seek(0)
            zf.writestr(f"gan_{i:03d}.png", buf.read())
    mem.seek(0)
    return ims, mem.read()