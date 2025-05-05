import argparse
import math
import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

# ----------------------- Utility Functions -----------------------

def exists(x):
    return x is not None

def gn(ch: int, max_groups: int = 8):
    groups = max(1, min(max_groups, ch))
    while ch % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, ch)

# cosine beta schedule

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-5, 0.999)

# ----------------------- UNet backbone -----------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, cond_dim=0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.cond_proj = nn.Linear(cond_dim, out_ch) if cond_dim else None
        self.block1 = nn.Sequential(gn(in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.block2 = nn.Sequential(gn(out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.res_skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t, y=None):
        h = self.block1(x)
        h = h + self.mlp(t)[:, :, None, None]
        if exists(y) and self.cond_proj is not None:
            h = h + self.cond_proj(y)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_skip(x)

class UNet(nn.Module):
    """Simplified UNet for 128×128 images."""

    def __init__(self, in_ch=3, base_ch=64, ch_mults=(1, 2, 4, 8), time_emb_dim=512, cond_dim=0):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Downsampling blocks
        ch_in = in_ch
        down_channels = []  # store feature map channels for skip connections
        for mult in ch_mults:
            ch_out = base_ch * mult
            self.downs.append(nn.ModuleList([
                ResidualBlock(ch_in, ch_out, time_emb_dim, cond_dim),
                ResidualBlock(ch_out, ch_out, time_emb_dim, cond_dim),
                nn.Conv2d(ch_out, ch_out, 4, 2, 1)  # stride‑2 downsample
            ]))
            ch_in = ch_out
            down_channels.append(ch_out)

        # Bottleneck
        self.mid = nn.ModuleList([
            ResidualBlock(ch_in, ch_in, time_emb_dim, cond_dim),
            ResidualBlock(ch_in, ch_in, time_emb_dim, cond_dim)
        ])

        # Upsampling blocks
        for mult in reversed(ch_mults):
            ch_out = base_ch * mult
            self.ups.append(nn.ModuleList([
                ResidualBlock(ch_in + down_channels.pop(), ch_out, time_emb_dim, cond_dim),
                ResidualBlock(ch_out, ch_out, time_emb_dim, cond_dim),
                nn.ConvTranspose2d(ch_out, ch_out, 4, 2, 1)
            ]))
            ch_in = ch_out

        # Final conv
        self.out_norm = gn(ch_in)
        self.out_conv = nn.Conv2d(ch_in, in_ch, 3, 1, 1)

    def forward(self, x, t, y=None):
        # ensure t shape = (B,1)
        if t.ndim == 1:
            t = t.unsqueeze(1)
        t = self.time_mlp(t)

        h = x
        residuals = []

        # Down path: save feature after downsample so spatial sizes align in up path
        for res1, res2, downsample in self.downs:
            h = res1(h, t, y)
            h = res2(h, t, y)
            h = downsample(h)
            residuals.append(h)

        # Bottleneck
        for res in self.mid:
            h = res(h, t, y)

        # Up path
        for res1, res2, upsample in self.ups:
            skip = residuals.pop()  # same spatial size as current h
            h = torch.cat([h, skip], dim=1)
            h = res1(h, t, y)
            h = res2(h, t, y)
            h = upsample(h)

        h = self.out_norm(h)
        return self.out_conv(F.silu(h))


# ----------------------- DDPM Core -----------------------

class GaussianDiffusion(nn.Module):
    def __init__(self, model: UNet, img_size=128, timesteps=1000, beta_schedule="cosine", device="cuda"):
        super().__init__()
        self.model = model
        self.device = device
        self.img_size = img_size
        self.timesteps = timesteps
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise NotImplementedError
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        # calc useful terms
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_variance', torch.log(torch.cat((betas[1:1], betas[1:]), dim=0)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self.sqrt_alphas_cumprod[t][:, None, None, None] * x_start +
            self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise
        )

    @torch.no_grad()
    def p_sample(self, x, t, y=None):
        model_out = self.model(x, t.float() / self.timesteps, y)
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = (1. / torch.sqrt(1. - betas_t))
        model_mean = (x - betas_t * model_out / sqrt_one_minus_alphas_cumprod_t) * sqrt_recip_alphas_t
        if t[0] == 0:
            return model_mean
        noise = torch.randn_like(x)
        posterior_variance_t = betas_t * (1. - self.alphas_cumprod_prev[t][:, None, None, None]) / (1. - self.alphas_cumprod[t][:, None, None, None])
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, batch_size=16, y=None):
        img = torch.randn(batch_size, 3, self.img_size, self.img_size, device=self.device)
        if exists(y):
            y = y.to(self.device)
        for t in reversed(range(self.timesteps)):
            img = self.p_sample(img, torch.full((batch_size,), t, device=self.device, dtype=torch.long), y)
        return img

    def p_losses(self, x_start, t, y, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t.float() / self.timesteps, y)
        return F.mse_loss(predicted_noise, noise)

# ----------------------- Data Loader -----------------------

def get_loader(data_dir, img_size=128, batch_size=32):
    trans = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    ds = ImageFolder(data_dir, transform=trans)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True), len(ds.classes)

# ----------------------- Training -----------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, n_classes = get_loader(args.data_dir, args.img_size, args.batch_size)
    cond_dim = n_classes if args.conditional else 0

    # Create unique log subdirectory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.log_dir, f"diffusion_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    args.log_dir = run_dir  # Overwrite original log_dir with new one

    # Save config to file
    with open(os.path.join(args.log_dir, "config.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(f"n_classes: {n_classes}\n")

    # Print config to console
    print("====== Training Configuration ======")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(f"n_classes: {n_classes}")
    print("====================================")

    model = UNet(in_ch=3, base_ch=args.base_ch, cond_dim=cond_dim).to(device)
    diffusion = GaussianDiffusion(model, img_size=args.img_size, timesteps=args.timesteps, device=device).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            if args.conditional:
                y = F.one_hot(labels, num_classes=n_classes).float().to(device)
            else:
                y = None
            t = torch.randint(0, args.timesteps, (imgs.size(0),), device=device).long()
            loss = diffusion.p_losses(imgs, t, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())
            global_step += 1
        # save samples
        if epoch % args.sample_every == 0:
            with torch.no_grad():
                if args.conditional:
                    y_grid = torch.eye(n_classes).to(device)
                    samples = diffusion.sample(batch_size=n_classes, y=y_grid)
                else:
                    samples = diffusion.sample(batch_size=16)
            grid = make_grid(samples, nrow=int(samples.size(0) ** 0.5))
            save_image(grid, os.path.join(args.log_dir, f"sample_epoch_{epoch}.png"))
            torch.save(model.state_dict(), os.path.join(args.log_dir, f"ddpm_epoch_{epoch}.pt"))

# ----------------------- CLI -----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Train DDPM on Impressionist dataset")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--log_dir", type=str, default="./logs")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--base_ch", type=int, default=64)
    ap.add_argument("--conditional", action="store_true")
    ap.add_argument("--sample_every", type=int, default=10)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
