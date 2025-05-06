import argparse
import os
import random
import datetime
from typing import Optional
import yaml
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image, make_grid
from tqdm import tqdm


# ------------------------ Utility Functions ------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------ Model Architecture ------------------------

class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 128, cond_dim: int = 0, img_size: int = 128):
        super().__init__()
        self.cond_dim = cond_dim
        self.map_size = img_size // 16
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + cond_dim, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.flatten = nn.Flatten()
        in_feats = 512 * self.map_size * self.map_size
        self.mu     = nn.Linear(in_feats, latent_dim)
        self.logvar = nn.Linear(in_feats, latent_dim)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        if self.cond_dim:
            y_img = y.view(y.size(0), self.cond_dim, 1, 1).expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, y_img], dim=1)
        h = self.conv(x)
        h = self.flatten(h)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, out_channels: int = 3, latent_dim: int = 128, cond_dim: int = 0, img_size: int = 128):
        super().__init__()
        self.map_size = img_size // 16
        self.fc = nn.Linear(latent_dim + cond_dim, 512 * self.map_size * self.map_size)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Sigmoid(),  # Output range 0~1
        )

    def forward(self, z: torch.Tensor, y: Optional[torch.Tensor] = None):
        if y is not None:
            z = torch.cat([z, y], dim=1)
        h = self.fc(z)
        h = h.view(h.size(0), 512, self.map_size, self.map_size)
        x_recon = self.deconv(h)
        return x_recon

class VAE(nn.Module):
    def __init__(self, img_channels: int = 3, latent_dim: int = 128, cond_dim: int = 0, img_size: int = 128):
        super().__init__()
        self.encoder = Encoder(img_channels, latent_dim, cond_dim, img_size)
        self.decoder = Decoder(img_channels, latent_dim, cond_dim, img_size)
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y=None):
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, y)
        return x_recon, mu, logvar

# ------------------------ Training & Evaluation ------------------------

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss, kld

def get_dataloaders(data_dir: str, img_size: int = 128, batch_size: int = 64):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return loader, len(dataset.classes)

def save_samples(model, device, save_path: str, epoch: int, n_samples: int = 64, y: Optional[torch.Tensor] = None):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, model.latent_dim).to(device)
        samples = model.decoder(z, y.to(device) if y is not None else None)
        grid = make_grid(samples, nrow=int(n_samples ** 0.5))
        save_image(grid, os.path.join(save_path, f"sample_epoch_{epoch}.png"))


# ------------------------ Training Pipeline ------------------------

def train(args):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, n_classes = get_dataloaders(args.data_dir, args.img_size, args.batch_size)
    cond_dim = n_classes if args.conditional else 0

    # Create unique log subdirectory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.log_dir, f"vae_{args.data_dir[7:]}_{timestamp}")
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


    latent_dim = args.latent_dim
    lr = args.lr
    kld_history = []     

    model = VAE(latent_dim=latent_dim, cond_dim=cond_dim, img_size=args.img_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(args.log_dir, exist_ok=True)

    # ---------- Training loop ----------
    for epoch in range(1, args.epochs + 1):
        # -------- KL annealing / cyclic β schedule --------
        # Linear warm‑up: β = 0 → args.beta over args.beta_warmup epochs.
        # If --beta_cycle > 0, this pattern repeats every beta_cycle epochs.
        if args.beta_cycle > 0:
            # Position within current cycle (0‑based)
            cycle_pos = (epoch - 1) % args.beta_cycle
            warm_frac = min(1.0, cycle_pos / max(1, args.beta_warmup))
        else:
            # One‑shot warm‑up that never resets
            warm_frac = min(1.0, epoch / max(1, args.beta_warmup))
        beta = args.beta * warm_frac
        # ---------------------------------------------------

        model.train()
        running_loss = 0.0
        epoch_kld_sum = 0.0
        
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            imgs = imgs.to(device)
            one_hot = nn.functional.one_hot(targets, num_classes=n_classes).float().to(device) if args.conditional else None
            recon, mu, logvar = model(imgs, one_hot)
            loss, recon_loss, kld = loss_function(recon, imgs, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_kld_sum += kld.item()

        avg_loss = running_loss / len(train_loader.dataset)
        avg_kld = epoch_kld_sum / len(train_loader.dataset)  # epoch_kld_sum 是你在 inner loop 累加的
        kld_history.append(avg_kld)
        print(f"Epoch {epoch}: loss={avg_loss:.4f} | β={beta:.4f} | KL={avg_kld:.4f}")

        if epoch % args.sample_every == 0:
            if args.conditional:
                y_grid = torch.eye(n_classes).repeat(args.sample_per_class, 1)
                save_samples(model, device, args.log_dir, epoch, y=y_grid)
            else:
                save_samples(model, device, args.log_dir, epoch)

            torch.save(model.state_dict(), os.path.join(args.log_dir, f"vae_epoch_{epoch}.pt"))
    plt.plot(range(1, len(kld_history)+1), kld_history)
    plt.xlabel("Epoch")
    plt.ylabel("Average KL divergence")
    plt.title("KL curve")
    plt.savefig(os.path.join(args.log_dir, "kl_curve.png"))
    plt.close()

# ------------------------ Inference ------------------------

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load number of classes from dataset
    loader, n_classes = get_dataloaders(args.data_dir, args.img_size, args.batch_size)
    cond_dim = n_classes if args.conditional else 0

    # Build model and load weights
    model = VAE(latent_dim=args.latent_dim, cond_dim=cond_dim, img_size=args.img_size).to(device)
    assert args.model_path is not None and os.path.exists(args.model_path), "Model checkpoint not found."
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Determine save path
    save_dir = os.path.dirname(args.model_path)

    print(f"[Inference] Generating {args.n_samples} samples using model from {args.model_path}")
    if args.conditional:
        y_grid = torch.eye(n_classes).repeat(args.sample_per_class, 1).to(device)
        save_samples(model, device, save_dir, epoch="inference", n_samples=len(y_grid), y=y_grid)
    else:
        save_samples(model, device, save_dir, epoch="inference", n_samples=args.n_samples)


# ------------------------ Argument Parsing ------------------------

def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    config_args, remaining_argv = config_parser.parse_known_args()

    config_defaults = {}
    if config_args.config and os.path.exists(config_args.config):
        with open(config_args.config, "r") as f:
            config_defaults = yaml.safe_load(f)
        print(f"✅ Loaded config from {config_args.config}")
    elif config_args.config:
        print(f"⚠️ Config file not found: {config_args.config}, using argparse defaults.")

    parser = argparse.ArgumentParser(
        parents=[config_parser],
        description="Impressionist Paintings Generator (VAE/CVAE)"
    )

    parser.set_defaults(**config_defaults)

    parser.add_argument("--data_dir", type=str, help="Dataset root directory")
    parser.add_argument("--log_dir", type=str, help="Directory for saving logs and samples")
    parser.add_argument("--epochs", type=int, help="Training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--latent_dim", type=int, help="Latent dimension")
    parser.add_argument("--img_size", type=int, help="Image size (square)")
    parser.add_argument("--conditional", action="store_true", help="Use Conditional VAE")
    parser.add_argument("--sample_every", type=int, help="Sample every N epochs")
    parser.add_argument("--sample_per_class", type=int, help="Samples per class for CVAE")
    parser.add_argument("--inference", action="store_true", help="Run inference mode only")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--n_samples", type=int, help="Samples to generate during inference")
    parser.add_argument("--beta", type=float, help="Weight for the KL divergence term in the loss")
    parser.add_argument("--beta_warmup", type=int, help="Number of epochs to linearly ramp β from 0 → --beta")
    parser.add_argument("--beta_cycle", type=int, help="Length of a full β cycle in epochs; 0 = no cycle (one‑shot warm‑up)")

    args = parser.parse_args(remaining_argv)

    return args

# ------------------------ Entry Point ------------------------

if __name__ == "__main__":
    args = parse_args()
    if args.inference:
        run_inference(args)
    else:
        train(args)
