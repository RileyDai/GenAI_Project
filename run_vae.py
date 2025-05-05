import argparse
import os
import random
import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ------------------------ Utility Functions ------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------ Model Architecture ------------------------

class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 128, cond_dim: int = 0):
        super().__init__()
        self.cond_dim = cond_dim
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
        self.mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.logvar = nn.Linear(512 * 8 * 8, latent_dim)

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
    def __init__(self, out_channels: int = 3, latent_dim: int = 128, cond_dim: int = 0):
        super().__init__()
        self.fc = nn.Linear(latent_dim + cond_dim, 512 * 8 * 8)
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
        h = h.view(h.size(0), 512, 8, 8)
        x_recon = self.deconv(h)
        return x_recon

class VAE(nn.Module):
    def __init__(self, img_channels: int = 3, latent_dim: int = 128, cond_dim: int = 0):
        super().__init__()
        self.encoder = Encoder(img_channels, latent_dim, cond_dim)
        self.decoder = Decoder(img_channels, latent_dim, cond_dim)
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

# ------------------------ Hyperparameter Tuning ------------------------

def objective(trial, args, n_classes):
    latent_dim = trial.suggest_categorical("latent_dim", [64, 128, 256])
    lr = trial.suggest_loguniform("lr", 1e-4, 5e-3)
    beta = trial.suggest_uniform("beta", 0.5, 1.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim=latent_dim, cond_dim=(n_classes if args.conditional else 0)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader, _ = get_dataloaders(args.data_dir, args.img_size, args.batch_size)
    model.train()
    epoch_loss = 0.0
    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        one_hot = nn.functional.one_hot(targets, num_classes=n_classes).float().to(device) if args.conditional else None
        recon, mu, logvar = model(imgs, one_hot)
        loss, _, _ = loss_function(recon, imgs, mu, logvar, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader.dataset)

# ------------------------ Training Pipeline ------------------------

def train(args):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, n_classes = get_dataloaders(args.data_dir, args.img_size, args.batch_size)
    cond_dim = n_classes if args.conditional else 0

    # Create unique log subdirectory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.log_dir, f"vae_{timestamp}")
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

    if args.tune and not HAS_OPTUNA:
        raise ImportError("Optuna is not installed. Please run 'pip install optuna' first.")

    # ---------- Hyperparameter tuning ----------
    if args.tune:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, args, n_classes), n_trials=args.n_trials)
        best_params = study.best_params
        print("[Optuna] Best parameters:", best_params)
        latent_dim = best_params["latent_dim"]
        lr = best_params["lr"]
        beta = best_params["beta"]
    else:
        latent_dim = args.latent_dim
        lr = args.lr
        beta = args.beta

    model = VAE(latent_dim=latent_dim, cond_dim=cond_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(args.log_dir, exist_ok=True)

    # ---------- Training loop ----------
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            imgs = imgs.to(device)
            one_hot = nn.functional.one_hot(targets, num_classes=n_classes).float().to(device) if args.conditional else None
            recon, mu, logvar = model(imgs, one_hot)
            loss, recon_loss, kld = loss_function(recon, imgs, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

        if epoch % args.sample_every == 0:
            if args.conditional:
                y_grid = torch.eye(n_classes).repeat(args.sample_per_class, 1)
                save_samples(model, device, args.log_dir, epoch, y=y_grid)
            else:
                save_samples(model, device, args.log_dir, epoch)

            torch.save(model.state_dict(), os.path.join(args.log_dir, f"vae_epoch_{epoch}.pt"))

# ------------------------ Argument Parsing ------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Impressionist Paintings Generator (VAE/CVAE)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root directory of the dataset, must follow ImageFolder structure")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save logs and generated samples")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--latent_dim", type=int, default=128, help="Dimensionality of the latent space")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for the KL divergence term in the loss")
    parser.add_argument("--img_size", type=int, default=128, help="Size to resize input images to (img_size x img_size)")
    parser.add_argument("--conditional", action="store_true", help="Enable Conditional VAE (CVAE)")
    parser.add_argument("--sample_every", type=int, default=10, help="Save sample images every N epochs")
    parser.add_argument("--sample_per_class", type=int, default=8, help="Number of samples per class to generate (only in CVAE mode)")
    parser.add_argument("--n_trials", type=int, default=25, help="Number of trials for Optuna hyperparameter tuning")
    parser.add_argument("--tune", action="store_true", help="Enable Optuna-based hyperparameter tuning")
    return parser.parse_args()

# ------------------------ Entry Point ------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
