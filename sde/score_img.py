import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ---------------------------
# Device: CPU or Apple Metal (MPS)
# ---------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print("Using device:", device)

# ---------------------------
# 1) Dataset (MNIST scaled to [-1, 1])
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2. - 1.)
])
train_data = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
dataloader = DataLoader(train_data, batch_size=128, shuffle=True,
                        num_workers=2 if device.type == "cpu" else 0)

# ---------------------------
# 2) Noise schedule (Cosine)
# ---------------------------
def betas_for_cosine_schedule(T: int, s: float = 0.008, device=None):
    steps = T + 1
    x = torch.linspace(0, T, steps, device=device)
    alphas_bar = torch.cos(((x / T + s) / (1 + s)) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return betas.clamp(1e-8, 0.999)

T = 100
betas = betas_for_cosine_schedule(T, device=device)
alphas = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

# ---------------------------
# 3) Tiny U-Net + sinusoidal t-embedding (predicts SCORE now)
# ---------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs)
        self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().unsqueeze(1)                  # [B, 1]
        ang = t * self.freqs.unsqueeze(0)          # [B, half]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)  # [B, dim]
        return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1   = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2   = nn.GroupNorm(8, out_ch)
        self.emb   = nn.Linear(t_dim, out_ch)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, t_emb):
        h = F.silu(self.gn1(self.conv1(x)))
        h = h + self.emb(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.gn2(self.conv2(h)))
        return h + self.skip(x)

class TinyUNet(nn.Module):
    def __init__(self, t_dim: int = 64, base: int = 32):
        super().__init__()
        self.tok = SinusoidalTimeEmbedding(t_dim)
        # encoder
        self.inp  = nn.Conv2d(1, base, 3, padding=1)       # 28x28
        self.down = nn.Conv2d(base, base*2, 4, 2, 1)       # 28->14
        self.rb1  = ResBlock(base*2, base*2, t_dim)
        # bottleneck
        self.mid1 = ResBlock(base*2, base*2, t_dim)
        # decoder
        self.up   = nn.ConvTranspose2d(base*2, base, 4, 2, 1)  # 14->28
        self.rb2  = ResBlock(base, base, t_dim)
        self.out  = nn.Conv2d(base, 1, 3, padding=1)  # predicts SCORE s_theta
    def forward(self, x, t):
        te = self.tok(t)                 # [B, t_dim]
        h = F.silu(self.inp(x))
        h = F.silu(self.down(h))
        h = self.rb1(h, te)
        h = self.mid1(h, te)
        h = F.silu(self.up(h))
        h = self.rb2(h, te)
        return self.out(h)               # shape [B,1,28,28], score estimate

model = TinyUNet(t_dim=64, base=32).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# (Optional) EMA for cleaner samples
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)

ema = EMA(model, decay=0.999)

# ---------------------------
# 4) Training loop (DSM for VP)
#    Target score: -eps / sqrt(1 - alpha_bar_t)
#    Optional weighting: lambda(t) = 1 - alpha_bar_t
# ---------------------------
epochs = 20  # increase for sharper digits
model.train()
for epoch in range(epochs):
    for x0, _ in dataloader:
        x0 = x0.to(device)                           # [B,1,28,28]
        B = x0.size(0)
        t = torch.randint(1, T+1, (B,), device=device)  # [B]
        alpha_bar_t = alphas_bar[t-1].view(B, 1, 1, 1)  # [B,1,1,1]

        # forward (noising)
        eps = torch.randn_like(x0)                    # [B,1,28,28]
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps

        # predict SCORE
        s_pred = model(x_t, t)                        # [B,1,28,28]

        # DSM target score (VP)
        denom = torch.sqrt(1 - alpha_bar_t + 1e-8)    # stabilize small t
        target_score = - eps / denom                  # [B,1,28,28]

        # weighting lambda(t) = 1 - alpha_bar_t (optional but helpful)
        lam = (1 - alpha_bar_t).detach()
        loss = (lam * (s_pred - target_score)**2).mean()

        # optimize
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

# ---------------------------
# 5) Sampling (reverse process) using EMA weights
#    Convert score -> noise: eps_theta = -sqrt(1 - alpha_bar_t) * s_theta
# ---------------------------
@torch.no_grad()
def sample_with_ema(model, n_samples=16):
    # copy EMA weights into a temp model
    tmp = TinyUNet(t_dim=64, base=32).to(device)
    tmp.load_state_dict(model.state_dict(), strict=True)
    ema.copy_to(tmp)
    tmp.eval()

    x_t = torch.randn(n_samples, 1, 28, 28, device=device)
    for t in reversed(range(1, T+1)):
        alpha_t     = alphas[t-1]
        alpha_bar_t = alphas_bar[t-1]
        beta_t      = betas[t-1]

        # predict SCORE then convert to noise prediction
        s_theta = tmp(x_t, torch.full((n_samples,), t, device=device))   # [B,1,28,28]
        eps_theta = - torch.sqrt(1 - alpha_bar_t) * s_theta              # [B,1,28,28]

        # DDPM-style reverse mean using eps_theta
        mean = (1/torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_theta
        )
        if t > 1:
            z = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            x_t = mean + sigma_t * z
        else:
            x_t = mean
    return x_t

# ---------------------------
# 6) Visualize samples
# ---------------------------
os.makedirs("mnist_samples", exist_ok=True)
samples = sample_with_ema(model, 16).cpu()

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flatten()):
    ax.imshow((samples[i, 0] + 1) / 2, cmap="gray")  # back to [0,1]
    ax.axis("off")
plt.tight_layout()
plt.savefig("mnist_samples/generated_unet_scorematching.png")
plt.show()

print("Saved samples to mnist_samples/generated_unet_scorematching.png")
