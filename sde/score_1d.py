import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------------------------
# 0) Repro + device
# ---------------------------
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cpu")

# ---------------------------
# 1) Toy dataset: mixture of two 1D Gaussians (-2 and +2)
# ---------------------------
def sample_data(batch_size):
    """Mixture of two Gaussians centered at -2 and +2."""
    centers = torch.randint(0, 2, (batch_size,), device=device) * 4 - 2  # -2 or +2
    return centers + 0.5 * torch.randn(batch_size, device=device)

# ---------------------------
# 2) VP/DDPM-style forward schedule (T steps)
#    x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps
# ---------------------------
T = 100
betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

# ---------------------------
# 3) Score network (predicts score s_theta(x_t, t))
#    Same MLP as before; we just interpret its output as a score.
# ---------------------------
class ScoreNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x_t, t):
        # x_t: [B], t: [B] in {1..T}
        t_norm = t.float() / T  # normalize to [0,1]
        inp = torch.stack([x_t, t_norm], dim=1)
        return self.net(inp).squeeze(1)  # s_theta(x_t, t)

model = ScoreNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# 4) Training (Denoising Score Matching, VP case)
#    Target score for VP:  -eps / sqrt(1 - alpha_bar_t)
# ---------------------------
batch_size = 128
num_steps = 20_000

for step in range(num_steps):
    x0 = sample_data(batch_size)                         # [B]
    t  = torch.randint(1, T+1, (batch_size,), device=device)  # [B]
    alpha_bar_t = alphas_bar[t-1]                        # [B]

    # forward (noising)
    eps = torch.randn(batch_size, device=device)         # [B]
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps

    # predict score
    s_pred = model(x_t, t)                               # [B]

    # DSM target (analytic conditional score under VP)
    target_score = - eps / torch.sqrt(1 - alpha_bar_t)   # [B]

    # loss (lambda(t) = 1 for VP; simple & works well here)
    loss = F.mse_loss(s_pred, target_score)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 2000 == 0:
        print(f"Step {step:6d} | Loss: {loss.item():.6f}")

# ---------------------------
# 5) Sampling: reverse process using score -> noise conversion
#    eps_theta(x_t,t) = - sqrt(1 - alpha_bar_t) * s_theta(x_t,t)
# ---------------------------
@torch.no_grad()
def sample(model, n_samples=5000, save_frames=True):
    x_t = torch.randn(n_samples, device=device)  # start from pure noise

    # Prepare reference (true distribution) once for overlays
    os.makedirs("denoising_steps", exist_ok=True)
    ref_samples = sample_data(200_000).cpu().numpy()
    bins = np.linspace(-6, 6, 200)
    ref_hist, ref_edges = np.histogram(ref_samples, bins=bins, density=True)
    ref_centers = 0.5 * (ref_edges[:-1] + ref_edges[1:])

    for t in reversed(range(1, T+1)):
        alpha_t     = alphas[t-1]        # scalar tensor
        alpha_bar_t = alphas_bar[t-1]    # scalar tensor
        beta_t      = betas[t-1]         # scalar tensor

        # predict score, then convert to a noise prediction
        s_pred = model(x_t, torch.full((n_samples,), t, device=device))
        eps_theta = - torch.sqrt(1 - alpha_bar_t) * s_pred  # [B]

        # reverse mean (DDPM formula, uses eps estimate)
        mean = (1/torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_theta
        )

        if t > 1:
            z = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            x_t = mean + sigma_t * z
        else:
            x_t = mean

        # Save frames
        if save_frames:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(x_t.cpu().numpy(), bins=bins, density=True, alpha=0.6, label="Current $x_t$")
            ax.plot(ref_centers, ref_hist, linewidth=2.0, label="True distribution")
            ax.set_title(f"Denoising step t={t}")
            ax.set_xlabel("x")
            ax.set_ylabel("Density")
            ax.legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(f"denoising_steps/step_{T - t:04d}.png")
            plt.close(fig)

    return x_t.cpu()

# Run sampling and save all 1000 frames
samples = sample(model, n_samples=5000, save_frames=True)

# ---------------------------
# 6) Compare final samples vs true dist (single summary plot)
# ---------------------------
final = samples.numpy()
true_data = sample_data(5000).cpu().numpy()

plt.figure(figsize=(10,4))
plt.hist(true_data, bins=50, density=True, alpha=0.6, label="True data")
plt.hist(final,     bins=50, density=True, alpha=0.6, label="Score-based samples")
plt.legend()
plt.title("1D Score-based model: Learned vs True Distribution")
plt.tight_layout()
os.makedirs("denoising_summary", exist_ok=True)
plt.savefig("denoising_summary/score_based_vs_true.png")
plt.show()

print("Saved frames to denoising_steps/ and summary to denoising_summary/score_based_vs_true.png")
