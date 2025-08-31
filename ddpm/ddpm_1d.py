import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------------------------
# 1. Toy dataset: mixture of Gaussians
# ---------------------------
def sample_data(batch_size):
    """Mixture of two Gaussians centered at -2 and +2."""
    centers = torch.randint(0, 2, (batch_size,)) * 4 - 2  # -2 or +2
    return centers + 0.5 * torch.randn(batch_size)

# ---------------------------
# 2. Noise schedule
# ---------------------------
T = 1000  # number of diffusion steps
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

# ---------------------------
# 3. Simple 1D model (MLP)
# ---------------------------
class NoisePredictor(nn.Module):
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
        # normalize timestep to [0,1]
        t_norm = t.float() / T
        inp = torch.stack([x_t, t_norm], dim=1)
        return self.net(inp).squeeze(1)

model = NoisePredictor()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# 4. Training loop
# ---------------------------
batch_size = 128
num_steps = 20000

for step in range(num_steps):
    x0 = sample_data(batch_size)

    # sample random timestep
    t = torch.randint(1, T+1, (batch_size,))
    alpha_bar_t = alphas_bar[t-1]

    # add noise
    eps = torch.randn(batch_size)
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps

    # predict noise
    eps_pred = model(x_t, t)

    # loss
    loss = F.mse_loss(eps_pred, eps)

    # optimize
    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 2000 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# ---------------------------
# 5. Sampling (reverse process)
# ---------------------------
@torch.no_grad()
def sample(model, n_samples=5000):
    x_t = torch.randn(n_samples)  # start from pure noise
    for t in reversed(range(1, T+1)):
        alpha_t = alphas[t-1]
        alpha_bar_t = alphas_bar[t-1]
        beta_t = betas[t-1]

        # predict noise
        eps_theta = model(x_t, torch.full((n_samples,), t))

        # compute mean
        mean = (1/torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_theta
        )

        if t > 1:
            z = torch.randn(n_samples)
            sigma_t = torch.sqrt(beta_t)
            x_t = mean + sigma_t * z
        else:
            x_t = mean
    return x_t

# ---------------------------
# 6. Visualize
# ---------------------------
samples = sample(model, 5000).numpy()
true_data = sample_data(5000).numpy()

plt.figure(figsize=(10,4))
plt.hist(true_data, bins=50, density=True, alpha=0.6, label="True data")
plt.hist(samples, bins=50, density=True, alpha=0.6, label="DDPM samples")
plt.legend()
plt.title("1D DDPM: Learned vs True Distribution")
plt.show()
