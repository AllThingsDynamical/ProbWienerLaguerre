import torch
import torch.nn as nn
import torch.nn.functional as F

class GMBarron(nn.Module):
    """
    Uniform mixture of K diagonal Gaussians over (a,w,b).
      Choose component k ~ Uniform{1..K}, then sample via reparameterization.

    f(x) ≈ (1/S) Σ a_s * σ(w_s^T x + b_s)
    """
    def __init__(self, d_in: int, K: int = 8, S: int = 64, out_dim: int = 1,
                 activation: str = "tanh"):
        super().__init__()
        self.d_in = d_in
        self.K = K
        self.S = S
        self.out_dim = out_dim

        # Means per component
        self.mW = nn.Parameter(torch.randn(K, d_in) * 0.1)                 # (K,d)
        self.mb = nn.Parameter(torch.zeros(K, 1))                          # (K,1)

        if out_dim == 1:
            self.ma = nn.Parameter(torch.zeros(K, 1))                      # (K,1)
        else:
            self.ma = nn.Parameter(torch.zeros(K, out_dim))                # (K,out)

        # Log-std per component (diagonal covariances)
        self.log_sW = nn.Parameter(torch.full((K, d_in), -1.0))             # (K,d)
        self.log_sb = nn.Parameter(torch.full((K, 1), -1.0))                # (K,1)
        if out_dim == 1:
            self.log_sa = nn.Parameter(torch.full((K, 1), -1.0))            # (K,1)
        else:
            self.log_sa = nn.Parameter(torch.full((K, out_dim), -1.0))      # (K,out)

        if activation == "tanh":
            self.act = torch.tanh
        elif activation == "gelu":
            self.act = F.gelu
        elif activation == "relu":
            self.act = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,d)
        B, d = x.shape
        device = x.device
        S = self.S

        # Sample mixture indices uniformly (weights fixed => no gradient needed here)
        idx = torch.randint(low=0, high=self.K, size=(S,), device=device)   # (S,)

        # Gather component params for sampled indices
        mW = self.mW[idx]                  # (S,d)
        mb = self.mb[idx]                  # (S,1)
        ma = self.ma[idx]                  # (S,out) or (S,1)

        sW = self.log_sW[idx].exp()         # (S,d)
        sb = self.log_sb[idx].exp()         # (S,1)
        sa = self.log_sa[idx].exp()         # (S,out) or (S,1)

        # Reparameterized samples
        epsW = torch.randn(S, d, device=device)
        epsb = torch.randn(S, 1, device=device)
        epsa = torch.randn_like(ma)

        W = mW + sW * epsW                 # (S,d)
        b = mb + sb * epsb                 # (S,1)
        a = ma + sa * epsa                 # (S,out) or (S,1)

        # Compute features: pre = x @ W^T + b^T => (B,S)
        pre = x @ W.t() + b.t()
        h = self.act(pre)                   # (B,S)

        if self.out_dim == 1:
            y = (h * a.squeeze(-1)[None, :]).mean(dim=1, keepdim=True)      # (B,1)
        else:
            # (B,S) x (S,out) -> (B,out), averaged over S
            y = torch.einsum("bs,so->bo", h, a) / S

        return y


def estimate_GM_model(model, x:torch.Tensor, S:int, unbiased=True, stratified=True):
    device = x.device
    B, d = x.shape
    K = model.K
    S = model.S if S is None else int(S)

    # mixture indices
    if stratified:
        if S % K != 0:
            raise ValueError(f"Stratified eval requires S divisible by K. Got S={S}, K={K}.")
        m = S // K
        idx = torch.arange(K, device=device).repeat_interleave(m)  # (S,)
    else:
        idx = torch.randint(low=0, high=K, size=(S,), device=device)

    # gather params
    mW = model.mW[idx]                  # (S,d)
    mb = model.mb[idx]                  # (S,1)
    ma = model.ma[idx]                  # (S,out) or (S,1)

    sW = model.log_sW[idx].exp()         # (S,d)
    sb = model.log_sb[idx].exp()         # (S,1)
    sa = model.log_sa[idx].exp()         # (S,out) or (S,1)

    # reparameterized samples
    W = mW + sW * torch.randn(S, d, device=device)     # (S,d)
    b = mb + sb * torch.randn(S, 1, device=device)     # (S,1)
    a = ma + sa * torch.randn_like(ma)                 # (S,out) or (S,1)

    # features
    pre = x @ W.t() + b.t()             # (B,S)
    h = model.act(pre)                  # (B,S)

    # sample-wise outputs y_s(x)
    if model.out_dim == 1:
        ys = h * a.squeeze(-1)[None, :]            # (B,S)
        ys = ys.unsqueeze(-1)                      # (B,S,1)
    else:
        # (B,S,1) * (1,S,out) -> (B,S,out)
        ys = h.unsqueeze(-1) * a.unsqueeze(0)      # (B,S,out)

    mean = ys.mean(dim=1)  # (B,out) or (B,1)

    # variance across samples
    if unbiased and S > 1:
        var = ys.var(dim=1, unbiased=True)
    else:
        var = ys.var(dim=1, unbiased=False)

    return mean, var
