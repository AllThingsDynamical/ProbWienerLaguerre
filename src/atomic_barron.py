import torch
import torch.nn as nn
import torch.nn.functional as F

class AtomicBarron(nn.Module):
    """
    Atomic Barron / 2-layer ridge network:
        f(x) = scale * sum_i a_i * sigma(w_i^T x + b_i)

    Shapes:
        x: (B, d)
        w: (K, d)
        b: (K,)
        a: (K,) or (K, out_dim)
    """
    def __init__(self, d_in: int, K: int, out_dim: int = 1,
                 activation: str = "tanh", normalize: bool = True):
        super().__init__()
        self.d_in = d_in # Input dimension
        self.K = K # Number of neurons
        self.out_dim = out_dim # Output dimension
        self.normalize = normalize 

        # Parameters of the empirical measure
        self.W = nn.Parameter(torch.randn(K, d_in) / (d_in ** 0.5))
        self.b = nn.Parameter(torch.zeros(K))

        if out_dim == 1:
            self.a = nn.Parameter(torch.randn(K) * 0.01)
        else:
            self.a = nn.Parameter(torch.randn(K, out_dim) * 0.01)

        if activation == "tanh":
            self.act = torch.tanh
        elif activation == "gelu":
            self.act = F.gelu
        elif activation == "relu":
            self.act = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,d) -> pre: (B,K)
        pre = x @ self.W.t() + self.b
        h = self.act(pre)  # (B,K)

        # combine with a
        if self.out_dim == 1:
            y = h @ self.a  # (B,)
            y = y.unsqueeze(-1)  # (B,1)
        else:
            # (B,K) @ (K,out_dim) -> (B,out_dim)
            y = h @ self.a

        if self.normalize:
            y = y / self.K

        return y
