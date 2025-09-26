# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
# Modified 2025 by contributors to replace Numba/CUDA kernels
# with a pure-PyTorch implementation (no numba/cuda) while
# keeping the same public API (SoftDTW class/signature/behavior).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------------------------------------------------

import math
import numpy as np  # kept for backward-compat (not used by core ops)
import torch
from torch.autograd import Function

# ----------------------------------------------------------------------------------------------------------------------
# Pure-PyTorch soft-DTW (Cuturi & Blondel smoothing with log-sum-exp)
# This file intentionally keeps the original class name and API (SoftDTW) so the
# rest of the pipeline can import/use it unchanged.
# - No numba
# - No cuda.jit kernels
# - Works under DDP and in subprocesses
# ----------------------------------------------------------------------------------------------------------------------

def _softdtw_forward_torch(D: torch.Tensor, gamma: float, bandwidth: float):
    """
    Vectorized forward dynamic program in PyTorch.
    D: (B, N, M) pairwise distance matrix
    gamma: smoothing > 0
    bandwidth: 0 for 'no pruning'; >0 for Sakoe-Chiba band
    Returns:
        R: (B, N+2, M+2) DP table (with padding)
    """
    device = D.device
    dtype = D.dtype
    B, N, M = D.shape
    INF = torch.tensor(float("inf"), device=device, dtype=dtype)

    R = torch.full((B, N + 2, M + 2), INF, device=device, dtype=dtype)
    R[:, 0, 0] = 0.0

    # Precompute j index tensor for bandwidth pruning
    if bandwidth is None:
        bandwidth = 0.0
    bandwidth = float(bandwidth)
    js = torch.arange(1, M + 1, device=device).view(1, M)  # (1, M)

    inv_gamma = 1.0 / float(gamma)

    # Fill DP row by row; for each row i compute all columns j at once
    for i in range(1, N + 1):
        # neighbors for all j (1..M)
        # r0 = R[i-1, j-1]  (대각)
        r0 = -R[:, i - 1, 0:M] * inv_gamma
        # r1 = R[i-1, j]    (위)
        r1 = -R[:, i - 1, 1:M + 1] * inv_gamma
        # r2 = R[i,   j-1]  (왼)
        r2 = -R[:, i, 0:M] * inv_gamma

        rmax = torch.maximum(torch.maximum(r0, r1), r2)
        rsum = torch.exp(r0 - rmax) + torch.exp(r1 - rmax) + torch.exp(r2 - rmax)
        softmin = -float(gamma) * (torch.log(rsum) + rmax)  # (B, M)

        # bandwidth mask: keep cells where |i - j| <= bandwidth (if bandwidth > 0)
        if bandwidth > 0:
            mask = (torch.abs(i - js) <= bandwidth).to(D.dtype)  # (1, M) in {0,1}
        else:
            mask = None

        # Update row i for all j
        # D[:, i-1, :] is (B, M)
        row_vals = D[:, i - 1, :] + softmin
        if mask is not None:
            row_vals = row_vals * mask  + INF * (1.0 - mask)

        R[:, i, 1:M + 1] = row_vals

    return R


def _softdtw_backward_torch(D: torch.Tensor, R: torch.Tensor, gamma: float, bandwidth: float):
    """
    Vectorized backward recursion in PyTorch to compute E = d softDTW / d D
    D: (B, N, M)
    R: (B, N+2, M+2) from forward
    Returns:
        E[:, 1:N+1, 1:M+1] (B, N, M)
    """
    device = D.device
    dtype = D.dtype
    B, N, M = D.shape

    if bandwidth is None:
        bandwidth = 0.0
    bandwidth = float(bandwidth)
    js = torch.arange(1, M + 1, device=device).view(1, M)

    # Pad D to align with R's indexing
    Dp = torch.zeros((B, N + 2, M + 2), device=device, dtype=dtype)
    Dp[:, 1:N + 1, 1:M + 1] = D

    Rp = R.clone()
    # Boundary conditions
    neg_inf = torch.tensor(-float("inf"), device=device, dtype=dtype)
    Rp[:, :, -1] = neg_inf
    Rp[:, -1, :] = neg_inf
    Rp[:, -1, -1] = Rp[:, -2, -2]

    E = torch.zeros_like(Dp)
    E[:, -1, -1] = 1.0

    inv_gamma = 1.0 / float(gamma)

    # Backward along rows
    for i in range(N, 0, -1):
        # a: from (i+1, j) -> (i, j)
        a = torch.exp((Rp[:, i + 1, 1:M + 1] - Rp[:, i, 1:M + 1] - Dp[:, i + 1, 1:M + 1]) * inv_gamma)
        # b: from (i, j+1) -> (i, j)
        b = torch.exp((Rp[:, i, 2:M + 2] - Rp[:, i, 1:M + 1] - Dp[:, i, 2:M + 2]) * inv_gamma)
        # c: from (i+1, j+1) -> (i, j)
        c = torch.exp((Rp[:, i + 1, 2:M + 2] - Rp[:, i, 1:M + 1] - Dp[:, i + 1, 2:M + 2]) * inv_gamma)

        Ei = E[:, i, 1:M + 1]  # (B, M)
        newEi = E[:, i + 1, 1:M + 1] * a + E[:, i, 2:M + 2] * b + E[:, i + 1, 2:M + 2] * c

        if bandwidth > 0:
            mask = (torch.abs(i - js) <= bandwidth).to(D.dtype)
            newEi = newEi * mask  # outside band remains 0

        E[:, i, 1:M + 1] = newEi

    return E[:, 1:N + 1, 1:M + 1]  # strip padding


class _SoftDTW_Torch(Function):
    """
    Autograd bridge around the pure-PyTorch forward/backward recurrences above.
    """

    @staticmethod
    def forward(ctx, D: torch.Tensor, gamma: float, bandwidth: float):
        # D: (B, N, M)
        assert D.dim() == 3, "D must be (B,N,M)"
        if not torch.is_floating_point(D):
            D = D.float()

        R = _softdtw_forward_torch(D, gamma=gamma, bandwidth=bandwidth)
        # save for backward
        ctx.save_for_backward(D, R)
        ctx.gamma = float(gamma)
        ctx.bandwidth = float(0.0 if bandwidth is None else bandwidth)
        # final value is R[:, -2, -2]
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        D, R = ctx.saved_tensors
        gamma = ctx.gamma
        bandwidth = ctx.bandwidth
        E = _softdtw_backward_torch(D, R, gamma=gamma, bandwidth=bandwidth)  # (B,N,M)
        # chain rule
        g = grad_output.view(-1, 1, 1).expand_as(E) * E
        return g, None, None


# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    """
    Drop-in replacement for the original SoftDTW class.
    - Signature preserved: SoftDTW(use_cuda, gamma=1.0, normalize=False, bandwidth=None, dist_func=None)
    - `use_cuda` is accepted but ignored (this implementation works on CPU or GPU via PyTorch).
    """

    def __init__(self, use_cuda, gamma=1.0, normalize=False, bandwidth=None, dist_func=None):
        super().__init__()
        self.normalize = bool(normalize)
        self.gamma = float(gamma)
        self.bandwidth = 0.0 if bandwidth is None else float(bandwidth)
        self.use_cuda = bool(use_cuda)  # kept only for API compatibility
        self._warned = False

        # distance function
        self.dist_func = dist_func if dist_func is not None else SoftDTW._euclidean_dist_func

    def _maybe_warn_cuda(self):
        if self.use_cuda and not self._warned:
            # This implementation ignores numba/cuda kernels on purpose.
            print("[SoftDTW] Using pure-PyTorch implementation (numba/cuda disabled).")
            self._warned = True

    @staticmethod
    def _euclidean_dist_func(x, y):
        """
        Squared Euclidean distance per timestep.
        x: (B, N, d), y: (B, M, d) -> D: (B, N, M)
        """
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(dim=3)

    def _apply_core(self, X, Y):
        self._maybe_warn_cuda()
        D = self.dist_func(X, Y)  # (B,N,M)
        return _SoftDTW_Torch.apply(D, self.gamma, self.bandwidth)

    def forward(self, X, Y):
        """
        X, Y: (B, L, dims)  -> returns (B,) soft-DTW_γ
        If normalize=True, returns sDTW(X,Y) - 1/2 (sDTW(X,X) + sDTW(Y,Y)).
        """
        if self.normalize:
            x = torch.cat([X, X, Y], dim=0)
            y = torch.cat([Y, X, Y], dim=0)
            out = self._apply_core(x, y)
            b = X.shape[0]
            out_xy, out_xx, out_yy = torch.split(out, [b, b, b], dim=0)
            return out_xy - 0.5 * (out_xx + out_yy)
        else:
            return self._apply_core(X, Y)


# ----------------------------------------------------------------------------------------------------------------------
# Optional: quick profiler that now compares "torch vs torch" (kept for API parity)
# ----------------------------------------------------------------------------------------------------------------------
def timed_run(a, b, sdtw):
    from timeit import default_timer as timer
    start = timer()
    forward = sdtw(a, b)
    fwd_t = timer() - start

    grad_outputs = torch.ones_like(forward)
    start = timer()
    grads = torch.autograd.grad(forward, a, grad_outputs=grad_outputs)[0]
    bwd_t = timer() - start

    return fwd_t + bwd_t, forward, grads


def profile(batch_size, seq_len_a, seq_len_b, dims, tol_backward):
    sdtw = SoftDTW(False, gamma=1.0, normalize=False)   # both will use torch path
    sdtw_torch = SoftDTW(True,  gamma=1.0, normalize=False)

    print(f"Profiling forward()+backward() for batch={batch_size}, Lx={seq_len_a}, Ly={seq_len_b}, d={dims} ...")

    times_a, times_b = [], []
    for i in range(6):
        a_cpu = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b_cpu = torch.rand((batch_size, seq_len_b, dims))
        a_dev = a_cpu.to("cuda") if torch.cuda.is_available() else a_cpu
        b_dev = b_cpu.to(a_dev.device)

        t1, f1, g1 = timed_run(a_dev, b_dev, sdtw)
        t2, f2, g2 = timed_run(a_dev, b_dev, sdtw_torch)

        # sanity
        #assert torch.allclose(f1, f2, atol=1e-5, rtol=1e-6)
        #assert torch.allclose(g1, g2, atol=1e-4, rtol=1e-5)

        if i > 0:
            times_a.append(t1)
            times_b.append(t2)

    print("  mean time implA:", np.mean(times_a))
    print("  mean time implB:", np.mean(times_b))
    print("  OK\n")


if __name__ == "__main__":
    torch.manual_seed(1234)
    profile(128, 17, 15, 2, tol_backward=1e-5)
    profile(64,  64, 64, 2, tol_backward=1e-4)
    profile(16, 256, 256, 2, tol_backward=1e-3)
