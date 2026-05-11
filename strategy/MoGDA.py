import torch
import torch.nn as nn


def _zero_like_none(grads, params):
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]


def _dot(grads_a, grads_b):
    out = None
    for ga, gb in zip(grads_a, grads_b):
        val = torch.sum(ga * gb)
        out = val if out is None else out + val
    return out


def _norm(grads, eps=1e-12):
    return torch.sqrt(_dot(grads, grads).clamp_min(0.0) + eps)


class MoGDAB(nn.Module):
    """Multi-model balancing for ReBorn MSST.

    g_common is the upstream representation-shaping gradient from the common
    model. g_barebone is the downstream segmentation gradient from the
    deployable missing-modality branch.
    """

    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps
        self.last_alpha = None
        self.last_cosine = None

    def forward(self, g_common, g_barebone):
        norm_common = _norm(g_common, self.eps)
        norm_barebone = _norm(g_barebone, self.eps)
        dot = _dot(g_barebone, g_common)
        self.last_cosine = (dot / (norm_common * norm_barebone + self.eps)).detach()

        conflict_scale = torch.minimum(
            torch.zeros((), device=dot.device, dtype=dot.dtype),
            dot / (norm_common.pow(2) + self.eps),
        )
        g_barebone_clean = [
            gb - conflict_scale * gc for gc, gb in zip(g_common, g_barebone)
        ]

        norm_clean = _norm(g_barebone_clean, self.eps)
        alpha = norm_clean / (norm_common + norm_clean + self.eps)
        self.last_alpha = alpha.detach()

        return [
            alpha * gc + (1.0 - alpha) * gb
            for gc, gb in zip(g_common, g_barebone_clean)
        ]


class MoGDAF(nn.Module):
    """Minimum-norm multi-trajectory fusion for ReBorn MCDP."""

    def __init__(self, max_iter=25, eps=1e-12):
        super().__init__()
        self.max_iter = max_iter
        self.eps = eps
        self.last_alpha = None

    def forward(self, trajectory_grads):
        n = len(trajectory_grads)
        if n == 1:
            self.last_alpha = torch.ones(1, device=trajectory_grads[0][0].device)
            return trajectory_grads[0]

        device = trajectory_grads[0][0].device
        gram = torch.empty(n, n, device=device)
        for i in range(n):
            for j in range(n):
                gram[i, j] = _dot(trajectory_grads[i], trajectory_grads[j])

        alpha = torch.full((n,), 1.0 / n, device=device)
        for t in range(self.max_iter):
            grad = 2.0 * gram.mv(alpha)
            vertex = torch.argmin(grad)
            direction = torch.zeros_like(alpha)
            direction[vertex] = 1.0
            direction = direction - alpha

            denom = direction @ gram @ direction
            if torch.abs(denom) <= self.eps:
                break
            step = torch.clamp(-((direction @ gram @ alpha) / denom), 0.0, 1.0)
            alpha = alpha + step * direction

            if step.item() < 1e-4:
                break

        self.last_alpha = alpha.detach()
        fused = []
        for param_id in range(len(trajectory_grads[0])):
            g = sum(alpha[i] * trajectory_grads[i][param_id] for i in range(n))
            fused.append(g)
        return fused


def grad_list(loss, params, retain_graph=True):
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return _zero_like_none(grads, params)
