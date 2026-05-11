from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .MoGDA import MoGDAB, MoGDAF, grad_list


Batch = Dict[str, torch.Tensor]
LossFn = Callable[[Dict[str, torch.Tensor], Batch], torch.Tensor]


@dataclass
class ReBornConfig:
    num_modalities: int = 4


def apply_modality_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply a [B, M] or [M] modality-availability mask to an image tensor."""
    if mask.dim() == 1:
        mask = mask.view(1, -1)
    view_shape = (mask.shape[0], mask.shape[1]) + (1,) * (x.dim() - 2)
    return x * mask.to(device=x.device, dtype=x.dtype).view(view_shape)


def endpoint_degradation_chains(num_modalities: int) -> List[List[torch.Tensor]]:
    """Build endpoint-covered degradation chains used by MCDP.

    Each chain starts from the full-modality state and removes modalities one
    by one while preserving a different endpoint modality.
    """
    chains = []
    full = torch.ones(num_modalities)
    for endpoint in range(num_modalities):
        chain = [full.clone()]
        current = full.clone()
        for modality in range(num_modalities):
            if modality == endpoint:
                continue
            current = current.clone()
            current[modality] = 0.0
            chain.append(current)
        chains.append(chain)
    return chains


class ReBornBackboneFactory:
    """Placeholder for model integration.

    Fill these two methods with the target segmentation model. The common model
    may keep training-only modules; the barebone model should be the deployable
    inference path.
    """

    def build_common_model(self) -> nn.Module:
        raise NotImplementedError("Please return the common/full training model.")

    def build_barebone_model(self) -> nn.Module:
        raise NotImplementedError("Please return the deployable barebone model.")


class ReBornMethod:
    """Backbone-agnostic ReBorn training method.

    This class implements the paper-level optimization logic only. It does not
    assume any concrete architecture.
    """

    def __init__(
        self,
        common_model: nn.Module,
        barebone_model: nn.Module,
        config: Optional[ReBornConfig] = None,
    ):
        self.common_model = common_model
        self.barebone_model = barebone_model
        self.config = config or ReBornConfig()
        self.mogda_b = MoGDAB()
        self.mogda_f = MoGDAF()

    def mcdp_step(
        self,
        batch_full: Batch,
        params: Sequence[nn.Parameter],
        loss_fn: LossFn,
    ) -> List[torch.Tensor]:
        """Return the fused MCDP gradient for one full-modality batch."""
        x_full = batch_full["image"]
        trajectory_grads = []
        chains = endpoint_degradation_chains(self.config.num_modalities)

        for chain in chains:
            loss_chain = 0.0
            for mask in chain:
                mask = mask.to(x_full.device)
                x_masked = apply_modality_mask(x_full, mask)
                outputs = self.common_model(x_masked)
                loss_chain = loss_chain + loss_fn(outputs, batch_full)
            loss_chain = loss_chain / len(chain)
            trajectory_grads.append(grad_list(loss_chain, params, retain_graph=True))

        return self.mogda_f(trajectory_grads)

    def msst_step(
        self,
        batch_missing: Batch,
        batch_full: Batch,
        barebone_params: Sequence[nn.Parameter],
        common_loss_fn: LossFn,
        barebone_loss_fn: LossFn,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Return the ReBorn MSST/MoGDA-B fused gradient.

        Expected batch keys:
            image: tensor [B, M, ...]
            mask: tensor [B, M] or [M]
            target: segmentation label tensor used by user-defined losses
        """
        x_missing = batch_missing["image"]
        x_full = batch_full["image"]
        mask = batch_missing["mask"]

        barebone_outputs = self.barebone_model(x_missing)
        barebone_loss = barebone_loss_fn(barebone_outputs, batch_missing)

        x_full_masked = apply_modality_mask(x_full, mask)
        common_outputs = self.common_model(x_full_masked)
        common_loss = common_loss_fn(common_outputs, batch_full)

        g_barebone = grad_list(barebone_loss, barebone_params, retain_graph=True)
        g_common = grad_list(common_loss, barebone_params, retain_graph=True)
        fused = self.mogda_b(g_common, g_barebone)

        return fused, common_loss, barebone_loss

    @staticmethod
    def assign_gradients(params: Iterable[nn.Parameter], grads: Sequence[torch.Tensor]) -> None:
        for param, grad in zip(params, grads):
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            param.grad.add_(grad.detach())
