import argparse
import copy
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def ijepa_distill_wrapper(Method=object):
    class IJEPADistillWrapper(Method):
        """CaSSLe-style distillation for I-JEPA.

        Freezes the previous task's context encoder and predictor, then trains a
        lightweight projector that maps the *current* model's patch predictions
        towards the *frozen* model's patch predictions.  This preserves the
        model's predictive capability over previously-seen distributions without
        touching the target-encoder EMA path.

        Loss (only active for task_idx > 0):
            distill_loss = smooth_l1(projector(current_preds), frozen_preds.detach())
            total_loss   = ijepa_loss + distill_lamb * distill_loss
        """

        def __init__(self, distill_lamb: float, distill_proj_hidden_dim: int, **kwargs):
            super().__init__(**kwargs)

            self.distill_lamb = distill_lamb
            D = self.features_dim

            # Frozen snapshots — initialised here, re-snapshotted at on_train_start
            # for every task > 0.
            self.frozen_encoder = copy.deepcopy(self.encoder)
            self.frozen_predictor = copy.deepcopy(self.predictor)
            for p in self.frozen_encoder.parameters():
                p.requires_grad = False
            for p in self.frozen_predictor.parameters():
                p.requires_grad = False

            # Learnable projector: maps current patch predictions → frozen space.
            # Applied independently per patch token (last dim D), so LayerNorm is
            # used instead of BatchNorm (which would mix across the patch axis).
            self.distill_projector = nn.Sequential(
                nn.Linear(D, distill_proj_hidden_dim),
                nn.LayerNorm(distill_proj_hidden_dim),
                nn.GELU(),
                nn.Linear(distill_proj_hidden_dim, D),
            )

        # ------------------------------------------------------------------
        # Lifecycle
        # ------------------------------------------------------------------

        def on_train_start(self):
            super().on_train_start()
            if self.current_task_idx > 0:
                # Snapshot the encoder and predictor at the start of each new task.
                self.frozen_encoder = copy.deepcopy(self.encoder)
                self.frozen_predictor = copy.deepcopy(self.predictor)
                for p in self.frozen_encoder.parameters():
                    p.requires_grad = False
                for p in self.frozen_predictor.parameters():
                    p.requires_grad = False

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        @torch.no_grad()
        def _frozen_forward(
            self,
            imgs: torch.Tensor,
            masks_enc_list: List[torch.Tensor],
            masks_pred_list: List[torch.Tensor],
        ) -> torch.Tensor:
            """Forward pass through the frozen encoder + predictor."""
            context = self.frozen_encoder(imgs, masks=masks_enc_list)
            preds = self.frozen_predictor(context, masks_enc_list, masks_pred_list)
            return preds  # (nenc*npred*B, K_pred, D)

        # ------------------------------------------------------------------
        # Learnable parameters
        # ------------------------------------------------------------------

        @property
        def learnable_params(self) -> List[Dict[str, Any]]:
            return super().learnable_params + [
                {
                    "name": "distill_projector",
                    "params": self.distill_projector.parameters(),
                    "lr": self.lr if self.distill_lamb >= 1 else self.lr / self.distill_lamb,
                    "weight_decay": self.weight_decay,
                },
            ]

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------

        def training_step(
            self, batch: Sequence[Any], batch_idx: int
        ) -> Dict[str, Any]:
            out = super().training_step(batch, batch_idx)

            if self.current_task_idx == 0:
                return out

            # Reuse features already computed by IJEPA.training_step — no second
            # forward pass through encoder + predictor.
            current_preds = out["preds"]                    # (nenc*npred*B, K_pred, D)
            masks_enc_list = out["masks_enc_list"]
            masks_pred_list = out["masks_pred_list"]

            # Still need imgs for the frozen forward
            (_, imgs_list, _), _, _ = batch[f"task{self.current_task_idx}"]
            imgs = imgs_list[0]

            # Frozen model predictions — no gradients
            frozen_preds = self._frozen_forward(imgs, masks_enc_list, masks_pred_list)
            # shape: (nenc*npred*B, K_pred, D)

            # Project current predictions towards frozen space, applied per patch token
            *batch_dims, D = current_preds.shape
            projected = self.distill_projector(current_preds.view(-1, D)).view(*batch_dims, D)

            distill_loss = F.smooth_l1_loss(projected, frozen_preds)
            self.log("train_ijepa_distill_loss", distill_loss, on_epoch=True, sync_dist=True)

            return out["loss"] + self.distill_lamb * distill_loss

        # ------------------------------------------------------------------
        # Args
        # ------------------------------------------------------------------

        @staticmethod
        def add_model_specific_args(
            parent_parser: argparse.ArgumentParser,
        ) -> argparse.ArgumentParser:
            parser = parent_parser.add_argument_group("ijepa_distiller")
            parser.add_argument("--distill_lamb", type=float, default=1.0)
            parser.add_argument("--distill_proj_hidden_dim", type=int, default=2048)
            return parent_parser

    return IJEPADistillWrapper
