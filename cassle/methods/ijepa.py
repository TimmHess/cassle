import copy
from argparse import ArgumentParser
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from cassle.methods.base import BaseModel
from cassle.models.vision_transformer import VisionTransformerPredictor, apply_masks, repeat_interleave_batch
from cassle.utils.momentum import MomentumUpdater, initialize_momentum_params


class IJEPA(BaseModel):
    def __init__(
        self,
        pred_depth: int,
        pred_emb_dim: int,
        base_tau_momentum: float,
        final_tau_momentum: float,
        final_weight_decay: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.final_weight_decay = final_weight_decay

        assert self.base_model is None, (
            "IJEPA requires a ViT encoder (e.g. --encoder vit_base)."
        )

        # Target encoder: frozen EMA copy of the context encoder
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor
        num_patches = self.encoder.patch_embed.num_patches
        self.predictor = VisionTransformerPredictor(
            num_patches=num_patches,
            embed_dim=self.features_dim,
            predictor_embed_dim=pred_emb_dim,
            depth=pred_depth,
            num_heads=self.encoder.num_heads,
        )

        self.momentum_updater = MomentumUpdater(base_tau_momentum, final_tau_momentum)
        self.last_step = 0

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        return super().learnable_params + [
            {"name": "predictor", "params": self.predictor.parameters()},
        ]

    def base_forward(self, X: torch.Tensor) -> Dict:
        """Full (unmasked) forward with mean pooling — used by the online classifier and KNN."""
        feats = self.encoder(X, masks=None)  # (B, N, D)
        feats = feats.mean(dim=1)            # (B, D)
        return {"feats": feats}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> Dict[str, Any]:
        # Batch format from MaskCollator:
        #   (collated_items, masks_enc, masks_pred)
        # where collated_items = (indices, [imgs], targets)
        #       masks_enc  : (nenc,  B, K_enc)
        #       masks_pred : (npred, B, K_pred)
        (_, imgs_list, _), masks_enc, masks_pred = batch[f"task{self.current_task_idx}"]
        imgs = imgs_list[0]  # (B, C, H, W)

        # Convert (nenc/npred, B, K) → list of (B, K) tensors expected by the ViT
        masks_enc_list = [masks_enc[i] for i in range(masks_enc.shape[0])]
        masks_pred_list = [masks_pred[i] for i in range(masks_pred.shape[0])]

        B = imgs.shape[0]

        # Context encoder: encode only the context patches
        context_feats = self.encoder(imgs, masks=masks_enc_list)  # (nenc*B, K_enc, D)

        # Predictor: predict the target patches from context
        preds = self.predictor(context_feats, masks_enc_list, masks_pred_list)  # (nenc*npred*B, K_pred, D)

        # Target encoder: encode the FULL image (no masks), then select target patch positions
        with torch.no_grad():
            target_feats = self.target_encoder(imgs, masks=None)  # (B, N, D)
            target_feats = F.layer_norm(target_feats, (target_feats.size(-1),))
            target_feats = apply_masks(target_feats, masks_pred_list)  # (npred*B, K_pred, D)
            # Repeat so each enc mask has a corresponding target
            target_feats = repeat_interleave_batch(target_feats, B, repeat=len(masks_enc_list))

        loss = F.smooth_l1_loss(preds, target_feats)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        return {"loss": loss}

    def on_train_start(self):
        self.last_step = 0
        if self.no_schedule_restart and self.current_task_idx > 0:
            # Skip tau annealing restart: target encoder is already well-trained
            self.momentum_updater.cur_tau = self.momentum_updater.final_tau
            # Skip weight-decay ramp restart: start at the final value
            if self.final_weight_decay is not None:
                for pg in self.optimizers().param_groups:
                    if pg.get("weight_decay", 0) > 0:
                        pg["weight_decay"] = self.final_weight_decay

    def on_train_batch_end(
        self, outputs: Dict[str, Any], batch: Sequence[Any], batch_idx: int
    ):
        if self.trainer.global_step > self.last_step:
            max_steps = (
                self.iters_per_task
                if self.iters_per_task
                else len(self.trainer.train_dataloader) * self.trainer.max_epochs // self.trainer.accumulate_grad_batches
            )
            self.momentum_updater.update(self.encoder, self.target_encoder)
            self.momentum_updater.update_tau(
                cur_step=self.trainer.global_step,
                max_steps=max_steps,
            )
            if self.final_weight_decay is not None:
                progress = min(self.trainer.global_step / max_steps, 1.0)
                wd_start = self.final_weight_decay if (self.no_schedule_restart and self.current_task_idx > 0) else self.weight_decay
                wd = wd_start + (self.final_weight_decay - wd_start) * progress
                for pg in self.optimizers().param_groups:
                    if pg.get("weight_decay", 0) > 0:
                        pg["weight_decay"] = wd
        self.last_step = self.trainer.global_step

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parent_parser = super(IJEPA, IJEPA).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("ijepa")

        # Predictor
        parser.add_argument("--pred_depth", type=int, default=6)
        parser.add_argument("--pred_emb_dim", type=int, default=384)

        # Target encoder EMA
        parser.add_argument("--base_tau_momentum", type=float, default=0.996)
        parser.add_argument("--final_tau_momentum", type=float, default=1.0)
        parser.add_argument("--final_weight_decay", type=float, default=None,
                            help="If set, linearly increase weight decay from --weight_decay to "
                                 "this value over training (I-JEPA paper: 0.04 -> 0.4).")

        # Mask collator (used in main_pretrain.py to build the DataLoader collate_fn)
        parser.add_argument("--enc_mask_scale", type=float, nargs=2, default=[0.85, 1.0])
        parser.add_argument("--pred_mask_scale", type=float, nargs=2, default=[0.15, 0.2])
        parser.add_argument("--aspect_ratio", type=float, nargs=2, default=[0.75, 1.5])
        parser.add_argument("--nenc", type=int, default=1)
        parser.add_argument("--npred", type=int, default=4)
        parser.add_argument("--min_keep", type=int, default=10)
        parser.add_argument("--allow_overlap", action="store_true")

        return parent_parser
