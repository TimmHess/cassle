# Adapted from the I-JEPA repository (Meta Platforms, Inc.)
# https://github.com/facebookresearch/ijepa

import math
from logging import getLogger
from multiprocessing import Value

import torch

logger = getLogger()


class MaskCollator:
    """
    Collate function that generates I-JEPA context and target block masks for each batch.

    The collated batch is a tuple:
        (original_collated_batch, masks_enc, masks_pred)

    where
        masks_enc  has shape (nenc,  B, K_enc)
        masks_pred has shape (npred, B, K_pred)
    and K_enc / K_pred are the (variable) number of patches kept per mask,
    truncated to the minimum across the batch so tensors can be stacked.
    """

    def __init__(
        self,
        input_size=224,
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=10,
        allow_overlap=False,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.patch_size = patch_size
        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self._itr_counter = Value("i", -1)  # shared across worker processes

    def step(self):
        with self._itr_counter.get_lock():
            self._itr_counter.value += 1
            return self._itr_counter.value

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        min_ar, max_ar = aspect_ratio_scale
        ar = min_ar + rand * (max_ar - min_ar)
        h = int(round(math.sqrt(max_keep * ar)))
        w = int(round(math.sqrt(max_keep / ar)))
        h = min(h, self.height - 1)
        w = min(w, self.width - 1)
        return h, w

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size
        tries = 0
        timeout = 20
        valid_mask = False
        while not valid_mask:
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top + h, left:left + w] = 1
            if acceptable_regions is not None:
                n_regions = max(int(len(acceptable_regions) - tries), 0)
                for k in range(n_regions):
                    mask *= acceptable_regions[k]
            mask = torch.nonzero(mask.flatten())
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = 20
                    logger.warning(
                        f"MaskCollator: valid mask not found, relaxing constraints [{tries}]"
                    )
        mask = mask.squeeze()
        complement = torch.ones((self.height, self.width), dtype=torch.int32)
        complement[top:top + h, left:left + w] = 0
        return mask, complement

    def __call__(self, batch):
        collated_batch = torch.utils.data.default_collate(batch)
        B = len(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(g, self.pred_mask_scale, self.aspect_ratio)
        e_size = self._sample_block_size(g, self.enc_mask_scale, (1.0, 1.0))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width

        for _ in range(B):
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, complement = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(complement)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = None if self.allow_overlap else masks_C
            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        # Stack to (npred, B, K_pred) and (nenc, B, K_enc)
        collated_masks_pred = torch.stack([
            torch.stack([collated_masks_pred[b][i][:min_keep_pred] for b in range(B)])
            for i in range(self.npred)
        ])

        collated_masks_enc = torch.stack([
            torch.stack([collated_masks_enc[b][i][:min_keep_enc] for b in range(B)])
            for i in range(self.nenc)
        ])

        return collated_batch, collated_masks_enc, collated_masks_pred
