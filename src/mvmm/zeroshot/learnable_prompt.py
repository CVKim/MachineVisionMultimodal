"""AnomalyCLIP-style learnable prompts on top of a frozen CLIP backbone.

Reference:
    Zhou et al. "AnomalyCLIP: Object-agnostic Prompt Learning for
    Zero-shot Anomaly Detection." ICLR 2024.
    https://arxiv.org/abs/2310.18961

Idea:
    Hand-crafted prompts ("a clean wafer", "a wafer with a scratch")
    plateau quickly — the *exact* phrasing matters a lot, and you can't
    enumerate every defect mode. AnomalyCLIP replaces the prompt
    embeddings with `N` learnable *context tokens* that are trained on
    a tiny (few-shot) labeled set. The CLIP image and text encoders
    remain *frozen*, so the parameter budget is tiny and overfitting
    is constrained.

Architecture (per class, two prompts: "normal" / "anomalous")::

    [SOS] [Vc1] [Vc2] ... [VcN] [CLS_template]  -> CLIP text encoder
            └────── learnable D-dim vectors ─────┘

We expose:
    * ``LearnablePromptCLIP``  — model class
    * ``fit_learnable_prompts`` — tiny few-shot training loop
    * ``LearnableAnomalyCLIP``  — drop-in replacement for AnomalyCLIP
                                  that uses the trained context tokens
                                  at inference time
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mvmm.anomaly.base import AnomalyDetector, AnomalyResult


@dataclass
class LearnablePromptConfig:
    n_ctx: int = 12  # number of learnable context tokens
    ctx_init: str = ""  # if non-empty, init context from these tokens
    class_token_position: str = "end"  # "end" or "middle"
    clip_model: str = "ViT-B-16"
    pretrained: str = "openai"
    device: str = "cuda"


class LearnablePromptCLIP(nn.Module):
    """Frozen CLIP + N learnable context tokens shared across all classes."""

    def __init__(
        self,
        classnames: list[str],
        cfg: LearnablePromptConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg or LearnablePromptConfig()
        if self.cfg.device == "cuda" and not torch.cuda.is_available():
            self.cfg.device = "cpu"

        import open_clip  # type: ignore

        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            self.cfg.clip_model, pretrained=self.cfg.pretrained, device=self.cfg.device
        )
        self.tokenizer = open_clip.get_tokenizer(self.cfg.clip_model)
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.clip.eval()

        self.classnames = list(classnames)
        self.embed_dim = self.clip.token_embedding.embedding_dim

        # Initialize learnable context.
        if self.cfg.ctx_init:
            with torch.no_grad():
                tok = self.tokenizer(self.cfg.ctx_init).to(self.cfg.device)
                emb = self.clip.token_embedding(tok).squeeze(0)
                # take the first n_ctx non-padding tokens after SOS
                ctx_vectors = emb[1 : 1 + self.cfg.n_ctx].clone()
        else:
            ctx_vectors = torch.empty(self.cfg.n_ctx, self.embed_dim, device=self.cfg.device)
            nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        # Pre-compute non-learnable parts: [SOS] + [classname tokens] + [EOS] padding.
        self._build_class_token_ids()

    def _build_class_token_ids(self):
        """Tokenize each classname; store (suffix_embeddings, sos_embedding)."""
        prompts = ["X " * self.cfg.n_ctx + name + "." for name in self.classnames]
        tokenized = self.tokenizer(prompts).to(self.cfg.device)
        with torch.no_grad():
            emb = self.clip.token_embedding(tokenized)
        # split into [SOS, ctx_placeholder * n_ctx, name_tokens, EOS, pad...]
        self.register_buffer("token_prefix", emb[:, :1, :])  # SOS
        self.register_buffer("token_suffix", emb[:, 1 + self.cfg.n_ctx :, :])  # name + EOS + pad
        self.register_buffer("tokenized", tokenized)

    def build_prompt_embeddings(self) -> torch.Tensor:
        """Splice the learnable context into each class's token sequence."""
        ctx = self.ctx.unsqueeze(0).expand(len(self.classnames), -1, -1)
        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        return prompts  # (n_class, ctx_len + ..., D)

    def encode_text(self) -> torch.Tensor:
        """Run CLIP text transformer on the assembled embeddings."""
        prompts = self.build_prompt_embeddings()
        # open_clip's text encoder expects token *embeddings* (B, T, D);
        # we call through its transformer manually.
        x = prompts + self.clip.positional_embedding.to(prompts.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.clip.ln_final(x)
        # take features at the EOS position
        eos_pos = self.tokenized.argmax(dim=-1)
        out = x[torch.arange(x.shape[0]), eos_pos] @ self.clip.text_projection
        return F.normalize(out, dim=-1)

    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        feat = self.clip.encode_image(image_tensor.to(self.cfg.device))
        return F.normalize(feat, dim=-1)

    def logits(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Return (B, n_class) logits — scaled cosine similarities."""
        img = self.encode_image(image_tensor)
        txt = self.encode_text()
        return 100.0 * img @ txt.T


def fit_learnable_prompts(
    model: LearnablePromptCLIP,
    images: list[np.ndarray],
    labels: list[int],
    epochs: int = 30,
    lr: float = 5e-3,
    batch_size: int = 8,
) -> list[float]:
    """Tiny few-shot trainer for the learnable context.

    Args:
        model:     :class:`LearnablePromptCLIP` instance.
        images:    list of RGB numpy arrays (uint8, HxWx3).
        labels:    same length, integer class id.
        epochs:    full passes over the few-shot set.
        lr:        learning rate (the only trainable param is ``model.ctx``).
        batch_size: small — few-shot regime.

    Returns:
        list of per-epoch mean cross-entropy losses.
    """
    from PIL import Image

    device = model.cfg.device
    opt = torch.optim.AdamW([model.ctx], lr=lr)
    crit = nn.CrossEntropyLoss()

    # Preprocess images once.
    x = torch.stack([model.preprocess(Image.fromarray(img)) for img in images]).to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)

    losses: list[float] = []
    n = len(images)
    for _ep in range(epochs):
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            logits = model.logits(x[idx])
            loss = crit(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss) * len(idx)
        losses.append(ep_loss / max(n, 1))
    return losses


class LearnableAnomalyCLIP(AnomalyDetector):
    """Drop-in AD detector backed by a trained LearnablePromptCLIP.

    Uses class index 0 = "normal", class index 1 = "anomalous". Returns
    softmax probability of the anomalous class as the image score, and
    falls back to a uniform map for the pixel-level score (the original
    AnomalyCLIP also runs a separate pixel-decoder; we expose the image
    branch here and refer to the upstream codebase for the pixel head).
    """

    name = "learnable-anomaly-clip"

    def __init__(self, model: LearnablePromptCLIP):
        if len(model.classnames) != 2:
            raise ValueError("LearnableAnomalyCLIP expects exactly 2 classes: [normal, anomalous]")
        self.model = model

    def fit(self, _loader) -> None:
        return None  # train via fit_learnable_prompts(model, images, labels)

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> AnomalyResult:
        logits = self.model.logits(images)
        probs = F.softmax(logits, dim=-1)[:, 1]
        h, w = images.shape[-2:]
        score_maps = probs.view(-1, 1, 1).expand(-1, h, w).cpu().numpy().astype(np.float32)
        return AnomalyResult(
            image_scores=probs.cpu().numpy().astype(np.float32),
            score_maps=score_maps,
        )

    def save(self, path):
        torch.save({"ctx": self.model.ctx.detach().cpu(), "classnames": self.model.classnames}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.model.cfg.device, weights_only=False)
        with torch.no_grad():
            self.model.ctx.copy_(ckpt["ctx"].to(self.model.cfg.device))
