"""Florence-2 wrapper — Microsoft's unified vision-language foundation model.

Reference:
    Xiao et al. "Florence-2: Advancing a Unified Representation for a
    Variety of Vision Tasks." Microsoft, 2024.
    https://huggingface.co/microsoft/Florence-2-large

A single model that does:
    * <CAPTION>            — image captioning
    * <DETAILED_CAPTION>
    * <OD>                 — object detection
    * <DENSE_REGION_CAPTION>
    * <REGION_TO_SEGMENTATION>
    * <REFERRING_EXPRESSION_SEGMENTATION>

We expose two of the most useful entry points: ``detect`` (OD) and
``caption`` (caption). For richer tasks call ``run_task(<TASK>, ...)``
directly.
"""

from __future__ import annotations

import numpy as np

from mvmm.tracking.detectors import Detections


class Florence2:
    """Lazy-loaded Florence-2 wrapper.

    Args:
        model_id: HuggingFace model id, e.g.
            * ``microsoft/Florence-2-base-ft``  (smaller / faster)
            * ``microsoft/Florence-2-large-ft`` (default in upstream)
        device:   "cuda" or "cpu".
    """

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-base-ft",
        device: str = "cuda",
        torch_dtype: str = "float16",
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
        except ImportError as e:
            raise ImportError("transformers is required — `pip install transformers`") from e

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        dtype = torch.float16 if (torch_dtype == "float16" and device == "cuda") else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
            .to(device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def run_task(
        self,
        image_rgb: np.ndarray,
        task: str = "<OD>",
        text_input: str | None = None,
        max_new_tokens: int = 1024,
    ) -> dict:
        import torch
        from PIL import Image

        pil = Image.fromarray(image_rgb)
        prompt = task if text_input is None else f"{task}{text_input}"
        inputs = self.processor(text=prompt, images=pil, return_tensors="pt").to(
            self.device, self.model.dtype
        )
        with torch.no_grad():
            gen = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=3,
            )
        text = self.processor.batch_decode(gen, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(text, task=task, image_size=pil.size)
        return parsed[task]

    def detect(self, image_rgb: np.ndarray) -> Detections:
        """Run <OD> and return our standard Detections container."""
        out = self.run_task(image_rgb, task="<OD>")
        boxes = np.asarray(out.get("bboxes", []), dtype=np.float32).reshape(-1, 4)
        labels_text = out.get("labels", [])
        # Florence-2 returns free-text labels; we map to indices via the
        # *unique* label list so downstream code stays type-safe.
        class_names: list[str] = []
        label_idx: list[int] = []
        for lt in labels_text:
            if lt not in class_names:
                class_names.append(lt)
            label_idx.append(class_names.index(lt))
        scores = np.ones(len(boxes), dtype=np.float32)  # Florence-2 doesn't emit scores
        return Detections(
            boxes=boxes,
            scores=scores,
            labels=np.asarray(label_idx, dtype=np.int64),
            class_names=class_names,
        )

    def caption(self, image_rgb: np.ndarray, level: str = "normal") -> str:
        task_map = {
            "short": "<CAPTION>",
            "normal": "<DETAILED_CAPTION>",
            "long": "<MORE_DETAILED_CAPTION>",
        }
        task = task_map.get(level, "<DETAILED_CAPTION>")
        out = self.run_task(image_rgb, task=task)
        if isinstance(out, str):
            return out
        return str(out)
