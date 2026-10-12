"""Shrink the classifier's Qwen3 embedding model to FP16, and check it.

    python scripts/quantize_classifier.py build
    python scripts/quantize_classifier.py evaluate <CreditMitra-Merchant-Classification>/data

`build` writes data/models/onnx_pipeline/qwen3_embedding.fp16.onnx next to the
FP32 model (needs `pip install onnx`, a build-time tool only). The classifier
loads the FP16 file whenever it is present.

`evaluate` labels the same 600 corpus names (300 person, 300 merchant) with
both models — the AttentionMLP and scaler on top are unchanged — and reports
agreement, accuracy against the corpus labels, and speed.

Why FP16 and not int8: on that check FP16 changed no labels (2.38 GB -> 1.19
GB, ~30% faster). Dynamic int8 was smaller (0.60 GB) but changed 4.7% of
labels per-tensor and still 3.2% per-channel, in both directions — too much for
a step that decides which payees count as merchants.
"""

from __future__ import annotations

import csv
import gc
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PIPELINE = ROOT / "data" / "models" / "onnx_pipeline"
FP32 = PIPELINE / "qwen3_embedding.onnx"
FP16 = PIPELINE / "qwen3_embedding.fp16.onnx"


def build() -> None:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    started = time.time()
    model = onnx.load(str(FP32), load_external_data=True)
    # keep_io_types: inputs/outputs stay FP32, so the scaler and MLP head are
    # fed exactly as before.
    half = convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)
    onnx.save(
        half,
        str(FP16),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{FP16.name}.data",
    )
    size = sum(p.stat().st_size for p in PIPELINE.glob(f"{FP16.name}*"))
    print(f"wrote {FP16.name} ({size / 1e9:.2f} GB) in {time.time() - started:.0f}s")


def _labels(model_path: Path, names: list[str]) -> tuple[list[str], float]:
    from app.services import merchant_classifier as mc

    service = mc.MerchantClassifierService.__new__(mc.MerchantClassifierService)
    service._load_models(embedding_path=str(model_path))
    started = time.time()
    labels = [service._onnx_classify(n)["label"] for n in names]
    per_name = (time.time() - started) / len(names)
    del service
    gc.collect()
    return labels, per_name


def evaluate(corpus_dir: Path, per_class: int = 300) -> None:
    random.seed(7)
    with (corpus_dir / "indian_person_names.csv").open(encoding="utf-8") as fh:
        persons = random.sample([r["name"] for r in csv.DictReader(fh)], per_class)
    with (corpus_dir / "indian_merchant_names.csv").open(encoding="utf-8") as fh:
        merchants = random.sample([r["name"] for r in csv.DictReader(fh)], per_class)
    names = persons + merchants
    truth = ["person"] * per_class + ["merchant"] * per_class

    full, full_speed = _labels(FP32, names)
    half, half_speed = _labels(FP16, names)

    def acc(pred: list[str]) -> float:
        return sum(p == t for p, t in zip(pred, truth, strict=True)) / len(truth)

    changed: dict[str, int] = {}
    for a, b in zip(full, half, strict=True):
        if a != b:
            changed[f"{a}->{b}"] = changed.get(f"{a}->{b}", 0) + 1
    agree = 1 - sum(changed.values()) / len(names)
    print(f"names: {len(names)} ({per_class} person, {per_class} merchant)")
    print(f"FP32 accuracy {acc(full):.3f}  {full_speed * 1000:.0f} ms/name")
    print(f"FP16 accuracy {acc(half):.3f}  {half_speed * 1000:.0f} ms/name")
    print(f"agreement {agree:.3f}  changed: {changed or 'none'}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build()
    elif sys.argv[1:2] == ["evaluate"]:
        evaluate(Path(sys.argv[2]))
    else:
        sys.exit(__doc__)
