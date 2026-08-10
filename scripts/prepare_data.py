import argparse
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

# Allow `python scripts/prepare_data.py` (repo root not on sys.path in that
# case) in addition to `python -m scripts.prepare_data`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import get_logger, get_project_root

logger = get_logger("prepare_data")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CAPTION_MODEL_ID = "microsoft/Florence-2-base-ft"
CAPTION_TASK_PROMPT = "<MORE_DETAILED_CAPTION>"


@dataclass
class Rejection:
    path: Path
    reason: str


def discover_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def init_face_analyzer():
    from insightface.app import FaceAnalysis

    analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    analyzer.prepare(ctx_id=-1, det_size=(640, 640))
    return analyzer


def load_image_bgr(path: Path) -> np.ndarray:
    pil_img = Image.open(path)
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def detect_and_crop_face(
    img_bgr: np.ndarray, analyzer, resolution: int, margin: float
) -> tuple[np.ndarray | None, str | None]:
    """Returns (crop, None) on success, or (None, rejection_reason) on failure."""
    faces = analyzer.get(img_bgr)
    if not faces:
        return None, "no_face_detected"

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = face.bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    face_side = max(x2 - x1, y2 - y1)

    h, w = img_bgr.shape[:2]
    if face_side > min(h, w):
        return None, "face_too_large_for_frame"

    # Prefer `margin`x the face size, but clamp to the largest square that
    # fits in the frame (e.g. close-up photos) rather than rejecting.
    side = min(face_side * margin, min(h, w))

    side_i = int(round(side))
    half = side / 2
    left_i = int(round(min(max(cx - half, 0), w - side)))
    top_i = int(round(min(max(cy - half, 0), h - side)))

    crop = img_bgr[top_i : top_i + side_i, left_i : left_i + side_i]
    # INTER_AREA is correct for downsampling but degrades to near-nearest-
    # neighbor when enlarging (OpenCV docs); use INTER_CUBIC when the crop
    # is smaller than the target resolution.
    interpolation = cv2.INTER_AREA if side_i >= resolution else cv2.INTER_CUBIC
    return cv2.resize(crop, (resolution, resolution), interpolation=interpolation), None


def compute_blur_score(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_phash(img_bgr: np.ndarray) -> imagehash.ImageHash:
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_img)


def is_near_duplicate(
    phash: imagehash.ImageHash, seen_hashes: list[imagehash.ImageHash], threshold: int
) -> bool:
    return any(phash - seen <= threshold for seen in seen_hashes)


def load_captioner():
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(CAPTION_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        CAPTION_MODEL_ID,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.float32,
    )
    model.eval()
    return processor, model


def caption_image(pil_img: Image.Image, processor, model, trigger_phrase: str) -> str:
    inputs = processor(text=CAPTION_TASK_PROMPT, images=pil_img, return_tensors="pt")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=256,
        num_beams=3,
        do_sample=False,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    parsed = processor.post_process_generation(
        generated_text, task=CAPTION_TASK_PROMPT, image_size=pil_img.size
    )
    caption = parsed[CAPTION_TASK_PROMPT].strip()
    return f"{trigger_phrase}{caption}"


def run_crop_and_filter(
    images: list[Path],
    output_dir: Path,
    analyzer,
    resolution: int,
    blur_threshold: float,
    hash_threshold: int,
) -> tuple[list[Path], list[Rejection]]:
    accepted_paths: list[Path] = []
    seen_hashes: list[imagehash.ImageHash] = []
    rejections: list[Rejection] = []

    for idx, src_path in enumerate(tqdm(images, desc="crop/filter"), start=1):
        try:
            img_bgr = load_image_bgr(src_path)
        except Exception as e:
            rejections.append(Rejection(src_path, f"unreadable ({e})"))
            continue

        crop, reject_reason = detect_and_crop_face(img_bgr, analyzer, resolution, margin=2.2)
        if crop is None:
            rejections.append(Rejection(src_path, reject_reason))
            continue

        if compute_blur_score(crop) < blur_threshold:
            rejections.append(Rejection(src_path, "blurry"))
            continue

        phash = compute_phash(crop)
        if is_near_duplicate(phash, seen_hashes, hash_threshold):
            rejections.append(Rejection(src_path, "near_duplicate"))
            continue

        out_path = output_dir / f"{idx:04d}_{src_path.stem}.png"
        if not cv2.imwrite(str(out_path), crop):
            rejections.append(Rejection(src_path, "write_failed"))
            continue
        seen_hashes.append(phash)
        accepted_paths.append(out_path)

    return accepted_paths, rejections


def run_captioning(accepted_paths: list[Path], trigger_phrase: str) -> None:
    if not accepted_paths:
        return
    processor, model = load_captioner()
    for img_path in tqdm(accepted_paths, desc="captioning"):
        pil_img = Image.open(img_path).convert("RGB")
        caption = caption_image(pil_img, processor, model, trigger_phrase)
        img_path.with_suffix(".txt").write_text(caption, encoding="utf-8")


def main() -> None:
    root = get_project_root()
    parser = argparse.ArgumentParser(description="Face crop, filter, and caption training photos.")
    parser.add_argument("--input-dir", type=Path, default=root / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=root / "data/processed")
    parser.add_argument("--trigger-phrase", default="a photo of ohwx man, ")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--blur-threshold", type=float, default=30.0)
    parser.add_argument("--hash-threshold", type=int, default=5)
    parser.add_argument("--skip-caption", action="store_true")
    args = parser.parse_args()

    images = discover_images(args.input_dir)
    logger.info(f"Found {len(images)} candidate images in {args.input_dir}")
    if not images:
        logger.info("Nothing to do.")
        return

    # data/processed is a derived/regenerated artifact directory: clear it so
    # reruns (e.g. after adding new raw photos) don't mix stale output with
    # new output, shift index prefixes, or bypass cross-run dedup.
    if args.output_dir.exists():
        for stale in args.output_dir.glob("*"):
            if stale.name != ".gitkeep":
                stale.unlink()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = init_face_analyzer()
    accepted_paths, rejections = run_crop_and_filter(
        images, args.output_dir, analyzer, args.resolution, args.blur_threshold, args.hash_threshold
    )

    if args.skip_caption:
        logger.info("Skipping captioning (--skip-caption)")
    else:
        run_captioning(accepted_paths, args.trigger_phrase)

    reason_counts = Counter(r.reason for r in rejections)
    logger.info(f"Accepted: {len(accepted_paths)} / {len(images)}")
    logger.info(f"Rejected: {len(rejections)} ({dict(reason_counts)})")
    for r in rejections:
        logger.info(f"  rejected {r.path.name}: {r.reason}")


if __name__ == "__main__":
    main()
