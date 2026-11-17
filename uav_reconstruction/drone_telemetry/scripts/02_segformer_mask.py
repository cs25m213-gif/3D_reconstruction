#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_images")
    parser.add_argument("output_masks")

    parser.add_argument(
        "--model",
        default="nvidia/segformer-b0-finetuned-ade-512-512"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_images = Path(args.output_images)
    output_masks = Path(args.output_masks)

    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Loading:", args.model)

    processor = SegformerImageProcessor.from_pretrained(args.model)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model)
    model.to(device)
    model.eval()

    id2label = {
        int(k): str(v).lower()
        for k, v in model.config.id2label.items()
    }

    # We intentionally KEEP:
    # water, ocean, sky, vegetation, terrain, buildings,
    # roads, vehicles, rocks, beach, etc.
    #
    # Only living human/animal classes are suppressed.

    remove_words = {
        "person",
        "people",
        "human",
        "animal",
        "dog",
        "cat",
        "bird",
        "horse",
        "cow",
        "cattle",
        "sheep",
        "goat"
    }

    remove_ids = []

    print("\nSelected suppression classes:")

    for class_id, label in sorted(id2label.items()):
        if any(word in label for word in remove_words):
            remove_ids.append(class_id)
            print(f"  {class_id:3d} : {label}")

    if not remove_ids:
        print("WARNING: No matching suppression classes found.")

    images = sorted(
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.jpeg")) +
        list(input_dir.glob("*.png"))
    )

    print("\nImages:", len(images))

    for n, image_path in enumerate(images, 1):

        image = Image.open(image_path).convert("RGB")
        rgb = np.asarray(image)
        h, w = rgb.shape[:2]

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            logits = model(**inputs).logits

        logits = F.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        segmentation = (
            logits.argmax(dim=1)[0]
            .cpu()
            .numpy()
        )

        if remove_ids:
            remove = np.isin(segmentation, remove_ids)
        else:
            remove = np.zeros((h, w), dtype=bool)

        # COLMAP convention:
        # white = usable
        # black = ignored

        mask = np.where(remove, 0, 255).astype(np.uint8)

        cleaned = rgb.copy()
        cleaned[remove] = 0

        Image.fromarray(cleaned).save(
            output_images / image_path.name,
            quality=95
        )

        # COLMAP mask naming:
        # frame_000001.jpg -> frame_000001.jpg.png

        mask_name = image_path.name + ".png"

        Image.fromarray(mask).save(
            output_masks / mask_name
        )

        removed_pct = 100.0 * remove.mean()

        print(
            f"\r{n}/{len(images)} "
            f"{image_path.name} "
            f"masked={removed_pct:.2f}%",
            end="",
            flush=True
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
