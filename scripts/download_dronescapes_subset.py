"""Download a bounded, scene-balanced Dronescapes Experts training subset."""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("datasets/dronescapes_experts_subset"))
    parser.add_argument("--repo", default="Meehai/dronescapes2")
    parser.add_argument("--rgb-only", action="store_true")
    parser.add_argument("--per-scene", type=int, default=24)
    parser.add_argument("--scenes", nargs="+", default=["herculane", "jupiter", "petrova"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    repo = args.repo
    siblings = HfApi().dataset_info(repo).siblings
    available = {s.rfilename for s in siblings}
    groups = defaultdict(list)
    prefix = "data/semisupervised_set/rgb/"
    for path in sorted(available):
        if path.startswith(prefix) and path.endswith(".npz"):
            scene = Path(path).name.split("_", 1)[0]
            if scene in args.scenes:
                depth = path.replace("/rgb/", "/depth_marigold/")
                if args.rgb_only or depth in available:
                    groups[scene].append((path, depth))
    pairs = []
    for scene in args.scenes:
        candidates = groups[scene]
        if not candidates:
            raise RuntimeError(f"No paired RGB/depth for {scene}")
        count = min(args.per_scene, len(candidates))
        indices = [round(i * (len(candidates) - 1) / max(count - 1, 1)) for i in range(count)]
        pairs.extend((scene, *candidates[i]) for i in indices)
    files = sorted({p for _, rgb, depth in pairs for p in ((rgb,) if args.rgb_only else (rgb, depth))})
    args.output.mkdir(parents=True, exist_ok=True)
    def download(path):
        return hf_hub_download(repo_id=repo, repo_type="dataset", filename=path,
                               local_dir=args.output)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, future in enumerate(as_completed([pool.submit(download, path) for path in files]), 1):
            future.result()
            print(f"downloaded {index}/{len(files)}", flush=True)
    manifest = {
        "repo": repo, "split": "semisupervised_set", "depth_type": None if args.rgb_only else "depth_marigold",
        "note": "RGB-only frames have no frame-selection action labels or calibrated poses." if args.rgb_only else
                "Generated monocular depth is a pseudo-label, not metric ground truth.",
        "pairs": [{"scene": scene, "rgb": rgb, "depth": None if args.rgb_only else depth} for scene, rgb, depth in pairs],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"verified manifest: {args.output / 'manifest.json'}; {len(pairs)} pairs", flush=True)


if __name__ == "__main__":
    main()
