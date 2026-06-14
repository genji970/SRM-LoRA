from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create pixel heatmaps from SR-LoRA soft-mask arrays.")
    p.add_argument("--visualization-dir", required=True, help="Directory containing arrays/; usually SR_VISUALIZATION_DIR/soft_mask")
    p.add_argument("--latest-only", action="store_true")
    p.add_argument("--max-plots", type=int, default=100000)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--poll-interval", type=float, default=2.0)
    p.add_argument("--stable-seconds", type=float, default=0.5)
    p.add_argument("--max-plots-per-file", type=int, default=1)
    return p.parse_args()


def numeric_step(path: Path) -> int:
    m = re.search(r"step[_-](\d+)", path.name)
    return int(m.group(1)) if m else -1


def array_files(arrays_dir: Path) -> list[Path]:
    if not arrays_dir.exists():
        return []
    out: list[Path] = []
    for p in arrays_dir.iterdir():
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".json", ".jsonl", ".txt", ".csv"}:
            continue
        out.append(p)
    return sorted(out, key=lambda x: (numeric_step(x), x.name))


def try_torch_load(path: Path) -> Any:
    import torch
    return torch.load(str(path), map_location="cpu")


def to_numpy(obj: Any) -> np.ndarray:
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().float().cpu().numpy()
    except Exception:
        pass

    if isinstance(obj, np.ndarray):
        return obj.astype(np.float32, copy=False)

    if isinstance(obj, dict):
        preferred = [
            "soft_mask",
            "mask",
            "pre_gradient_soft_mask",
            "softmask",
            "soft_mask_value",
            "arr_0",
        ]
        for k in preferred:
            if k in obj:
                return to_numpy(obj[k])
        for v in obj.values():
            try:
                arr = to_numpy(v)
                if arr.size:
                    return arr
            except Exception:
                continue

    if isinstance(obj, (list, tuple)):
        for v in obj:
            try:
                arr = to_numpy(v)
                if arr.size:
                    return arr
            except Exception:
                continue

    return np.asarray(obj, dtype=np.float32)


def load_array(path: Path) -> np.ndarray:
    errors: list[str] = []

    # np.load works for .npy/.npz and also many extensionless numpy files.
    try:
        loaded = np.load(str(path), allow_pickle=True)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            keys = list(loaded.files)
            preferred = ["soft_mask", "mask", "pre_gradient_soft_mask", "arr_0"]
            for k in preferred:
                if k in keys:
                    return to_numpy(loaded[k])
            if keys:
                return to_numpy(loaded[keys[0]])
        arr = to_numpy(loaded)
        if arr.dtype == object and arr.shape == ():
            return to_numpy(arr.item())
        return arr
    except Exception as e:
        errors.append(f"np.load: {type(e).__name__}: {e}")

    try:
        return to_numpy(try_torch_load(path))
    except Exception as e:
        errors.append(f"torch.load: {type(e).__name__}: {e}")

    raise RuntimeError("; ".join(errors))


def squeeze_to_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        # Keep the last two dimensions as the mask image; flatten all leading dims into rows.
        arr = arr.reshape(-1, arr.shape[-1])
    return arr


def normalize_minmax(raw: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    raw = np.asarray(raw, dtype=np.float32)
    mn = float(np.min(raw)) if raw.size else 0.0
    mx = float(np.max(raw)) if raw.size else 0.0
    std = float(np.std(raw)) if raw.size else 0.0
    denom = mx - mn
    if denom <= 1e-12:
        norm = np.zeros_like(raw, dtype=np.float32)
    else:
        norm = (raw - mn) / denom
    return np.clip(norm, 0.0, 1.0), mn, mx, std


def clean_title(path: Path) -> str:
    name = path.name
    step_m = re.search(r"step[_-](\d+)", name)
    step = f"step_{int(step_m.group(1)):06d}" if step_m else path.stem

    layer_m = re.search(r"layers\.(\d+)", name)
    layer = f"layer {layer_m.group(1)}" if layer_m else "layer ?"

    module = ""
    mod_m = re.search(r"\.mlp\.([A-Za-z0-9_]+)\.", name)
    if mod_m:
        module = f"mlp: {mod_m.group(1)}"
    else:
        for cand in ("gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"):
            if cand in name:
                module = f"mlp: {cand}" if cand in {"gate_proj", "up_proj", "down_proj"} else cand
                break

    lora = "lora_B" if "lora_B" in name else ("lora_A" if "lora_A" in name else "")
    parts = [step, layer]
    if module:
        parts.append(module)
    if lora:
        parts.append(lora)
    return " | ".join(parts)


def output_name(path: Path) -> str:
    stem = path.name
    for suf in (".npz", ".npy", ".pt", ".pth", ".safetensors"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return f"{stem}__minmax_softmask_heatmap.png"


def make_heatmap(path: Path, heatmaps_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    raw = squeeze_to_2d(load_array(path))
    vis, mn, mx, std = normalize_minmax(raw)

    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    out_path = heatmaps_dir / output_name(path)

    fig, ax = plt.subplots(figsize=(10.8, 7.0), constrained_layout=True)
    im = ax.imshow(vis, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(clean_title(path), fontsize=13, pad=10)
    ax.set_xlabel("mask column")
    ax.set_ylabel("mask row")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    cbar.set_label("")

    stats = f"raw min={mn:.6f}\nraw max={mx:.6f}\nraw std={std:.6f}\nvisualized=min-max normalized"
    cbar.ax.text(
        2.15,
        0.5,
        stats,
        transform=cbar.ax.transAxes,
        va="center",
        ha="left",
        fontsize=10,
    )

    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out_path


def is_stable(path: Path, stable_seconds: float) -> bool:
    try:
        return time.time() - path.stat().st_mtime >= stable_seconds
    except FileNotFoundError:
        return False


def run_once(vis_dir: Path, latest_only: bool, max_plots: int, stable_seconds: float = 0.0) -> list[str]:
    arrays_dir = vis_dir / "arrays"
    heatmaps_dir = vis_dir / "heatmaps"
    files = array_files(arrays_dir)
    if latest_only and files:
        files = [files[-1]]
    if max_plots > 0:
        files = files[-max_plots:]

    made: list[str] = []
    for f in files:
        if stable_seconds > 0 and not is_stable(f, stable_seconds):
            continue
        out = heatmaps_dir / output_name(f)
        if out.exists() and out.stat().st_mtime >= f.stat().st_mtime:
            continue
        try:
            made_path = make_heatmap(f, heatmaps_dir)
            made.append(str(made_path))
            print(f"[softmask-heatmap] saved {made_path}", flush=True)
        except Exception as e:
            print(f"[softmask-heatmap] skipped {f}: {type(e).__name__}: {e}", flush=True)

    manifest = {
        "visualization_dir": str(vis_dir),
        "arrays_dir": str(arrays_dir),
        "heatmaps_dir": str(heatmaps_dir),
        "mode": "minmax_normalized_soft_mask_pixel_heatmap",
        "colorbar": "0..1 after per-file min-max normalization",
        "stats": "raw min/max/std are printed next to the colorbar",
        "created_or_updated": made,
    }
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    (heatmaps_dir / "heatmap_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return made


def main() -> None:
    args = parse_args()
    vis_dir = Path(args.visualization_dir)
    if args.watch:
        print(f"[softmask-heatmap] watching {vis_dir / 'arrays'}", flush=True)
        while True:
            run_once(vis_dir, latest_only=False, max_plots=args.max_plots, stable_seconds=args.stable_seconds)
            time.sleep(args.poll_interval)
    else:
        run_once(vis_dir, latest_only=args.latest_only, max_plots=args.max_plots, stable_seconds=0.0)


if __name__ == "__main__":
    main()
