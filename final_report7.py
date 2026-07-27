import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from train_autoencoder7 import build_model
from dataset_loader import ImageFolderDataset

torch.set_default_dtype(torch.float64)


def find_checkpoint(run_dir):
    for name in ("model_FINAL.pth",):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            return p
    best = sorted(glob.glob(os.path.join(run_dir, "model_epoch_*_BEST.pth")))
    if best:
        return best[-1]
    epochs = sorted(glob.glob(os.path.join(run_dir, "model_epoch_*.pth")),
                     key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p).split('_')[2]))))
    if epochs:
        return epochs[-1]
    raise FileNotFoundError(f"Не знайшов жодного чекпоінта в {run_dir}")


def latest_run_dir(runs_root="runs"):
    dirs = sorted(glob.glob(os.path.join(runs_root, "run_*")))
    if not dirs:
        raise FileNotFoundError("Немає жодної папки runs/run_*")
    return dirs[-1]


def tensor_to_uint8(img):
    arr = img.detach().cpu().numpy()
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, default=None, help="За замовч. -- найновіша runs/run_*")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--data-dir", type=str, default="data2")
    p.add_argument("--num-qubits", type=int, default=10)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--bottleneck", type=int, default=256)
    p.add_argument("--patch-mode", action="store_true")
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--color", action="store_true")
    p.add_argument("--smooth-conv", action="store_true")
    p.add_argument("--smooth-channels", type=int, default=16)
    p.add_argument("--smooth-layers", type=int, default=2)
    p.add_argument("--use-wavelet", action="store_true")
    p.add_argument("--upscale", type=int, default=32)
    p.add_argument("--out", type=str, default="final_report")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"DEBUG: use_wavelet={args.use_wavelet} | smooth_conv={args.smooth_conv} | "
          f"smooth_channels={args.smooth_channels} | smooth_layers={args.smooth_layers} | "
          f"num_qubits={args.num_qubits} | num_layers={args.num_layers}")
    run_dir = args.run_dir or latest_run_dir()
    checkpoint_path = args.checkpoint or find_checkpoint(run_dir)
    print(f"Run: {run_dir}")
    print(f"Checkpoint: {checkpoint_path}")

    model, image_size, num_qubits, backend_used = build_model(args)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    dataset = ImageFolderDataset(args.data_dir, image_size, grayscale=not args.color)
    if len(dataset) != 1:
        print(f"УВАГА: у '{args.data_dir}' {len(dataset)} файл(ів), а не 1 -- беру перший.")

    img, _ = dataset[0]
    batch = img.unsqueeze(0)
    target = batch if args.color else batch.squeeze(1)
    with torch.no_grad():
        recon = model(batch)
    mae = torch.nn.functional.l1_loss(recon, target).item()

    size_px = image_size * args.upscale
    mode = "RGB" if args.color else "L"

    def to_pil(t):
        arr = tensor_to_uint8(t)
        if args.color:
            arr = np.transpose(arr, (1, 2, 0))
        return Image.fromarray(arr, mode=mode).resize((size_px, size_px), Image.NEAREST)

    true_orig = Image.open(dataset.paths[0]).convert(mode).resize((size_px, size_px), Image.LANCZOS)
    target_im = to_pil(target[0])
    recon_im = to_pil(recon[0])

    gap = 8
    label_h = 60
    canvas_mode = "RGB"
    combined = Image.new(canvas_mode, (size_px * 3 + gap * 2, size_px + label_h), color=(255, 255, 255))
    combined.paste(true_orig.convert(canvas_mode), (0, label_h))
    combined.paste(target_im.convert(canvas_mode), (size_px + gap, label_h))
    combined.paste(recon_im.convert(canvas_mode), (size_px * 2 + gap * 2, label_h))

    draw = ImageDraw.Draw(combined)

    font = None
    for font_path in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ):
        try:
            font = ImageFont.truetype(font_path, 26)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
        print("УВАГА: не знайшов системний шрифт з кирилицею -- підписи можуть бути нечитабельними.")

    labels = [
        "1. ОРИГІНАЛ (повний розмір)",
        f"2. ВХІД МОДЕЛІ ({image_size}x{image_size}, стиснуто)",
        f"3. ВИХІД МОДЕЛІ (реконструкція, MAE={mae:.4f})",
    ]
    for i, label in enumerate(labels):
        x = i * (size_px + gap) + 6
        draw.text((x, 14), label, fill=(200, 0, 0), font=font)
        draw.rectangle(
            [i * (size_px + gap), 0, i * (size_px + gap) + size_px, combined.height - 1],
            outline=(0, 0, 0), width=1,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    img_path = f"{args.out}.png"
    combined.save(img_path)

    metrics_path = os.path.join(run_dir, "metrics.json")
    history = {}
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    best_val = min(history.get("val_loss", [mae])) if history.get("val_loss") else mae
    epochs_ran = len(history.get("val_loss", []))

    report_lines = [
        f"# Фінальний звіт -- квантовий кодер-декодер (PQC)",
        f"",
        f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Знімок: {os.path.basename(dataset.paths[0])}",
        f"Джерело даних: {args.data_dir}",
        f"",
        f"## Параметри моделі",
        f"- num_qubits: {num_qubits}",
        f"- num_layers: {args.num_layers}",
        f"- hidden_dim (класичний енкодер/декодер): {args.hidden_dim}",
        f"- bottleneck: {args.bottleneck}",
        f"- image_size (роздільність патча): {image_size}x{image_size}",
        f"- color: {args.color}",
        f"- backend: {backend_used}",
        f"",
        f"## Результати навчання",
        f"- Епох пройдено (за metrics.json цього run): {epochs_ran}",
        f"- Найкращий val_loss за весь прогін: {best_val:.4f}",
        f"- MAE на цьому знімку (checkpoint {os.path.basename(checkpoint_path)}): {mae:.4f}",
        f"",
        f"## Зображення",
        f"Файл: {img_path}",
        f"Зліва направо: справжній оригінал (повна роздільність) -> те, що бачила модель "
        f"({image_size}x{image_size}) -> реконструкція моделі.",
    ]
    report_path = f"{args.out}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nMAE={mae:.4f} | найкращий val_loss за прогін={best_val:.4f} | епох пройдено={epochs_ran}")
    print(f"Картинка: {img_path}")
    print(f"Звіт: {report_path}")

    try:
        os.startfile(os.path.abspath(img_path))
    except Exception:
        pass


if __name__ == "__main__":
    main()
