import argparse
import os

import numpy as np
import torch
import pennylane as qml
from PIL import Image, ImageDraw, ImageFont

from circuit import create_qnode
from train_autoencoder7 import build_model, PennyLaneQuantumLayer
from dataset_loader import ImageFolderDataset

torch.set_default_dtype(torch.float64)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Чекпоінт моделі, натренованої БЕЗ шуму (--num-qubits/--num-layers мають "
                         "збігатись з тим прогоном).")
    p.add_argument("--data-dir", type=str, default="data2")
    p.add_argument("--num-qubits", type=int, default=10)
    p.add_argument("--num-layers", type=int, default=15)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--bottleneck", type=int, default=256)
    p.add_argument("--patch-mode", action="store_true")
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--color", action="store_true")
    p.add_argument("--smooth-conv", action="store_true")
    p.add_argument("--smooth-channels", type=int, default=32)
    p.add_argument("--smooth-layers", type=int, default=3)
    p.add_argument("--use-wavelet", action="store_true")
    p.add_argument("--noise-levels", type=float, nargs="+", default=[0.01, 0.05, 0.1],
                    help="Рівні деполяризуючого шуму на кожен гейт кожного шару. "
                         "0.01 = легкий, 0.05 = помірний, 0.1 = сильний.")
    p.add_argument("--out-dir", type=str, default="noise_test")
    p.add_argument("--upscale", type=int, default=32)
    return p.parse_args()


def tensor_to_uint8(img):
    arr = img.detach().cpu().numpy()
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


def load_font(size):
    for font_path in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main():
    args = parse_args()

    model, image_size, num_qubits, backend_used = build_model(args)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Модель завантажено: {args.checkpoint} (без шуму, backend={backend_used})")

    dataset = ImageFolderDataset(args.data_dir, image_size, grayscale=True)
    img, _ = dataset[0]
    batch = img.unsqueeze(0)
    target = batch.squeeze(1)

    os.makedirs(args.out_dir, exist_ok=True)
    total_wires = num_qubits + 1
    size_px = image_size * args.upscale

    def to_pil(t):
        arr = tensor_to_uint8(t)
        return Image.fromarray(arr, mode="L").resize((size_px, size_px), Image.NEAREST)

    true_orig = Image.open(dataset.paths[0]).convert("L").resize((size_px, size_px), Image.LANCZOS)

    with torch.no_grad():
        recon_clean = model(batch)
    mae_clean = torch.nn.functional.l1_loss(recon_clean, target).item()
    print(f"\nБЕЗ ШУМУ (базовий рівень): MAE={mae_clean:.4f}")

    # Панелі: 1) справжній оригінал, 2) вхід моделі (стиснуто), 3) вихід без шуму, 4+) вихід із шумом
    panels = [
        ("1. ОРИГІНАЛ\n(повний розмір)", true_orig, None),
        (f"2. ВХІД МОДЕЛІ\n({image_size}x{image_size}, стиснуто)", to_pil(target[0]), None),
        ("3. ВИХІД, без шуму", to_pil(recon_clean[0]), mae_clean),
    ]
    results = [("clean (0.0)", mae_clean)]

    for i, noise_level in enumerate(args.noise_levels, start=4):
        print(f"\nБудую зашумлене коло: noise_level={noise_level} ...")
        dev_noisy = qml.device("default.mixed", wires=total_wires)
        qnode_noisy = create_qnode(num_qubits, args.num_layers, dev_noisy, noise_level=noise_level)
        model.quantum_layer = PennyLaneQuantumLayer(qnode_noisy)

        with torch.no_grad():
            recon_noisy = model(batch)
        mae_noisy = torch.nn.functional.l1_loss(recon_noisy, target).item()
        print(f"noise_level={noise_level}: MAE={mae_noisy:.4f} "
              f"(зростання проти чистого: {mae_noisy - mae_clean:+.4f})")

        panels.append((f"{i}. ВИХІД, noise={noise_level}", to_pil(recon_noisy[0]), mae_noisy))
        results.append((f"noise={noise_level}", mae_noisy))

    gap = 10
    label_h = 90
    n = len(panels)
    combined = Image.new("RGB", (size_px * n + gap * (n - 1), size_px + label_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(combined)
    font = load_font(20)

    for i, (label, im, mae) in enumerate(panels):
        x = i * (size_px + gap)
        combined.paste(im.convert("RGB"), (x, label_h))
        text = label if mae is None else f"{label}\nMAE={mae:.4f}"
        draw.multiline_text((x + 4, 6), text, fill=(200, 0, 0), font=font, spacing=4)
        draw.rectangle([x, 0, x + size_px, combined.height - 1], outline=(0, 0, 0), width=1)

    out_path = os.path.join(args.out_dir, "noise_comparison.png")
    combined.save(out_path)

    print(f"\n{'=' * 50}\nПІДСУМОК (вплив шуму на якість реконструкції):")
    for label, mae in results:
        print(f"  {label}: MAE={mae:.4f}")
    print(f"{'=' * 50}")
    print(f"Порівняльна картинка (з ОРИГІНАЛОМ і ВХОДОМ для повної картини): {out_path}")

    try:
        os.startfile(os.path.abspath(out_path))
    except Exception:
        pass


if __name__ == "__main__":
    main()
