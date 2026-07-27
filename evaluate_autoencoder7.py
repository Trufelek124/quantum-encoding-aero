import argparse
import glob
import os
import re

import numpy as np
import torch
from PIL import Image

from train_autoencoder7 import build_model
from dataset_loader import ImageFolderDataset


def find_latest_checkpoint(runs_root="runs"):
    run_dirs = sorted(glob.glob(os.path.join(runs_root, "run_*")))
    if not run_dirs:
        raise FileNotFoundError(f"У '{runs_root}' немає жодної папки run_*.")
    latest_run = run_dirs[-1]  # назви містять timestamp -> сортування = хронологія

    final_path = os.path.join(latest_run, "model_FINAL.pth")
    if os.path.isfile(final_path):
        return final_path

    epoch_files = glob.glob(os.path.join(latest_run, "model_epoch_*.pth"))
    if not epoch_files:
        raise FileNotFoundError(
            f"У '{latest_run}' немає ні model_FINAL.pth, ні model_epoch_*.pth. "
            f"Схоже, тренування не встигло зберегти чекпоінт (запустіть `dir {latest_run}` щоб перевірити)."
        )

    def epoch_num(path):
        m = re.search(r"model_epoch_(\d+)\.pth", path)
        return int(m.group(1)) if m else -1

    epoch_files.sort(key=epoch_num)
    return epoch_files[-1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Шлях до .pth. Якщо не вказано -- автоматично береться "
                         "найновіший runs/run_*/model_FINAL.pth (або останній epoch, якщо FINAL немає).")
    p.add_argument("--num-qubits", type=int, default=6)
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--hidden-dim", type=int, default=48)
    p.add_argument("--bottleneck", type=int, default=23)
    p.add_argument("--patch-mode", action="store_true")
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--data-dir", type=str, default="data/aero")
    p.add_argument("--out-dir", type=str, default="eval_out")
    p.add_argument("--only-file", type=str, default=None,
                    help="Показати лише зображення, ім'я файлу якого містить цей підрядок "
                         "(напр. --only-file DSC07504_tile_0063), замість усього датасету.")
    p.add_argument("--upscale", type=int, default=32,
                    help="У скільки разів збільшити картинку для перегляду (8x8 нема сенсу дивитись у натуральну величину).")
    p.add_argument("--color", action="store_true",
                    help="Кольоровий (RGB) режим -- має збігатись з тим, як тренували модель.")
    p.add_argument("--smooth-conv", action="store_true",
                    help="Має збігатись з тим, чи використовували --smooth-conv при тренуванні.")
    p.add_argument("--smooth-channels", type=int, default=16)
    p.add_argument("--smooth-layers", type=int, default=2)
    p.add_argument("--use-wavelet", action="store_true")
    return p.parse_args()


def tensor_to_uint8(img):
    arr = img.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def to_pil(img_tensor, color, size_px):
    """img_tensor: (H,W) для grayscale або (3,H,W) для кольору."""
    if color:
        arr = tensor_to_uint8(img_tensor)          # (3, H, W)
        arr = np.transpose(arr, (1, 2, 0))          # (H, W, 3)
        im = Image.fromarray(arr, mode="RGB")
    else:
        arr = tensor_to_uint8(img_tensor)
        im = Image.fromarray(arr, mode="L")
    return im.resize((size_px, size_px), Image.NEAREST)


def main():
    args = parse_args()

    checkpoint_path = args.checkpoint or find_latest_checkpoint()
    if args.checkpoint is None:
        print(f"--checkpoint не вказано, беру найновіший автоматично: {checkpoint_path}")

    # Архітектура має збігатись з тією, на якій тренували
    # (ті самі --num-qubits/--num-layers/--patch-mode/--color, що й у train_autoencoder.py)
    model, image_size, num_qubits, backend_used = build_model(args)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Модель завантажено: {checkpoint_path}")
    print(f"num_qubits={num_qubits} | image_size={image_size} | backend={backend_used} | color={args.color}")

    dataset = ImageFolderDataset(args.data_dir, image_size, grayscale=not args.color)
    os.makedirs(args.out_dir, exist_ok=True)

    indices_to_show = range(len(dataset))
    if args.only_file:
        indices_to_show = [i for i, p in enumerate(dataset.paths) if args.only_file in os.path.basename(p)]
        if not indices_to_show:
            raise ValueError(f"Жоден файл не містить '{args.only_file}' у назві.")
        print(f"Показую лише {len(indices_to_show)} файл(и), що відповідають '{args.only_file}'")

    total_mae = 0.0
    with torch.no_grad():
        for idx in indices_to_show:
            img, _ = dataset[idx]
            batch = img.unsqueeze(0)                              # (1, C, H, W)
            target = batch if args.color else batch.squeeze(1)    # (1,3,H,W) або (1,H,W)
            recon = model(batch)

            mae = torch.nn.functional.l1_loss(recon, target).item()
            total_mae += mae

            size_px = image_size * args.upscale
            orig_im = to_pil(target[0], args.color, size_px)
            recon_im = to_pil(recon[0], args.color, size_px)

            # Справжній оригінал -- як файл виглядав ДО стиснення до image_size,
            # щоб було видно, скільки інформації втрачається на самому ресайзі,
            # ще до того, як зображення взагалі потрапляє в модель.
            mode = "RGB" if args.color else "L"
            true_orig = Image.open(dataset.paths[idx]).convert(mode).resize((size_px, size_px), Image.LANCZOS)

            gap = 8
            combined = Image.new(mode, (size_px * 3 + gap * 2, size_px), color=255)
            combined.paste(true_orig, (0, 0))
            combined.paste(orig_im, (size_px + gap, 0))
            combined.paste(recon_im, (size_px * 2 + gap * 2, 0))

            name = os.path.splitext(os.path.basename(dataset.paths[idx]))[0]
            out_path = os.path.join(args.out_dir, f"{name}_orig_vs_recon.png")
            combined.save(out_path)
            print(f"[{idx}] {os.path.basename(dataset.paths[idx])} | MAE={mae:.4f} -> {out_path}")

    n_shown = len(list(indices_to_show)) if not isinstance(indices_to_show, range) else len(indices_to_show)
    print(f"\nСередній MAE по {n_shown} зображеннях: {total_mae/n_shown:.4f}")
    print("У кожному файлі зліва направо: СПРАВЖНІЙ оригінал (повна роздільність) -> "
          f"те, що бачила модель ({image_size}x{image_size}, ціль для реконструкції) -> "
          "реконструкція моделі.")


if __name__ == "__main__":
    main()
