import argparse
import os
import time
import json
import random
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from torch.utils.data import DataLoader, Subset

from circuit import create_qnode, get_device
from autoencoder7 import Autoencoder
from hybrid_autoencoder7 import HybridAutoencoder, ColorHybridAutoencoder
from patch_autoencoder import PatchHybridAutoencoder
from dataset_loader import get_mnist_dataloaders, get_image_dataloaders

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.set_default_dtype(torch.float64)


class PennyLaneQuantumLayer(nn.Module):
    def __init__(self, qnode):
        super().__init__()
        self.qnode = qnode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.qnode(xi) for xi in x])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-qubits", type=int, default=6,
                    help="Має бути парним. image_size = 2**(num_qubits/2). "
                         "6->8x8 (baseline), 8->16x16, 10->32x32, 12->64x64.")
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--hidden-dim", type=int, default=48,
                    help="Розмір прихованих шарів класичного енкодера/декодера. "
                         "Більше = більша ємність, менше розмиття, повільніше навчання.")
    p.add_argument("--bottleneck", type=int, default=23,
                    help="Розмір 'вузького горла' між енкодером і декодером. "
                         "Більше = менше стиснення, менш розмита реконструкція.")
    p.add_argument("--epochs", type=int, default=None,
                    help="За замовчуванням 100, або 2 якщо --quick")
    p.add_argument("--quick", action="store_true",
                    help="Швидкий sanity-check: 2 епохи, дуже малий шматок даних, часте логування.")
    p.add_argument("--patch-mode", action="store_true",
                    help="Використати PatchHybridAutoencoder замість прямого масштабування "
                         "num_qubits. num_qubits лишається малим (patch_size**2 == 2**num_qubits), "
                         "а роздільність задається --image-size.")
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=None,
                    help="Лише для --patch-mode. Має ділитися на --patch-size.")
    p.add_argument("--data-dir", type=str, default="data/aero",
                    help="Папка з вашими .jpg/.jpeg зображеннями (рекурсивно). "
                         "За замовчуванням 'data/aero'. Щоб примусово взяти MNIST -- "
                         "передайте --data-dir mnist.")
    p.add_argument("--val-fraction", type=float, default=0.1,
                    help="Частка --data-dir, яка йде у валідацію (лише для власного датасету).")
    p.add_argument("--resume-from", type=str, default=None,
                    help="Шлях до .pth (напр. runs/run_.../model_epoch_10.pth), з якого продовжити "
                         "навчання замість початку з нуля -- відновлює ваги моделі, оптимізатора "
                         "і планувальника.")
    p.add_argument("--limit-train", type=int, default=None,
                    help="Обмежити train лише до N зображень -- для швидкої перевірки, чи взагалі "
                         "збігається реконструкція на цій роздільності, перш ніж чекати години на "
                         "всьому датасеті.")
    p.add_argument("--color", action="store_true",
                    help="Кольорове (RGB) зображення замість grayscale. Потребує num_qubits, "
                         "достатніх для 3*image_size**2 значень -- зазвичай на 2+ кубіти більше, "
                         "ніж grayscale-варіант тієї самої роздільності.")
    p.add_argument("--save-every", type=int, default=10,
                    help="Зберігати проміжний чекпоінт кожні N епох (плюс завжди останню), "
                         "щоб не заповнювати диск файлами на кожну епоху.")
    p.add_argument("--tv-weight", type=float, default=0.0,
                    help="Вага штрафу за різкі перепади між сусідніми пікселями (total variation). "
                         "0 = вимкнено (як раніше). Спробуйте 0.05-0.2, щоб прибрати "
                         "'шахматку'/шум і зробити реконструкцію гладкішою.")
    p.add_argument("--smooth-conv", action="store_true",
                    help="Додати навчальний Conv2d-шар після квантового виходу -- дає моделі "
                         "вбудований механізм бачити сусідні пікселі (на відміну від TV-loss, "
                         "який лише штрафує ззовні).")
    p.add_argument("--smooth-channels", type=int, default=16,
                    help="К-сть каналів у прихованих шарах smooth-conv (більше = більша ємність).")
    p.add_argument("--smooth-layers", type=int, default=2,
                    help="К-сть згорткових шарів у smooth-conv (більше = складніші просторові "
                         "патерни, повільніше).")
    p.add_argument("--use-wavelet", action="store_true",
                    help="Інтерпретувати вихід кванта як 2D Haar-вейвлет коефіцієнти (груба "
                         "апроксимація + деталі) і застосувати обернене вейвлет-перетворення "
                         "замість прямого трактування як пікселів. Можна поєднувати зі --smooth-conv.")
    p.add_argument("--train-noise-level", type=float, default=0.0,
                    help="Тренувати ОДРАЗУ на зашумленому колі (default.mixed, деполяризуючий шум "
                         "з цією ймовірністю на гейт) замість ідеального. Значно повільніше "
                         "(матриця густини замість вектора стану), але дає моделі шанс "
                         "навчитись компенсувати шум, а не просто ламатись від нього.")
    return p.parse_args()


def build_model(args):
    color = getattr(args, "color", False)
    if args.patch_mode:
        num_qubits = args.num_qubits
        image_size = args.image_size or (args.patch_size * 4)
        num_params = (num_qubits * 4) * args.num_layers

        train_noise = getattr(args, "train_noise_level", 0.0)
        if train_noise > 0:
            import pennylane as qml
            dev = qml.device("default.mixed", wires=num_qubits + 1)
            backend_used = f"default.mixed (noise={train_noise})"
        else:
            dev, backend_used = get_device(num_qubits + 1)
        # adjoint не підтримує структуру патч-кола на цій версії PennyLane
        # (QuantumFunctionError) -- parameter-shift повільніший, але сумісний з усім.
        diff_method = "backprop" if train_noise > 0 else "parameter-shift"
        qnode = create_qnode(num_qubits, args.num_layers, dev, noise_level=train_noise, diff_method=diff_method)
        quantum_layer = PennyLaneQuantumLayer(qnode)

        classical_ae = Autoencoder(image_size=args.patch_size, num_params=num_params,
                                    hidden_dim=args.hidden_dim, bottleneck_size=args.bottleneck, activation="ReLU",
                                    channels=3 if color else 1)
        model = PatchHybridAutoencoder(classical_ae, quantum_layer, num_qubits,
                                        args.patch_size, image_size)
        return model, image_size, num_qubits, backend_used
    else:
        num_qubits = args.num_qubits
        if num_qubits % 2 != 0:
            raise ValueError("--num-qubits має бути парним.")
        if color:
            # Потрібно 2**num_qubits >= 3 * image_size**2, тому розмір менший,
            # ніж для grayscale з тією самою кількістю кубітів.
            image_size = int((2 ** num_qubits / 3) ** 0.5)
            image_size = max(image_size, 1)
        else:
            image_size = int(round(2 ** (num_qubits / 2)))
        num_params = (num_qubits * 4) * args.num_layers

        train_noise = getattr(args, "train_noise_level", 0.0)
        if train_noise > 0:
            import pennylane as qml
            dev = qml.device("default.mixed", wires=num_qubits + 1)
            backend_used = f"default.mixed (noise={train_noise})"
        else:
            dev, backend_used = get_device(num_qubits + 1)
        diff_method = "backprop" if train_noise > 0 else "adjoint"
        qnode = create_qnode(num_qubits, args.num_layers, dev, noise_level=train_noise, diff_method=diff_method)
        quantum_layer = PennyLaneQuantumLayer(qnode)

        classical_ae = Autoencoder(image_size=image_size, num_params=num_params,
                                    hidden_dim=args.hidden_dim, bottleneck_size=args.bottleneck, activation="ReLU",
                                    channels=3 if color else 1)
        use_smoothing = getattr(args, "smooth_conv", False)
        use_wavelet = getattr(args, "use_wavelet", False)
        if color:
            model = ColorHybridAutoencoder(classical_ae, quantum_layer, num_qubits, image_size,
                                            use_smoothing=use_smoothing)
        else:
            model = HybridAutoencoder(classical_ae, quantum_layer, num_qubits=num_qubits,
                                       use_smoothing=use_smoothing,
                                       smooth_channels=getattr(args, "smooth_channels", 16),
                                       smooth_layers=getattr(args, "smooth_layers", 2),
                                       use_wavelet=use_wavelet)
        return model, image_size, num_qubits, backend_used


def main():
    args = parse_args()
    quick = args.quick

    EPOCHS = args.epochs if args.epochs is not None else (2 if quick else 100)
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    TRAIN_FRACTION = 0.005 if quick else 0.05
    VAL_FRACTION = 0.005 if quick else 0.05
    LOG_FREQ = 5 if quick else 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("runs", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    log_file_path = os.path.join(run_dir, "training_log.txt")
    metrics_file_path = os.path.join(run_dir, "metrics.json")

    def log_print(message):
        print(message)
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    model, image_size, num_qubits, backend_used = build_model(args)
    model.to(device)

    state_vector_size = 2 ** (num_qubits + 1)
    log_print(f"Режим: {'PATCH' if args.patch_mode else 'DIRECT'}")
    log_print(f"num_qubits={num_qubits} | image_size={image_size} | backend={backend_used}")
    log_print(f"Розмір статевектора симуляції: 2**{num_qubits+1} = {state_vector_size}")
    if args.patch_mode:
        log_print(f"patch_size={args.patch_size} | patches_per_side={image_size // args.patch_size}")

    with open(os.path.join(run_dir, "architecture.txt"), "w", encoding="utf-8") as f:
        f.write(str(model))

    use_mnist = args.data_dir.strip().lower() == "mnist"

    if not use_mnist:
        if not os.path.isdir(args.data_dir):
            os.makedirs(args.data_dir, exist_ok=True)
            log_print(f"Папку '{args.data_dir}' не знайдено -- створив її. "
                       f"Покладіть туди .jpg/.jpeg зображення і запустіть команду ще раз.")
            return

        color_label = "RGB" if args.color else "grayscale"
        log_print(f"Датасет: власні зображення з '{args.data_dir}' ({color_label}, resize -> {image_size}x{image_size})")
        train_loader, val_loader = get_image_dataloaders(
            args.data_dir, image_size=image_size, batch_size=BATCH_SIZE,
            val_fraction=args.val_fraction, grayscale=not args.color)
        num_train_samples = len(train_loader.dataset)
        num_val_samples = len(val_loader.dataset)

        limit_n = args.limit_train if args.limit_train else (
            None if not quick else None)
        if quick and not args.limit_train:
            limit_n = max(1, int(num_train_samples * max(TRAIN_FRACTION, 0.2)))

        if limit_n:
            train_dataset = train_loader.dataset
            limit_n = min(limit_n, len(train_dataset))
            indices = torch.randperm(len(train_dataset))[:limit_n]
            train_loader = DataLoader(Subset(train_dataset, indices), batch_size=BATCH_SIZE, shuffle=True)
            num_train_samples = limit_n

            # train_dataset тут -- Subset (від random_split), а не сам ImageFolderDataset,
            # тому шлях до файлу треба брати через .dataset.paths[.indices[i]]
            if hasattr(train_dataset, "paths"):
                selected_files = [os.path.basename(train_dataset.paths[i]) for i in indices.tolist()]
            elif hasattr(train_dataset, "dataset") and hasattr(train_dataset.dataset, "paths"):
                selected_files = [os.path.basename(train_dataset.dataset.paths[train_dataset.indices[i]])
                                   for i in indices.tolist()]
            else:
                selected_files = ["<не вдалось визначити назву файлу>"]

            log_print(f"--limit-train: обмежено train до {limit_n} зображень (швидка перевірка збіжності).")
            log_print(f"Обрані файли: {selected_files}")
    else:
        log_print("Датасет: MNIST (baseline, --data-dir mnist)")
        full_train_loader, full_val_loader = get_mnist_dataloaders(
            image_size=image_size, batch_size=BATCH_SIZE)

        train_dataset = full_train_loader.dataset
        num_train_samples = max(1, int(len(train_dataset) * TRAIN_FRACTION))
        indices = torch.randperm(len(train_dataset))[:num_train_samples]
        train_loader = DataLoader(Subset(train_dataset, indices), batch_size=BATCH_SIZE, shuffle=True)

        val_dataset = full_val_loader.dataset
        num_val_samples = max(1, int(len(val_dataset) * VAL_FRACTION))
        val_indices = torch.randperm(len(val_dataset))[:num_val_samples]
        val_loader = DataLoader(Subset(val_dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False)

    log_print(f"Train samples: {num_train_samples} | Val samples: {num_val_samples}")

    def total_variation_loss(img):
        """Штраф за різкі перепади між сусідніми пікселями (по рядках і колонках).
        img: (..., H, W) або (..., C, H, W)."""
        diff_h = (img[..., 1:, :] - img[..., :-1, :]).abs().mean()
        diff_w = (img[..., :, 1:] - img[..., :, :-1]).abs().mean()
        return diff_h + diff_w

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.L1Loss()

    start_epoch = 0
    if args.resume_from:
        log_print(f"Відновлюю з {args.resume_from} ...")
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        log_print(f"Відновлено з епохи {start_epoch}. Тренування продовжиться з епохи {start_epoch + 1} "
                   f"до {start_epoch + EPOCHS} (--epochs={EPOCHS} означає 'ще стільки ж епох', а не 'всього').")

    history = {"train_loss": [], "val_loss": [], "learning_rate": [], "epoch_time_sec": []}
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None

    for epoch_offset in range(EPOCHS):
        epoch = start_epoch + epoch_offset
        epoch_start_time = time.time()
        model.train()
        train_epoch_loss = 0.0

        for batch_idx, (batch_images, _) in enumerate(train_loader):
            batch_images = batch_images.to(device)
            target_images = batch_images if args.color else batch_images.squeeze(1)

            optimizer.zero_grad()
            reconstructed_batch = model(batch_images)
            recon_loss = criterion(reconstructed_batch, target_images)
            loss = recon_loss
            if args.tv_weight > 0:
                loss = loss + args.tv_weight * total_variation_loss(reconstructed_batch)
            loss.backward()
            optimizer.step()

            train_epoch_loss += recon_loss.item()  # логуємо саме L1, щоб цифри лишались порівнюваними з попередніми прогонами

            current_batch = batch_idx + 1
            if current_batch % LOG_FREQ == 0 or current_batch == len(train_loader):
                elapsed = time.time() - epoch_start_time
                avg_batch_time = elapsed / current_batch
                log_print(f"  [Train] Epoka {epoch+1:02d} | Batch {current_batch:04d}/{len(train_loader)} "
                           f"| Loss: {loss.item():.4f} | Śr. czas/batch: {avg_batch_time:.3f}s")

        avg_train_loss = train_epoch_loss / len(train_loader)

        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for batch_images, _ in val_loader:
                batch_images = batch_images.to(device)
                target_images = batch_images if args.color else batch_images.squeeze(1)
                reconstructed_batch = model(batch_images)
                val_epoch_loss += criterion(reconstructed_batch, target_images).item()
        avg_val_loss = val_epoch_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_duration = time.time() - epoch_start_time

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["learning_rate"].append(current_lr)
        history["epoch_time_sec"].append(epoch_duration)
        with open(metrics_file_path, "w") as f:
            json.dump(history, f, indent=4)

        log_print(f"=== Epoka {epoch+1}/{start_epoch + EPOCHS} | Train: {avg_train_loss:.4f} | "
                   f"Val: {avg_val_loss:.4f} | LR: {current_lr:.6f} | Czas: {epoch_duration:.1f}s ===\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        save_every = getattr(args, "save_every", 10)
        is_last = (epoch_offset == EPOCHS - 1)
        if is_last:
            # В кінці зберігаємо і останню, і найкращу (можуть бути різними епохами) --
            # це найважливіші два стани, решту проміжних НЕ пишемо на диск, щоб не
            # повторити переповнення диска через забагато великих чекпоінтів.
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(run_dir, f"model_epoch_{epoch+1}_LAST.pth"))

            if best_state is not None and best_epoch != epoch + 1:
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': best_state,
                    'val_loss': best_val_loss,
                }, os.path.join(run_dir, f"model_epoch_{best_epoch}_BEST.pth"))
            log_print(f"Найкраща епоха за val_loss: {best_epoch} (val_loss={best_val_loss:.4f})")
        elif (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(run_dir, f"model_epoch_{epoch+1}.pth"))

    final_state = best_state if best_state is not None else model.state_dict()
    torch.save({
        'epoch': best_epoch if best_state is not None else start_epoch + EPOCHS,
        'model_state_dict': final_state,
        'val_loss': best_val_loss,
    }, os.path.join(run_dir, "model_FINAL.pth"))

    log_print(f"model_FINAL.pth = найкраща епоха {best_epoch} (val_loss={best_val_loss:.4f}).")
    log_print("Готово. Дивіться epoch_time_sec у metrics.json -- це головна метрика "
               "для оцінки того, чи масштабування взагалі практичне на вашому залізі.")


if __name__ == "__main__":
    main()
