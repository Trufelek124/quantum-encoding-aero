"""
Оптимізація гіперпараметрів (Optuna) для гібридного автоенкодера.

Що шукаємо:
  - кількість ансатців (num_layers кола),
  - кількість І розміри прихованих шарів енкодера,
  - кількість І розміри прихованих шарів декодера,
  - розмір bottleneck.

Кожен trial будує модель через build_model() з train_autoencoder7.py, тож
конфігурація trial-у -- це рівно те, що збудує і головний скрипт тренування
(зокрема num_params кола завжди узгоджений через circuit.total_params).

Study зберігається у SQLite (--output-dir), тому запуск можна перервати і
продовжити тією самою командою -- завершені trial-и не повторюються.
Слабкі trial-и обрізаються достроково (MedianPruner) за val_loss по епохах.

Приклад:
  python tune_hyperparams.py --data-dir data2 --num-qubits 6 \
      --n-trials 40 --epochs-per-trial 15
"""

import argparse
import datetime
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader, Subset

from circuit import total_params
from dataset_loader import get_mnist_dataloaders, get_image_dataloaders
from train_autoencoder7 import build_model

torch.set_default_dtype(torch.float64)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-qubits", type=int, default=6,
                    help="Фіксована к-сть кубітів (парна). image_size = 2**(num_qubits/2). "
                         "Кубіти НЕ оптимізуються, бо змінюють роздільність, і val_loss "
                         "різних роздільностей непорівнянний.")
    p.add_argument("--data-dir", type=str, default="data/aero",
                    help="Папка з .jpg/.jpeg або 'mnist' (як у train_autoencoder7.py).")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--epochs-per-trial", type=int, default=15,
                    help="Коротке тренування на trial. MedianPruner обрізає слабкі раніше.")
    p.add_argument("--timeout", type=int, default=None,
                    help="Загальний ліміт оптимізації в секундах (напр. 8 годин = 28800).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--diff-method", type=str, default=None,
                    help="Перевизначити метод диференціювання кола (див. train_autoencoder7.py). "
                         "Нові версії PennyLane не приймають adjoint із qml.probs -- тоді "
                         "передайте parameter-shift.")
    p.add_argument("--train-fraction", type=float, default=0.05,
                    help="Лише для mnist: частка train-даних (як у train_autoencoder7.py).")
    p.add_argument("--val-fraction", type=float, default=0.1,
                    help="Для mnist: частка val-даних; для власного датасету: частка на валідацію.")
    p.add_argument("--limit-train", type=int, default=None,
                    help="Обмежити train до N зображень (пришвидшує кожен trial).")

    # Межі простору пошуку
    p.add_argument("--min-ansatz", type=int, default=2)
    p.add_argument("--max-ansatz", type=int, default=8,
                    help="Діапазон к-сті ансатців (повторів блоку U3+CPhase).")
    p.add_argument("--max-encoder-layers", type=int, default=3,
                    help="Максимум прихованих шарів енкодера (мінімум завжди 1).")
    p.add_argument("--max-decoder-layers", type=int, default=4,
                    help="Максимум прихованих шарів декодера (мінімум завжди 1).")
    p.add_argument("--min-dim", type=int, default=16)
    p.add_argument("--max-dim", type=int, default=512,
                    help="Діапазон розміру КОЖНОГО прихованого шару (лог-шкала).")
    p.add_argument("--min-bottleneck", type=int, default=8)
    p.add_argument("--max-bottleneck", type=int, default=256)

    p.add_argument("--study-name", type=str, default=None,
                    help="Назва study. За замовчуванням tune_qN_<data>. Та сама назва + "
                         "той самий --output-dir = продовження попереднього запуску.")
    p.add_argument("--output-dir", type=str, default=os.path.join("runs", "optuna"))
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_loaders(args, image_size):
    """Дані будуються ОДИН раз (image_size фіксований кубітами) і спільні для всіх trial-ів."""
    if args.data_dir.strip().lower() == "mnist":
        full_train, full_val = get_mnist_dataloaders(image_size=image_size, batch_size=args.batch_size)
        train_ds, val_ds = full_train.dataset, full_val.dataset
        n_train = max(1, int(len(train_ds) * args.train_fraction))
        n_val = max(1, int(len(val_ds) * args.val_fraction))
        g = torch.Generator().manual_seed(args.seed)
        train_idx = torch.randperm(len(train_ds), generator=g)[:n_train]
        val_idx = torch.randperm(len(val_ds), generator=g)[:n_val]
        train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=args.batch_size, shuffle=False)
    else:
        train_loader, val_loader = get_image_dataloaders(
            args.data_dir, image_size=image_size, batch_size=args.batch_size,
            val_fraction=args.val_fraction, grayscale=True, seed=args.seed)

    if args.limit_train and args.limit_train < len(train_loader.dataset):
        ds = train_loader.dataset
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(ds), generator=g)[:args.limit_train]
        train_loader = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=True)
    return train_loader, val_loader


def suggest_config(trial, args):
    num_ansatz = trial.suggest_int("num_ansatz_layers", args.min_ansatz, args.max_ansatz)

    n_enc = trial.suggest_int("n_encoder_layers", 1, args.max_encoder_layers)
    encoder_dims = [trial.suggest_int(f"enc_dim_{i}", args.min_dim, args.max_dim, log=True)
                    for i in range(n_enc)]
    n_dec = trial.suggest_int("n_decoder_layers", 1, args.max_decoder_layers)
    decoder_dims = [trial.suggest_int(f"dec_dim_{i}", args.min_dim, args.max_dim, log=True)
                    for i in range(n_dec)]
    bottleneck = trial.suggest_int("bottleneck", args.min_bottleneck, args.max_bottleneck, log=True)
    return num_ansatz, encoder_dims, decoder_dims, bottleneck


def trial_namespace(args, num_ansatz, encoder_dims, decoder_dims, bottleneck):
    """Namespace у форматі train_autoencoder7.build_model -- прямий (не patch) grayscale-режим."""
    return argparse.Namespace(
        num_qubits=args.num_qubits,
        num_layers=num_ansatz,
        hidden_dim=48,  # не використовується, коли encoder/decoder_dims задані явно
        bottleneck=bottleneck,
        encoder_dims=",".join(map(str, encoder_dims)),
        decoder_dims=",".join(map(str, decoder_dims)),
        patch_mode=False,
        patch_size=8,
        image_size=None,
        color=False,
        smooth_conv=False,
        use_wavelet=False,
        train_noise_level=0.0,
        diff_method=args.diff_method,
    )


def objective(trial, args, train_loader, val_loader, device, log_print):
    num_ansatz, encoder_dims, decoder_dims, bottleneck = suggest_config(trial, args)
    n_params = total_params(args.num_qubits, num_ansatz)
    trial.set_user_attr("encoder_dims", encoder_dims)
    trial.set_user_attr("decoder_dims", decoder_dims)
    trial.set_user_attr("circuit_params", n_params)

    # Однакові seed-и для кожного trial -- порівнюємо архітектури, а не випадкові ініціалізації.
    set_seed(args.seed)
    margs = trial_namespace(args, num_ansatz, encoder_dims, decoder_dims, bottleneck)
    model, _, _, backend_used = build_model(margs)
    model.to(device)

    log_print(f"[Trial {trial.number:03d}] ансатців={num_ansatz} (params={n_params}) | "
              f"енкодер={encoder_dims} | bottleneck={bottleneck} | декодер={decoder_dims} | "
              f"backend={backend_used}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(args.epochs_per_trial):
        model.train()
        for batch_images, _ in train_loader:
            batch_images = batch_images.to(device)
            target = batch_images.squeeze(1)
            optimizer.zero_grad()
            loss = criterion(model(batch_images), target)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_images, _ in val_loader:
                batch_images = batch_images.to(device)
                target = batch_images.squeeze(1)
                val_loss += criterion(model(batch_images), target).item()
        val_loss /= len(val_loader)
        best_val = min(best_val, val_loss)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            log_print(f"[Trial {trial.number:03d}]   обрізано на епосі {epoch+1} "
                      f"(val={val_loss:.4f}, {time.time()-t0:.0f}s)")
            raise optuna.TrialPruned()

    log_print(f"[Trial {trial.number:03d}]   завершено: best_val={best_val:.4f} "
              f"({time.time()-t0:.0f}s)")
    return best_val


def best_train_command(args, best):
    ua = best.user_attrs
    return (f"python train_autoencoder7.py --data-dir {args.data_dir} "
            f"--num-qubits {args.num_qubits} "
            f"--num-layers {best.params['num_ansatz_layers']} "
            f"--encoder-dims \"{','.join(map(str, ua['encoder_dims']))}\" "
            f"--decoder-dims \"{','.join(map(str, ua['decoder_dims']))}\" "
            f"--bottleneck {best.params['bottleneck']} --epochs 100")


def main():
    args = parse_args()
    if args.num_qubits % 2 != 0:
        raise ValueError("--num-qubits має бути парним.")
    image_size = int(round(2 ** (args.num_qubits / 2)))

    data_tag = "mnist" if args.data_dir.strip().lower() == "mnist" else os.path.basename(os.path.normpath(args.data_dir))
    study_name = args.study_name or f"tune_q{args.num_qubits}_{data_tag}"
    os.makedirs(args.output_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(args.output_dir, study_name + '.db')}"
    log_file = os.path.join(args.output_dir, study_name + ".log")

    def log_print(message):
        print(message)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {message}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_loaders(args, image_size)

    log_print(f"Study '{study_name}' | storage={storage}")
    log_print(f"device={device} | num_qubits={args.num_qubits} | image_size={image_size} | "
              f"train={len(train_loader.dataset)} | val={len(val_loader.dataset)}")
    log_print(f"Простір пошуку: ансатці {args.min_ansatz}..{args.max_ansatz} | "
              f"енкодер 1..{args.max_encoder_layers} шарів | декодер 1..{args.max_decoder_layers} шарів | "
              f"розміри {args.min_dim}..{args.max_dim} | bottleneck {args.min_bottleneck}..{args.max_bottleneck}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2),
    )
    done = len([t for t in study.trials if t.state.is_finished()])
    if done:
        log_print(f"Продовжую study: вже завершено {done} trial-ів.")

    study.optimize(
        lambda trial: objective(trial, args, train_loader, val_loader, device, log_print),
        n_trials=args.n_trials,
        timeout=args.timeout,
        # RuntimeError (напр. OOM у конкретного великого trial-у) не вбиває весь
        # запуск -- trial позначається FAIL, оптимізація йде далі.
        catch=(RuntimeError,),
    )

    best = study.best_trial
    log_print("\n=== НАЙКРАЩИЙ TRIAL ===")
    log_print(f"Trial {best.number}: val_loss={best.value:.4f}")
    log_print(f"  ансатців: {best.params['num_ansatz_layers']} "
              f"(параметрів кола: {best.user_attrs['circuit_params']})")
    log_print(f"  енкодер: {best.user_attrs['encoder_dims']} | "
              f"bottleneck: {best.params['bottleneck']} | "
              f"декодер: {best.user_attrs['decoder_dims']}")
    command = best_train_command(args, best)
    log_print(f"Команда для повного тренування:\n  {command}")

    best_config = {
        "study_name": study_name,
        "best_trial_number": best.number,
        "best_val_loss": best.value,
        "num_qubits": args.num_qubits,
        "num_ansatz_layers": best.params["num_ansatz_layers"],
        "circuit_params": best.user_attrs["circuit_params"],
        "encoder_dims": best.user_attrs["encoder_dims"],
        "bottleneck": best.params["bottleneck"],
        "decoder_dims": best.user_attrs["decoder_dims"],
        "epochs_per_trial": args.epochs_per_trial,
        "train_command": command,
    }
    best_path = os.path.join(args.output_dir, study_name + "_best.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=4, ensure_ascii=False)
    log_print(f"Найкращу конфігурацію збережено у {best_path}")


if __name__ == "__main__":
    main()
