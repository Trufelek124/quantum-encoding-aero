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

from circuit import create_qnode
from autoencoder7 import Autoencoder
from hybrid_autoencoder7 import HybridAutoencoder
from dataset_loader import get_image_dataloaders

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


def main(num_layers_override=None, noise_level_override=None, data_dir="data2", epochs_override=None):
    if num_layers_override is not None:
        NUM_LAYERS = num_layers_override
    else:
        NUM_LAYERS = 5

    NUM_QUBITS = 6
    NUM_PARAMETERS = (NUM_QUBITS * 4) * NUM_LAYERS

    IMAGE_SIZE = int(2 ** (NUM_QUBITS / 2))
    BATCH_SIZE = 8
    EPOCHS = epochs_override if epochs_override is not None else 100
    LEARNING_RATE = 0.001
    ACTIVATION = "ReLU"
    HIDDEN = 48
    BOTTLENECK = 23
    NOISE_LEVEL = noise_level_override if noise_level_override is not None else 0.0

    OPTIMIZER_NAME = "Adam"
    LOSS_FUNCTION_NAME = "L1Loss"
    # За наявності шуму потрібен density-matrix симулятор (lightning.qubit
    # рахує лише чисті стани і не підтримує канали шуму).
    BACKEND_NAME = "default.mixed" if NOISE_LEVEL > 0 else "lightning.qubit"
    GRADIENT_NAME = "Backprop_PL"
    SCHEDULER_NAME = "ReduceLROnPlateau"

    TRAIN_FRACTION = 1.0
    VAL_FRACTION = 1.0

    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    LOG_FREQ = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("runs", f"run_noise{NOISE_LEVEL}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    log_file_path = os.path.join(run_dir, "training_log.txt")
    metrics_file_path = os.path.join(run_dir, "metrics.json")

    def log_print(message):
        print(message)
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    hyperparameters = {
        "NUM_QUBITS": NUM_QUBITS,
        "NUM_LAYERS": NUM_LAYERS,
        "NUM_PARAMETERS": NUM_PARAMETERS,
        "IMAGE_SIZE": IMAGE_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "OPTIMIZER": OPTIMIZER_NAME,
        "LOSS_FUNCTION": LOSS_FUNCTION_NAME,
        "BACKEND": BACKEND_NAME,
        "GRADIENT": GRADIENT_NAME,
        "HIDDEN": HIDDEN,
        "BOTTLENECK": BOTTLENECK,
        "ACTIVATION": ACTIVATION,
        "NOISE_LEVEL": NOISE_LEVEL,
        "DATA_DIR": data_dir,
    }
    with open(os.path.join(run_dir, "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(hyperparameters, f, indent=4)

    log_print(f"Нова сесія. NOISE_LEVEL={NOISE_LEVEL} | NUM_LAYERS={NUM_LAYERS} | Файли в: {run_dir}")

    dev = qml.device(BACKEND_NAME, wires=NUM_QUBITS + 1)
    qnode = create_qnode(NUM_QUBITS, NUM_LAYERS, dev, NOISE_LEVEL)
    quantum_layer = PennyLaneQuantumLayer(qnode)

    classical_ae = Autoencoder(image_size=IMAGE_SIZE, num_params=NUM_PARAMETERS,
                                hidden_dim=HIDDEN, bottleneck_size=BOTTLENECK, activation=ACTIVATION)
    model_hybrydowy = HybridAutoencoder(classical_ae, quantum_layer, num_qubits=NUM_QUBITS)
    model_hybrydowy.to(device)

    with open(os.path.join(run_dir, "architecture.txt"), "w", encoding="utf-8") as f:
        f.write(str(model_hybrydowy))

    full_train_loader, full_val_loader = get_image_dataloaders(
        data_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        val_fraction=0.5 if len(os.listdir(data_dir)) > 1 else 0.0, grayscale=True)

    train_loader = full_train_loader
    val_loader = full_val_loader

    log_print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    optimizer = optim.Adam(model_hybrydowy.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)
    criterion = nn.L1Loss()

    history = {"train_loss": [], "val_loss": [], "learning_rate": []}
    total_start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model_hybrydowy.train()
        train_epoch_loss = 0.0

        for batch_idx, (batch_images, _) in enumerate(train_loader):
            batch_images = batch_images.to(device)
            target_images = batch_images.squeeze(1)

            optimizer.zero_grad()
            reconstructed_batch = model_hybrydowy(batch_images)
            loss = criterion(reconstructed_batch, target_images)
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()

        avg_train_loss = train_epoch_loss / len(train_loader)

        model_hybrydowy.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for batch_images, _ in val_loader:
                batch_images = batch_images.to(device)
                target_images = batch_images.squeeze(1)
                reconstructed_batch = model_hybrydowy(batch_images)
                val_epoch_loss += criterion(reconstructed_batch, target_images).item()
        avg_val_loss = val_epoch_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["learning_rate"].append(current_lr)
        with open(metrics_file_path, "w") as f:
            json.dump(history, f, indent=4)

        epoch_duration = time.time() - epoch_start_time
        if (epoch + 1) % LOG_FREQ == 0 or epoch == EPOCHS - 1:
            log_print(f"Epoka {epoch+1}/{EPOCHS} | Train: {avg_train_loss:.4f} | "
                       f"Val: {avg_val_loss:.4f} | LR: {current_lr:.6f} | Czas: {epoch_duration:.1f}s")

    total_duration = time.time() - total_start_time
    log_print(f"\nTrening zakończony! Całkowity czas: {total_duration/60:.2f} min.")

    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model_hybrydowy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'NOISE_LEVEL': NOISE_LEVEL,
        'NUM_LAYERS': NUM_LAYERS,
    }, os.path.join(run_dir, "model_FINAL.pth"))

    return run_dir, avg_val_loss


if __name__ == "__main__":
    import sys

    # Легкий шум -- єдиний рівень, що дав РЕАЛЬНУ (не "порожню") реконструкцію
    # в попередньому тесті. Довше тренування, щоб довести до якісного результату.
    noise_levels_to_test = [0.01]
    epochs = 200

    results = []
    print("=== БАЗОВИЙ РІВЕНЬ (без шуму) ===")
    run_dir, val_loss = main(noise_level_override=0.0, epochs_override=epochs)
    results.append(("noise=0.0 (базовий)", val_loss, run_dir))

    for i, nl in enumerate(noise_levels_to_test, start=1):
        print(f"\n=== РІВЕНЬ ШУМУ {i}: NOISE_LEVEL={nl} ===")
        run_dir, val_loss = main(noise_level_override=nl, epochs_override=epochs)
        results.append((f"noise={nl} (рівень {i})", val_loss, run_dir))

    print("\n" + "=" * 60)
    print("ПІДСУМОК:")
    for label, val_loss, run_dir in results:
        print(f"  {label}: val_loss={val_loss:.4f}  ({run_dir})")
    print("=" * 60)
