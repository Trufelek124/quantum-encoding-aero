import math
import torch
import torch.nn as nn


class PatchHybridAutoencoder(nn.Module):
    """
    Альтернатива прямому нарощуванню num_qubits.

    Замість того, щоб num_qubits ріс разом із роздільністю (і симуляція
    квантового стану росла як 2**n), ділимо зображення на патчі
    фіксованого розміру (напр. 8x8 -> 6 кубітів, як у baseline) і
    прокидаємо КОЖЕН патч через один і той самий (спільний за вагами)
    невеликий квантовий декодер. Роздільність зображення росте лінійно
    через кількість патчів, а не експоненційно через кубіти.

    Компроміс: кількість forward-викликів quantum_layer росте як
    (image_size / patch_size)**2 * batch_size. Це дешевше за експоненційне
    зростання, але для дуже великих зображень і без ефективного батчингу
    самого quantum_layer це все одно може бути повільно -- дивіться
    README_scaling.md, розділ "Батчинг quantum_layer".
    """

    def __init__(self, classical_encoder, quantum_layer, num_qubits, patch_size, image_size):
        super().__init__()
        if num_qubits % 2 != 0:
            raise ValueError("num_qubits має бути парним (2**num_qubits має бути точним квадратом).")
        if image_size % patch_size != 0:
            raise ValueError("image_size має ділитися націло на patch_size.")

        expected_patch_pixels = patch_size * patch_size
        if expected_patch_pixels != 2 ** num_qubits:
            raise ValueError(
                f"patch_size**2 ({expected_patch_pixels}) != 2**num_qubits "
                f"({2 ** num_qubits}); підберіть відповідні num_qubits/patch_size "
                f"(напр. patch_size=8 -> num_qubits=6)."
            )

        self.classical_encoder = classical_encoder
        self.quantum_layer = quantum_layer
        self.num_qubits = num_qubits
        self.num_pixels_per_patch = 2 ** num_qubits
        self.patch_size = patch_size
        self.image_size = image_size
        self.patches_per_side = image_size // patch_size
        self.num_patches = self.patches_per_side ** 2

        # Позиційна ознака патча (row, col) в [-1, 1] -- щоб спільний
        # декодер знав, який саме патч він зараз генерує.
        self.register_buffer("patch_positions", self._build_positions())

    def _build_positions(self):
        n = self.patches_per_side
        coords = []
        for r in range(n):
            for c in range(n):
                rr = (r / max(n - 1, 1)) * 2 - 1
                cc = (c / max(n - 1, 1)) * 2 - 1
                coords.append([rr, cc])
        return torch.tensor(coords, dtype=torch.get_default_dtype())

    def forward(self, x):
        # x: (B, 1, H, W)
        batch_size = x.shape[0]
        device = x.device
        ps = self.patch_size
        n = self.patches_per_side

        patches = x.unfold(2, ps, ps).unfold(3, ps, ps)  # (B, 1, n, n, ps, ps)
        patches = patches.contiguous().view(batch_size, self.num_patches, 1, ps, ps)
        flat_patches = patches.reshape(batch_size * self.num_patches, 1, ps, ps)

        pos = self.patch_positions.to(device).unsqueeze(0).expand(batch_size, -1, -1)
        pos = pos.reshape(batch_size * self.num_patches, 2)

        params = self.classical_encoder(flat_patches)
        # Проста ін'єкція позиції патча в перші два параметри ансатцу.
        # Це робочий baseline; за потреби замініть на конкатенацію ознак
        # усередині самого classical_encoder (надійніше для навчання).
        params = params.clone()
        params[:, :2] = params[:, :2] + pos * math.pi

        quantum_probs = self.quantum_layer(params)

        num_pixels = self.num_pixels_per_patch
        output_pixels = quantum_probs[:, num_pixels:]
        output_patches = (output_pixels * num_pixels).view(
            batch_size, n, n, ps, ps
        )

        output_image = output_patches.permute(0, 1, 3, 2, 4).reshape(
            batch_size, self.image_size, self.image_size
        )
        return output_image
