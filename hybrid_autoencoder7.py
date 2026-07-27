import torch
import torch.nn as nn


def haar_matrix(n):
    """Ортонормована 1D Haar wavelet матриця n x n (n має бути степенем двійки).
    Рядки впорядковані від найгрубішого (апроксимація) до найдетальнішого."""
    if n == 1:
        return torch.ones(1, 1)
    h_prev = haar_matrix(n // 2)
    approx = torch.kron(h_prev, torch.tensor([[1.0, 1.0]])) / (2 ** 0.5)
    detail = torch.kron(torch.eye(n // 2), torch.tensor([[1.0, -1.0]])) / (2 ** 0.5)
    return torch.cat([approx, detail], dim=0)


class HybridAutoencoder(nn.Module):
    """
    ВАЖЛИВО щодо масштабування роздільності:
    image_size**2 == 2**num_qubits, тобто num_qubits = 2*log2(image_size).
    Симуляція квантового стану росте як 2**(num_qubits+1) -- експоненційно.
    Для роздільностей вищих за ~32x32 дивіться patch_autoencoder.py.

    use_wavelet=True: замість того, щоб напряму вважати вихід кванта
    пікселями, інтерпретуємо його як 2D Haar-вейвлет коефіцієнти (груба
    апроксимація в одному куті + деталі різного масштабу в інших) і
    застосовуємо ОБЕРНЕНЕ вейвлет-перетворення (фіксована, математично
    строга операція), щоб зібрати фінальне зображення. Це дає моделі
    вбудовану багаторівневу просторову структуру -- природну для реальних
    фото (переважно гладкі ділянки + різкі краї), на відміну від "плаского"
    списку незалежних пікселів.

    use_smoothing=True додає невеликий навчальний згортковий шар (Conv2d)
    ПІСЛЯ розгортання зображення (з вейвлета чи напряму) -- ще один,
    навчальний механізм бачити сусідні пікселі. Можна поєднувати з
    use_wavelet.
    """

    def __init__(self, classical_encoder, quantum_layer, num_qubits=6, use_smoothing=False,
                 smooth_channels=16, smooth_layers=2, use_wavelet=False):
        super().__init__()
        if num_qubits % 2 != 0:
            raise ValueError(
                f"num_qubits={num_qubits} має бути парним, інакше 2**num_qubits "
                f"не є точним квадратом і зображення не складеться у size x size."
            )
        self.classical_encoder = classical_encoder
        self.quantum_layer = quantum_layer
        self.num_qubits = num_qubits
        self.num_pixels = 2 ** num_qubits
        self.use_smoothing = use_smoothing
        self.use_wavelet = use_wavelet

        size = int(round(self.num_pixels ** 0.5))
        if use_wavelet:
            H = haar_matrix(size)
            self.register_buffer("haar_H", H)

        if use_smoothing:
            layers = []
            in_ch = 1
            for i in range(smooth_layers):
                out_ch = smooth_channels if i < smooth_layers - 1 else 1
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
                if i < smooth_layers - 1:
                    layers.append(nn.ReLU())
                in_ch = out_ch
            self.smooth = nn.Sequential(*layers)
            # Перший шар ініціалізуємо близько до box-blur, щоб з перших
            # кроків навчання вихід уже був гладким, а не випадковим.
            with torch.no_grad():
                first_conv = self.smooth[0]
                first_conv.weight.fill_(1.0 / (9.0 * first_conv.out_channels))
                first_conv.bias.zero_()

    def forward(self, x):
        params = self.classical_encoder(x)
        quantum_probs = self.quantum_layer(params)

        num_pixels = self.num_pixels
        output_pixels = quantum_probs[:, num_pixels:]
        output_image_scaled = output_pixels * num_pixels
        size = int(round(num_pixels ** 0.5))
        coeffs = output_image_scaled.view(-1, size, size)

        if self.use_wavelet:
            # Обернене 2D Haar-перетворення: img = H^T @ coeffs @ H
            # (H ортонормована, тож H^T -- це і є обернене перетворення).
            img = torch.einsum("ji,bjk,kl->bil", self.haar_H, coeffs, self.haar_H)
        else:
            img = coeffs

        if self.use_smoothing:
            img = self.smooth(img.unsqueeze(1)).squeeze(1)
        return img


class ColorHybridAutoencoder(nn.Module):
    """
    Те саме, що HybridAutoencoder, але для КОЛЬОРОВОГО (RGB) зображення.
    Потрібно втричі більше вихідних значень (R, G, B на кожен піксель), тому
    беремо num_qubits такими, щоб 2**num_qubits покривав 3*image_size**2
    значень -- зайві (якщо є) просто відкидаються. Практично це означає:
    для того самого image_size кольоровий варіант потребує приблизно на
    2 кубіти більше, ніж чорно-білий (бо 3 не є степенем двійки, тож
    беремо найближчу достатню ємність).
    """

    def __init__(self, classical_encoder, quantum_layer, num_qubits, image_size, use_smoothing=False):
        super().__init__()
        self.classical_encoder = classical_encoder
        self.quantum_layer = quantum_layer
        self.num_qubits = num_qubits
        self.image_size = image_size
        self.needed = 3 * image_size * image_size
        self.use_smoothing = use_smoothing

        capacity = 2 ** num_qubits
        if capacity < self.needed:
            raise ValueError(
                f"2**num_qubits={capacity} замало для кольору {image_size}x{image_size} "
                f"(потрібно {self.needed} значень: 3 канали x {image_size}x{image_size}). "
                f"Візьміть num_qubits більшим."
            )
        if use_smoothing:
            self.smooth = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3)
            with torch.no_grad():
                self.smooth.weight.fill_(1.0 / 9.0)
                self.smooth.bias.zero_()

    def forward(self, x):
        params = self.classical_encoder(x)
        quantum_probs = self.quantum_layer(params)

        num_pixels = 2 ** self.num_qubits
        output_vals = quantum_probs[:, num_pixels:num_pixels + self.needed]
        output_scaled = output_vals * num_pixels
        img = output_scaled.view(-1, 3, self.image_size, self.image_size)
        if self.use_smoothing:
            img = self.smooth(img)
        return img
