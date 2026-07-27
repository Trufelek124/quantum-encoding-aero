import torch
import torch.nn as nn

class Autoencoder(nn.Module):

    def __init__(
        self,
        image_size=8,
        num_params=48,
        hidden_dim=32,
        bottleneck_size=24,
        activation="ReLU",
        channels=1,
    ):
        super().__init__()
        self.channels = channels
        self.input_dim = image_size * image_size * channels
        try:
            self.activation_class = getattr(nn, activation)
        except AttributeError as exc:
            raise ValueError(f"Unknown activation '{activation}'") from exc

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, hidden_dim),
            self.activation_class(),
            nn.Linear(hidden_dim, bottleneck_size),
            self.activation_class(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_dim),
            self.activation_class(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            self.activation_class(),
            nn.Linear(hidden_dim * 2, num_params),
            # LayerNorm перед Tanh -- без неї стек Linear+ReLU легко "розганяє"
            # значення до дуже великих чисел вже при ініціалізації, Tanh
            # миттєво насичується до +-1 (тобто +-pi після масштабування) для
            # БУДЬ-ЯКОГО входу, і градієнт крізь цю ділянку майже нульовий --
            # мережа фізично не може навчитись відрізняти зображення.
            nn.LayerNorm(num_params),
            nn.Tanh(),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded * torch.pi
