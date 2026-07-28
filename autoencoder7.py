import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    К-сть і розміри шарів налаштовуються списками:

      encoder_hidden_dims -- приховані шари енкодера (вхід -> ... -> bottleneck),
      decoder_hidden_dims -- приховані шари декодера (bottleneck -> ... -> num_params).

    Якщо списки не задані (None), береться стара захардкоджена архітектура:
    [hidden_dim] для енкодера і [hidden_dim, hidden_dim * 2] для декодера --
    тож старі чекпоінти завантажуються без змін.
    """

    def __init__(
        self,
        image_size=8,
        num_params=48,
        hidden_dim=32,
        bottleneck_size=24,
        activation="ReLU",
        channels=1,
        encoder_hidden_dims=None,
        decoder_hidden_dims=None,
    ):
        super().__init__()
        self.channels = channels
        self.input_dim = image_size * image_size * channels
        self.num_params = num_params
        try:
            self.activation_class = getattr(nn, activation)
        except AttributeError as exc:
            raise ValueError(f"Unknown activation '{activation}'") from exc

        if encoder_hidden_dims is None:
            encoder_hidden_dims = [hidden_dim]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [hidden_dim, hidden_dim * 2]
        encoder_hidden_dims = [int(d) for d in encoder_hidden_dims]
        decoder_hidden_dims = [int(d) for d in decoder_hidden_dims]
        if not encoder_hidden_dims or not decoder_hidden_dims:
            raise ValueError("encoder_hidden_dims і decoder_hidden_dims мають містити хоча б один шар.")
        if any(d <= 0 for d in encoder_hidden_dims + decoder_hidden_dims):
            raise ValueError("Розміри шарів мають бути додатними.")
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims

        encoder_layers = [nn.Flatten()]
        in_dim = self.input_dim
        for dim in encoder_hidden_dims + [bottleneck_size]:
            encoder_layers.append(nn.Linear(in_dim, dim))
            encoder_layers.append(self.activation_class())
            in_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = bottleneck_size
        for dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(in_dim, dim))
            decoder_layers.append(self.activation_class())
            in_dim = dim
        decoder_layers.append(nn.Linear(in_dim, num_params))
        # LayerNorm перед Tanh -- без неї стек Linear+ReLU легко "розганяє"
        # значення до дуже великих чисел вже при ініціалізації, Tanh
        # миттєво насичується до +-1 (тобто +-pi після масштабування) для
        # БУДЬ-ЯКОГО входу, і градієнт крізь цю ділянку майже нульовий --
        # мережа фізично не може навчитись відрізняти зображення.
        decoder_layers.append(nn.LayerNorm(num_params))
        decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded * torch.pi
