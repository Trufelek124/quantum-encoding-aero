import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

IMG_EXTENSIONS = (".jpg", ".jpeg", ".JPG", ".JPEG")


class ImageFolderDataset(Dataset):
    """
    Рекурсивно збирає всі .jpg/.jpeg у data_dir, ресайзить до
    (image_size, image_size), за замовчуванням конвертує у grayscale
    (1 канал) -- саме такий формат очікує існуючий Autoencoder/
    HybridAutoencoder (input_dim = image_size*image_size, без каналів).

    Повертає (image_tensor, 0) -- другий елемент фіктивний (клас/label),
    він ігнорується скрізь у training.py (автоенкодеру label не потрібен),
    лишений лише для сумісності з циклом `for batch_images, _ in loader`.
    """

    def __init__(self, data_dir, image_size, grayscale=True):
        self.paths = []
        for ext in IMG_EXTENSIONS:
            self.paths.extend(glob.glob(os.path.join(data_dir, "**", f"*{ext}"), recursive=True))
        # Прибираємо контактні листи (_contact_*.jpg), які ми самі генерували
        # для ручної розмітки -- це не реальні дані, а мозаїки-прев'ю.
        self.paths = [p for p in self.paths if not os.path.basename(p).startswith("_contact")]
        self.paths = sorted(set(self.paths))
        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"У '{data_dir}' не знайдено жодного .jpg/.jpeg файлу "
                f"(шукав рекурсивно по підпапках)."
            )

        tfs = [transforms.Resize((image_size, image_size))]
        if grayscale:
            tfs.append(transforms.Grayscale(num_output_channels=1))
        tfs.append(transforms.ToTensor())
        self.transform = transforms.Compose(tfs)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, 0


def get_image_dataloaders(data_dir, image_size=8, batch_size=16, val_fraction=0.1,
                           grayscale=True, seed=42):
    dataset = ImageFolderDataset(data_dir, image_size, grayscale=grayscale)

    if len(dataset) < 2:
        # Sanity-check режим: 1 (чи 2) зображення -- використовуємо той самий
        # датасет і для train, і для val, аби просто перевірити, що модель
        # взагалі прокручується. Для реального навчання так, звісно, не робимо.
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    num_val = max(1, int(len(dataset) * val_fraction))
    num_train = len(dataset) - num_val
    if num_train < 1:
        num_train, num_val = len(dataset) - 1, 1

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def get_mnist_dataloaders(image_size=8, batch_size=16):
    """Лишено для порівняння з baseline / --dataset mnist у training.py."""
    from torchvision import datasets

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
