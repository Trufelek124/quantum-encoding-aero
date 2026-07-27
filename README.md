# Найкращі результати дослідження — файли та команди відтворення

Усі файли в цьому пакеті узгоджені між собою (правильні імпорти). Покладіть їх усі
в одну робочу папку разом з `data2` (папка з вашим тестовим знімком).

## Результат №1: Максимальна якість реконструкції

**MAE = 0.0334** — найкращий показник за все дослідження. 32×32, форма об'єкта
на реконструкції візуально помітна.

```powershell
C:\Users\user\.conda\envs\myenv\python.exe train_autoencoder7.py --num-qubits 10 --data-dir data2 --epochs 200 --hidden-dim 512 --bottleneck 256 --num-layers 8 --smooth-conv --smooth-channels 64 --smooth-layers 4
```

Перевірка результату (підставте реальну назву папки з `dir runs`):
```powershell
C:\Users\user\.conda\envs\myenv\python.exe final_report7.py --run-dir runs\НАЗВА_ПАПКИ --data-dir data2 --num-qubits 10 --num-layers 8 --hidden-dim 512 --bottleneck 256 --smooth-conv --smooth-channels 64 --smooth-layers 4 --out report_best_quality
```

## Результат №2: Стійкість до шуму (готовність до реального заліза)

Мале, швидке коло (8×8), натреноване одразу з шумом вимірів. Чесно перевірене
(не "пастка оманливого MAE").

```powershell
C:\Users\user\.conda\envs\myenv\python.exe train_noise_simple.py
```

(цей скрипт сам прожене базовий рівень + рівень шуму 0.01 по 200 епох кожен;
за потреби відредагуйте список `noise_levels_to_test` та `epochs` у самому файлі)

## Перевірка стійкості до шуму (для обох результатів)

**Варіант A — шум від помилок гейтів** (повільніший, `default.mixed`):
```powershell
C:\Users\user\.conda\envs\myenv\python.exe test_noise_levels.py --checkpoint runs\НАЗВА_ПАПКИ\model_FINAL.pth --num-qubits 10 --num-layers 8 --hidden-dim 512 --bottleneck 256 --smooth-conv --smooth-channels 64 --smooth-layers 4 --noise-levels 0.01 0.05 0.1
```

**Варіант B — шум від обмеженої кількості вимірів (shots)** (швидший, `lightning.qubit`,
рекомендовано як пріоритетний наступний тест):
```powershell
C:\Users\user\.conda\envs\myenv\python.exe test_shots_noise.py --checkpoint runs\НАЗВА_ПАПКИ\model_FINAL.pth --num-qubits 10 --num-layers 8 --hidden-dim 512 --bottleneck 256 --smooth-conv --smooth-channels 64 --smooth-layers 4 --shots-levels 1024 256 64
```

## Список файлів у пакеті

| Файл | Призначення |
|---|---|
| `circuit.py` | Побудова квантового кола (U3 + ControlledPhaseShift), підтримка шуму |
| `autoencoder7.py` | Класичний енкодер/декодер (з LayerNorm-виправленням насичення) |
| `hybrid_autoencoder7.py` | Об'єднання класичної й квантової частин, smooth-conv, вейвлет-режим |
| `dataset_loader.py` | Завантаження власних зображень (раніше `data.py`) |
| `train_autoencoder7.py` | Головний скрипт тренування (усі режими: прямий, патч, шум, вейвлет) |
| `evaluate_autoencoder7.py` | Перегляд якості реконструкції на всьому датасеті |
| `final_report7.py` | Генерація підписаного порівняння оригінал/вхід/вихід + текстовий звіт |
| `train_noise_simple.py` | Спрощене тренування з шумом (мала конфігурація, оригінальний стиль коду) |
| `test_noise_levels.py` | Перевірка стійкості до шуму гейтів (default.mixed) |
| `test_shots_noise.py` | Перевірка стійкості до шуму вимірів (shots, lightning.qubit) |
| `patch_autoencoder.py` | Патч-режим (перевірений, відхилений — гірший результат за прямий підхід) |
