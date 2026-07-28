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

## Конфігурація архітектури (train_autoencoder7.py)

- `--num-qubits N` — кількість кубітів кола (має бути парною; image_size = 2^(N/2)).
- `--num-layers N` — кількість ансатців (повторів блоку U3 + ControlledPhaseShift).
- `--encoder-dims "128,64"` — кількість і розміри прихованих шарів енкодера
  (список через кому). За замовчуванням один шар розміру `--hidden-dim`.
- `--decoder-dims "64,128,256"` — те саме для декодера. За замовчуванням два шари:
  `--hidden-dim` і `2*--hidden-dim`.
- `--bottleneck N` — розмір вузького горла між енкодером і декодером.

Кількість параметрів кола рахується автоматично як
`total_params(num_qubits, num_layers)` з `circuit.py` (єдине джерело правди), і вихід
класичного енкодера завжди дорівнює їй; коло додатково перевіряє це на кожному
виклику й падає зі зрозумілою помилкою при розбіжності.

Приклад: 8 кубітів, 6 ансатців, енкодер 256→128, декодер 128→256→512:
```powershell
python train_autoencoder7.py --num-qubits 8 --num-layers 6 --encoder-dims "256,128" --decoder-dims "128,256,512" --bottleneck 64
```

## Оптимізація гіперпараметрів (tune_hyperparams.py)

Потребує `pip install optuna`. Шукає найкращу комбінацію: кількість ансатців,
кількість і розміри шарів енкодера й декодера, розмір bottleneck. Кубіти фіксовані
(`--num-qubits`), бо вони змінюють роздільність зображення.

```powershell
python tune_hyperparams.py --data-dir data2 --num-qubits 6 --n-trials 40 --epochs-per-trial 15
```

- Кожен trial -- коротке тренування; слабкі конфігурації обрізаються достроково
  (MedianPruner) за val_loss по епохах.
- Study зберігається у SQLite (`runs/optuna/<study>.db`): повторний запуск тієї самої
  команди ПРОДОВЖУЄ пошук з того місця, а не починає з нуля.
- Межі простору пошуку: `--min/max-ansatz`, `--max-encoder-layers`, `--max-decoder-layers`,
  `--min/max-dim`, `--min/max-bottleneck`. Ліміт часу: `--timeout` (секунди).
- Результат: `runs/optuna/<study>_best.json` з найкращою конфігурацією та ГОТОВОЮ
  командою для повного тренування через train_autoencoder7.py.

## Список файлів у пакеті

| Файл | Призначення |
|---|---|
| `circuit.py` | Побудова квантового кола (U3 + ControlledPhaseShift), підтримка шуму |
| `autoencoder7.py` | Класичний енкодер/декодер (з LayerNorm-виправленням насичення) |
| `hybrid_autoencoder7.py` | Об'єднання класичної й квантової частин, smooth-conv, вейвлет-режим |
| `dataset_loader.py` | Завантаження власних зображень (раніше `data.py`) |
| `train_autoencoder7.py` | Головний скрипт тренування (усі режими: прямий, патч, шум, вейвлет) |
| `tune_hyperparams.py` | Optuna-оптимізація: к-сть ансатців, шари енкодера/декодера, bottleneck |
| `evaluate_autoencoder7.py` | Перегляд якості реконструкції на всьому датасеті |
| `final_report7.py` | Генерація підписаного порівняння оригінал/вхід/вихід + текстовий звіт |
| `train_noise_simple.py` | Спрощене тренування з шумом (мала конфігурація, оригінальний стиль коду) |
| `test_noise_levels.py` | Перевірка стійкості до шуму гейтів (default.mixed) |
| `test_shots_noise.py` | Перевірка стійкості до шуму вимірів (shots, lightning.qubit) |
| `patch_autoencoder.py` | Патч-режим (перевірений, відхилений — гірший результат за прямий підхід) |
