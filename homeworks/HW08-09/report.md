# HW08-09 – PyTorch MLP: регуляризация и оптимизация обучения

## 1. Кратко: что сделано

- **Какой датасет выбран (A/B/C) и почему.** Вариант C: CIFAR10. 10 классов, 32×32×3, доступен через torchvision без проблем с загрузкой.
- **Что сравнивалось в части A (регуляризация):** E1 — base MLP без регуляризации; E2 — + Dropout; E3 — + BatchNorm; E4 — лучший из E2/E3 с EarlyStopping.
- **Что сравнивалось в части B (оптимизация):** O1 — слишком большой LR; O2 — слишком маленький LR; O3 — SGD+momentum+weight decay.

## 2. Среда и воспроизводимость

- Python: 3.10
- torch / torchvision: 2.10.0 / 0.25.0
- Устройство (CPU/GPU): CPU
- Seed: 42 (torch, numpy)
- Как запустить: открыть `HW08-09.ipynb` и выполнить Run All.

## 3. Данные

- Датасет: CIFAR10
- Разделение: train/val 80/20 от стандартного train; test — стандартный test из torchvision. Seed=42.
- Трансформации (transform): ToTensor()
- Комментарий: 10 классов, 32×32×3 канала. Валидация отделена от train для оценки переобучения.

## 4. Базовая модель и обучение

- Модель MLP (кратко): Flatten → Linear(3072, 256) → ReLU → [Dropout/BatchNorm] → Linear(256, 128) → ReLU → … → Linear(→ 10). 2 скрытых слоя.
- Loss: CrossEntropyLoss
- Базовый Optimizer (для части A): Adam (lr=1e-3)
- Batch size: 128
- Epochs (макс): 20 (EarlyStopping обрезает раньше)
- EarlyStopping: patience=5, metric=val_accuracy

## 5. Часть A (S08): регуляризация (E1–E4)

- **E1 (base):** 2 скрытых слоя (256, 128), без Dropout/BatchNorm.
- **E2 (Dropout):** как E1 + Dropout(p=0.3).
- **E3 (BatchNorm):** как E1 + BatchNorm1d между Linear и ReLU.
- **E4 (EarlyStopping):** лучший из (E2/E3) по val_accuracy + EarlyStopping (patience=5). Сохранён `best_model.pt`.

## 6. Часть B (S09): LR, оптимизаторы, weight decay (O1–O3)

- **O1:** LR слишком большой — Adam, lr=0.1, 6 эпох.
- **O2:** LR слишком маленький — Adam, lr=1e-5, 6 эпох.
- **O3:** SGD, momentum=0.9, weight_decay=1e-4, lr=1e-2, 10–12 эпох.

## 7. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: [./artifacts/runs.csv](./artifacts/runs.csv)
- Лучшая модель: [./artifacts/best_model.pt](./artifacts/best_model.pt)
- Конфиг лучшей модели: [./artifacts/best_config.json](./artifacts/best_config.json)
- Кривые лучшего прогона: [./artifacts/figures/curves_best.png](./artifacts/figures/curves_best.png)
- Кривые "плохих LR": [./artifacts/figures/curves_lr_extremes.png](./artifacts/figures/curves_lr_extremes.png)

Короткая сводка:

- Лучший эксперимент части A: E4 (EarlyStopping на лучшем из E2/E3).
- Лучшая val_accuracy: 0.4712 (E4).
- Итоговая test_accuracy (для лучшей модели): 0.4634.
- O1 (слишком большой LR): loss нестабилен или высок, accuracy низкая (~0.23).
- O2 (слишком маленький LR): метрики почти не меняются, обучение не идёт.
- O3 (SGD+momentum+weight decay): по метрикам близко к Adam, может сходиться чуть иначе по кривым.

## 8. Анализ

На графиках E1 видно переобучение: train loss падает, val loss перестаёт улучшаться или растёт. Dropout (E2) и BatchNorm (E3) снижают переобучение и улучшают val_accuracy. EarlyStopping в E4 останавливает обучение при отсутствии улучшения val_accuracy в течение 5 эпох и сохраняет лучшую модель по val. O1 при слишком большом LR даёт нестабильный или высокий loss и низкую accuracy. O2 при слишком маленьком LR почти не меняет loss и accuracy за 6 эпох. SGD+momentum с weight decay даёт регуляризацию весов и может сходиться к сопоставимому качеству с Adam. Выбранный конфиг (E4 с Dropout и EarlyStopping) разумен для CIFAR10 на MLP: ограничивает переобучение и даёт лучшую val/test accuracy.

## 9. Итоговый вывод

Базовый конфиг: E4 — MLP с Dropout(0.3) и EarlyStopping (patience=5), Adam lr=1e-3. Он даёт лучшую val_accuracy и разумную test_accuracy. Дальше можно попробовать: нормализацию входов (Normalize для CIFAR10) и подбор learning rate (например, scheduler или перебор lr).

## 10. Приложение (опционально)

Не выполнялось.
