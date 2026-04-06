# HW08-09 – PyTorch MLP: регуляризация и оптимизация обучения

## 1. Кратко: что сделано

- **Какой датасет выбран (A/B/C) и почему.** Использован **FashionMNIST** (аналогично MNIST/KMNIST: 28×28, 10 классов). Выбран из‑за стабильной загрузки через torchvision; по смыслу соответствует «простому» варианту A для MLP.
- **Что сравнивалось в части A (регуляризация):** E1 — базовый MLP; E2 — + Dropout(0.3); E3 — + BatchNorm; E4 — архитектура лучшего из E2/E3 по `val_accuracy` (оказался E2) + EarlyStopping.
- **Что сравнивалось в части B (оптимизация):** O1 — слишком большой LR (Adam 0.1); O2 — слишком маленький (Adam 1e-5); O3 — SGD+momentum+weight decay.

## 2. Среда и воспроизводимость

- Python: 3.14 (локальная venv в репозитории)
- torch / torchvision: см. вывод первой ячейки ноутбука после Run All
- Устройство (CPU/GPU): CPU (или CUDA при наличии)
- Seed: 42 (torch, numpy)
- Как запустить: открыть `homeworks/HW08-09/HW08-09.ipynb` и выполнить Run All из каталога `HW08-09/`.

## 3. Данные

- Датасет: FashionMNIST
- Разделение: от стандартного `train` отделено 20% на `val` (`random_split` с генератором `seed=42`); `test` — официальный test из torchvision.
- Трансформации (transform): `ToTensor()` (значения в [0, 1]).
- Комментарий: 10 классов, 784 признака после Flatten; задача линейно разделима хуже, чем чистый MNIST, поэтому видны эффекты регуляризации и настройки оптимизатора.

## 4. Базовая модель и обучение

- Модель MLP (кратко): Flatten → Linear(784→256) → ReLU → … → Linear(128→10); в E2/E4 после ReLU — Dropout(0.3); в E3 — BatchNorm1d после Linear, затем ReLU.
- Loss: `CrossEntropyLoss`
- Базовый Optimizer (для части A): Adam (lr=1e-3)
- Batch size: 256
- Epochs (макс): 15 для E1–E3 и верхняя граница для E4
- EarlyStopping: patience=4, метрика улучшения — `val_accuracy` (сохраняется лучший чекпоинт)

## 5. Часть A (S08): регуляризация (E1-E4)

- **E1 (base):** два скрытых слоя 256×128, без Dropout/BatchNorm.
- **E2 (Dropout):** как E1 + Dropout(p=0.3) после активаций в скрытых слоях.
- **E3 (BatchNorm):** как E1 + BatchNorm1d между Linear и ReLU.
- **E4 (EarlyStopping):** по `val_accuracy` между E2 и E3 выбран **E2**; обучение с EarlyStopping и сохранением `best_model.pt`.

## 6. Часть B (S09): LR, оптимизаторы, weight decay (O1-O3)

- **O1:** Adam, lr=0.1 (слишком большой), 7 эпох.
- **O2:** Adam, lr=1e-5 (слишком маленький), 7 эпох.
- **O3:** SGD, momentum=0.9, weight_decay=1e-4, lr=1e-2, 12 эпох.

## 7. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: [./artifacts/runs.csv](./artifacts/runs.csv)
- Лучшая модель: [./artifacts/best_model.pt](./artifacts/best_model.pt)
- Конфиг лучшей модели: [./artifacts/best_config.json](./artifacts/best_config.json)
- Кривые лучшего прогона: [./artifacts/figures/curves_best.png](./artifacts/figures/curves_best.png)
- Кривые «плохих LR»: [./artifacts/figures/curves_lr_extremes.png](./artifacts/figures/curves_lr_extremes.png)

Короткая сводка:

- Лучший эксперимент части A: **E4** (E2 + EarlyStopping), финальная оценка на test — один раз после обучения.
- Лучшая val_accuracy (E4): см. `best_config.json` и `runs.csv` (после выполнения ноутбука).
- Итоговая test_accuracy (лучшая модель E4): см. `best_config.json`.
- **O1:** очень большой LR — рост loss, валидация около случайного уровня (~0.1 accuracy по классам).
- **O2:** очень маленький LR — loss и accuracy почти не улучшаются за 7 эпох.
- **O3:** SGD+momentum+wd даёт сходимость без Adam, качество сопоставимо с разумным диапазоном, но ниже удачного Adam на этой задаче.

## 8. Анализ

На FashionMNIST базовый MLP (E1) уже даёт высокую точность; Dropout и BatchNorm по отдельности дают сопоставимое качество на `val`, без явного «провала» train/val как на сложных данных. EarlyStopping в E4 останавливает переобучение и фиксирует лучший чекпоинт по `val_accuracy`. O1 демонстрирует нестабильное/плохое обучение при завышенном LR. O2 — почти отсутствие прогресса при заниженном LR. O3 показывает роль явной L2-регуляризации (`weight_decay`) и momentum в SGD. Для данного датасета разумный выбор — Adam с умеренным lr и при необходимости Dropout + EarlyStopping.

## 9. Итоговый вывод

Базовый практичный конфиг: Adam lr=1e-3, MLP с Dropout(0.3), EarlyStopping по `val_accuracy` — как в E4. Дальше имеет смысл попробовать нормализацию входов, scheduler LR или чуть большую модель; для изображений сильнее выиграют свёрточные сети, но для учебной MLP-части достаточно показанных приёмов.

## 10. Приложение (опционально)

Не выполнялось.
