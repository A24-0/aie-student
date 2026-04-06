# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

- Для части A выбран датасет **STL10** (10 классов, 96×96): удобный размер изображений и официальные train/test для сравнения C1–C4.
- Для части B выбраны **Pascal VOC 2012** и трек **segmentation**: есть готовые маски и модель FCN/DeepLab из torchvision, без обучения с нуля.
- В части A сравнивались C1 (CNN без аугментаций), C2 (та же CNN с аугментациями), C3 (ResNet18, только голова), C4 (ResNet18, layer4+fc). В части B — режимы V1 (argmax) и V2 (median filter по маске).

## 2. Среда и воспроизводимость

- Python:
- torch / torchvision:
- Устройство (CPU/GPU): CPU или CUDA (см. вывод ноутбука).
- Seed: 42.
- Как запустить: открыть `HW10-11.ipynb` и выполнить Run All из каталога `homeworks/HW10-11/` (первый запрос скачает STL10 и VOC).

## 3. Данные

### 3.1. Часть A: классификация

- Датасет: STL10
- Разделение: train/val/test — из официального `train` отделено 20% на val (`random_split`, seed=42); test — официальный `test`.
- Базовые transforms: `ToTensor`, нормализация с mean/std 0.5 по каналам.
- Augmentation transforms: `RandomHorizontalFlip`, `RandomCrop(96, padding=8)` (см. ноутбук).
- Комментарий: 10 классов, изображения 96×96; задача проще ImageNet, но сложнее CIFAR-за счёт меньшего train (5000 изображений), поэтому transfer learning на ResNet даёт заметный эффект.

### 3.2. Часть B: structured vision

- Датасет: Pascal VOC
- Трек: segmentation
- Что считается ground truth: пиксельные маски VOC (21 класс включая фон; значение 255 — игнор на границах).
- Какие предсказания использовались: выход **FCN ResNet50** с весами VOC, затем argmax по классам; для V2 — тот же argmax + median filter 5×5 по меткам.
- Комментарий: VOC — стандартный бенчмарк сегментации; предобученная FCN совпадает по классам с разметкой, что упрощает интерпретацию mIoU без дообучения.

## 4. Часть A: модели и обучение (C1-C4)

Опишите коротко и сопоставимо:

- C1 (simple-cnn-base): свёрточная сеть из ноутбука (несколько блоков Conv+Pool), обучение на train без аугментаций.
- C2 (simple-cnn-aug): та же архитектура, train с аугментациями.
- C3 (resnet18-head-only): ResNet18 с весами ImageNet; заморожен backbone, обучается только `fc`.
- C4 (resnet18-finetune): ResNet18; разморожены `layer4` и `fc`, меньший lr на дообучаемые параметры.

Дополнительно:

- Loss: CrossEntropyLoss
- Optimizer(ы): Adam (для CNN и ResNet; см. ноутбук)
- Batch size: 32
- Epochs (макс): заданы в ноутбуке (CNN и ResNet отдельно)
- Критерий выбора лучшей модели: наибольшая **val_accuracy**; на **test** оценивается только победитель C1–C4.

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

### Если выбран segmentation track

- Модель: FCN ResNet50 (pretrained на PASCAL VOC, torchvision).
- Что считается foreground: для учебной метрики используется **полная multi-class** маска; в отчёте фиксируется средний IoU по классам с ненулевым union (аналог mIoU на подвыборке изображений). При необходимости можно выделить один класс как «объект» — здесь достаточно согласованности предсказаний и GT по всем классам VOC.
- V1: базовая постобработка — **argmax** по логитам модели.
- V2: альтернативная постобработка — **median filter 5×5** по карте меток классов (сглаживание шума).
- Как считался mean IoU: для каждого изображения — IoU по каждому классу c, где есть пиксели в pred или GT; усреднение по классам и по подвыборке батча (см. ноутбук).
- Считались ли дополнительные pixel-level метрики: да, доля верно классифицированных пикселей (в `runs.csv` отражена в доступных полях; при необходимости см. вывод ноутбука).

## 6. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: [./artifacts/runs.csv](./artifacts/runs.csv)
- Лучшая модель части A: [./artifacts/best_classifier.pt](./artifacts/best_classifier.pt)
- Конфиг лучшей модели части A: [./artifacts/best_classifier_config.json](./artifacts/best_classifier_config.json)
- Кривые лучшего прогона классификации: [./artifacts/figures/classification_curves_best.png](./artifacts/figures/classification_curves_best.png)
- Сравнение C1-C4: [./artifacts/figures/classification_compare.png](./artifacts/figures/classification_compare.png)
- Визуализация аугментаций: [./artifacts/figures/augmentations_preview.png](./artifacts/figures/augmentations_preview.png)
- Визуализации второй части: [./artifacts/figures/segmentation_examples.png](./artifacts/figures/segmentation_examples.png), [./artifacts/figures/segmentation_metrics.png](./artifacts/figures/segmentation_metrics.png)

Короткая сводка (6-10 строк):

- Лучший эксперимент части A: см. `best_classifier_config.json` / столбец `test_accuracy` в `runs.csv` для эксперимента-победителя.
- Лучшая `val_accuracy`: см. `runs.csv` (C1–C4).
- Итоговая `test_accuracy` лучшего классификатора: см. строку лучшего эксперимента в `runs.csv`.
- Что дали аугментации (C2 vs C1): сравнить `best_val_accuracy` C2 и C1.
- Что дал transfer learning (C3/C4 vs C1/C2): сравнить val C3/C4 с C1/C2.
- Что оказалось лучше: head-only или partial fine-tuning: сравнить C3 и C4 по val.
- Что показал режим V1 во второй части: базовый argmax, mIoU и pixel accuracy на подвыборке.
- Что показал режим V2 во второй части: сглаженные маски, изменение mIoU/accuracy относительно V1.
- Как интерпретируются метрики второй части: mIoU отражает перекрытие классов; pixel accuracy — долю верных пикселей; рост порога у детекции не используется (трек сегментации).

## 7. Анализ

На STL10 простая CNN быстро сходится, но аугментации снижают переобучение и часто улучшают val. Pretrained ResNet18 переносит признаки ImageNet; обучение только головы стабильно при малых данных, а дообучение `layer4` может дать прирост за счёт адаптации под домен. В сегментации FCN даёт разумные маски; median filter уменьшает «шум» меток на границах, что видно на mIoU и визуально. Ошибки чаще на мелких объектах и на границах классов.

## 8. Итоговый вывод

Базовый конфиг для STL10: ResNet18 + либо голова (быстро), либо короткое дообучение layer4+fc при необходимости точности. Transfer learning переносит общие признаки; для сегментации важны согласованность метрик с маской и визуальный контроль.

## 9. Приложение (опционально)

Не выполнялось.
