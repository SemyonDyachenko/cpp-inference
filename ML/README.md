# MNIST Model Training

## Установка

```
pip install -r requirements.txt
```

## Обучение

```
python train.py
```

## Экспорт в ONNX

```
python export.py
```

## Проверка

```
python test_model.py
```

## Структура модели

```
Input: (batch_size, 1, 28, 28) — нормализованные изображения
Conv1: 32 фильтра 3x3 + ReLU + MaxPool → (batch_size, 32, 14, 14)
Conv2: 64 фильтра 3x3 + ReLU + MaxPool → (batch_size, 64, 7, 7)
FC1: 3136 → 128 + ReLU + Dropout
FC2: 128 → 10 (логиты для цифр 0-9)
```

Нормализация входа: mean=0.1307, std=0.3081
