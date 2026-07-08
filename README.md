# inference_server

HTTP-сервис на [userver framework](https://github.com/userver-framework/userver), который принимает изображение рукописной цифры в формате MNIST (28x28, нормализованные пиксели) и возвращает предсказание CNN-модели, экспортированной в ONNX.

## Сборка через Docker

```
docker build -t inference_server .
```

## Запуск

```
docker run --rm -p 8080:8080 -v $(pwd)/models:/app/models inference_server
```

Убедитесь, что `models/mnist.onnx` существует — модель в образ не встраивается, монтируется отдельно.

## Пример запроса

```
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"pixels": [0.0, 0.0, ..., 0.98, 0.0]}'
```

Ответ:

```json
{
  "digit": 7,
  "confidence": 0.98,
  "probabilities": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.98, 0.01, 0.01]
}
```

## Архитектура

`Inference` оборачивает ONNX Runtime C++ API: грузит модель один раз при старте и гоняет батч изображений через `session.Run()`, превращая логиты в вероятности через softmax.

`BatchCollector` между HTTP-хендлером и `Inference`. Каждый запрос кладёт пушит изображение в общую очередь и получает `std::future` на результат; фоновый поток копит запросы и отправляет их в `Inference` одним батчем, как только очередь достигла `max_batch_size` или истёк `timeout_ms` — так инференс на GPU/CPU не гоняется по одному изображению за раз.

`PredictHandler` — обычный userver HTTP-хендлер на `/predict`: разбирает JSON тело через `userver::formats::json`, отдаёт изображение в `BatchCollector` и блокируется на future до получения результата, который сериализует обратно в JSON.
