## Команды для использования triton

0. поднимаем poetry-окружение `poetry shell`
1. `dvc pull` - подтягиваем веса модели(onnx), train и test датасеты(csv файлы)
2. копируем data/model.onnx в triton/onnx-human-activity/1
3. собираем докер-контейнер `docker-compose up --build`
4. запускаем контейнер
   `docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.04-py3-sdk`
5. запускаем perf_analyzer
   `perf_analyzer -m onnx-human-activity -u localhost:8500 --concurrency-range 8 --shape inputs:1,562 --shape predictions:1,6 --input-data zero --measurement-mode count_windows --measurement-request-count 100000`
6. для тестов работы triton запускаем `python3 client.py` из папки triton

## Отчет о системе

- Maс OS 11.6.2 Big Sur
- CPU: Intel Core i5 1.6 GHZ, 2 Cores
- RAM: 8GB 2133 MHz LPDDR3

## О наборе данных

База данных распознавания человеческой деятельности была построена на основе
записей 30 участников исследования, выполняющих повседневную деятельность (ADL),
при этом они носили закрепленный на поясе смартфон со встроенными инерционными
датчиками.

Цель состоит в том, чтобы классифицировать действия по одному из шести
выполняемых видов деятельности.

Эксперименты проводились на группе из 30 добровольцев в возрасте от 19 до 48
лет.

Каждый человек выполнял шесть действий (WALKING, WALKING_UPSTAIRS,
WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING), нося смартфон (Samsung Galaxy S
II) на поясе.

## Отчет о проделанной работе

Данные взяты из
https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones

Построена torch-модель из 6 линейных слоев, архитектура в model.py.

| max_queue_delay_microseconds |    throughput     |   latency |
| ---------------------------- | :---------------: | --------: |
| 100                          | 3077.37 infer/sec | 2598 usec |
| 500                          | 3671.47 infer/sec | 2178 usec |
| 1000                         | 3724.43 infer/sec | 2147 usec |
