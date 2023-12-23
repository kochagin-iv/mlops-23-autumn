## Команды для использования triton

0. поднимаем poetry-окружение `poetry shell`
1. `dvc pull` - подтягиваем веса модели, train и test датасеты
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

## Отчет о проделанной работе

Данные взяты из
https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones

Построена модель из 6 линейных слоев, архитектура в model.py.
