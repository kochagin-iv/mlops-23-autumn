# mlops-23-autumn

Project for my tasks in mlops course, autumn 2023

Порядок работы с репозиторием

1. `git clone`
2. `poetry shell` - подъем виртуального окружения
3. `poetry install`
4. `pre-commit install`
5. `pre-commit run -a`
6. `python3 mlops-23-autumn/train.py` - обучение модели
7. `python3 mlops-23-autumn/infer.py` - получение предсказаний

Структура проекта:

```
    .
    └── mlops-23-autumn
        ├── answers
        |   └── *.csv - предсказания модели
        ├── configs
        |   ├── conf.yaml - настройки путей, количества эпох модели
        ├── datasets - сохраненные датасеты для обучения
        ├── graphics - графики, полученные в результате обучения и предсказания
        ├── infer.py - предсказание модели
        ├── train.py - обучение модели
        └── model.py - структура модели
```
