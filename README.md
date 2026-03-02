# RescueAI: запуск через Docker

## Для тех, кто заходит в репозиторий впервые

1. Установите Docker Desktop.
2. Запустите Docker Desktop и дождитесь статуса `Engine running`.
3. Откройте Терминал.

## Клонирование и запуск

1. Клонируйте репозиторий:

```bash
git clone https://github.com/ykvnkm/diplom-prod.git
```

2. Перейдите в папку проекта:

```bash
cd diplom-prod
```

3. Запустите сервис:

```bash
docker compose -f docker-compose.unified.yml up --build
```

4. Дождитесь логов без ошибок и сообщений о запуске `uvicorn`.

## Куда заходить

Откройте в браузере:

`http://127.0.0.1:8010`

## Как остановить

1. Вернитесь в окно терминала, где запущен сервис.
2. Нажмите `Ctrl + C`.

Чтобы убрать контейнеры и сеть:

```bash
docker compose -f docker-compose.unified.yml down
```
