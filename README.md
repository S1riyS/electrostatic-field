# Исходный код
Исходный код расположен в GitHub репозитории по адресу https://github.com/S1riyS/electrostatic-field

# Размещение
Проект размещен публично по адресу https://electro.lavrentious.ru (у сервера ограниченная вычислительная мощнсть, расчёт может занять до 1 минуты)

# Инструкция по локальному запуску приложения (в режиме разработки):
Требуемые инструменты:
 - [**NodeJS**](https://nodejs.org/en)
 - [**Python**](https://www.python.org/) (3.10+)
 - [**Poetry**](https://python-poetry.org/docs/#installing-with-the-official-installer)

# Настройка и запуск:

0. Клонирование исходного кода: `git clone https://github.com/S1riyS/electrostatic-field.git`
1. Клиент (в директории `client`):
	1. **Установка зависимостей**: `npm install`
	2. **Конфигурация**: cоздать конфигурационный файл `.env.development` с содержанием ```VITE_API_BASE_URL=http://localhost:8000```
	3. **Запуск**: `npm run dev`
2. Сервер (в директории `server`):
	1. **Установка зависимостей** и создание окружения: `poetry install`
	2. **Конфигурация**: cоздать конфигурационный файл `.env` с содержанием
		```env
		APP_PORT=8000
		ALLOWED_ORIGINS=http://localhost:5173
		```
	3. **Запуск**: `poetry run fastapi dev src/main.py`

Приложение будет доступно в браузере по адресу http://localhost:5137