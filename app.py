import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, FileResponse
import shutil
import uvicorn
import argparse
import logging
from fastapi.staticfiles import StaticFiles
import pandas as pd
from model import Model

MODEL_PATH = "models/model_heart_attack.pkl"
UPLOAD_FOLDER = "uploads"
PREDICTION_PATH = "predictions.csv"

# Создание папки, если она не существует
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Инициализация приложения
app = FastAPI(
    title = "Heart Attack prediction",
    description = "Сервис предсказания сердечного приступа",
    version = "0.1",
    contact={
        "name": "Marat Kadyrov",
        "url": "https://github.com/MaratKadyrov",
        "email": "kadyrovmsh@gmail.com"
    }

)

app.mount("/tmp", StaticFiles(directory="tmp"), name='images')
templates = Jinja2Templates(directory="templates")

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_handler = logging.StreamHandler()
app_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)

# Инициализируем модель
try:
    model = Model(MODEL_PATH)
except Exception as e:
    print(f"Ошибка - {e}")

# Проверка API
@app.get(
    path="/health",
    summary="Проверка работоспособности API",
    description=("Выполняет проверку работоспособности API. Возвращает OK.\n"
        "Используется для мониторинга состояния сервиса.\n"
        "Пример запроса: GET /health\n"
        "Пример ответа: {'status': 'OK'}\n"
        "Код состояния: 200"
                 )
)
def health():
    return {"status": "OK"}

# Загрузка главной страницы
@app.get(
    path="/",
    summary="Загрузка главной страницы",
    responses={
        200: {"description": "Успешный ответ"},
        404: {"description": "Страница не найдена"},
        500: {"description": "Внутренняя ошибка сервера"}}
)
def main(request: Request):
    return templates.TemplateResponse("start_form.html",
                                      {"request": request})

# предсказание
@app.post(
    path="/predict",
    summary="Выполняет предсказание сердечного приступа на основе загруженного на сервер .csv файла",
    description="Выполняет предсказание сердечного приступа, используя данные из "
        "загруженного .csv файла. Возвращает JSON с предсказанием.\n\n"
        "Формат входного .csv файла: ['Age', -'Cholesterol', 'Heart rate', 'Diabetes', 'Family History',\
       'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week',\
       'Diet', 'Previous Heart Problems', 'Medication Use', 'Stress Level',\
       'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',\
       'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Blood sugar',\
       'CK-MB', 'Troponin', 'Gender', 'Systolic blood pressure',\
       'Diastolic blood pressure']\n"
        "Пример возвращаемого JSON: {\"predictions\":\"[{\"id\":7746,\"prediction\":0.0}}\n\n"
        "Коды состояния: \n"
        "- 200: Успешное предсказание\n"
        "- 422: Семантическая ошибка\n"
        "- 500: Ошибка сервера\n"
)
def predict(file: UploadFile, request: Request):
    # Сохраняем файл
    save_pth = "tmp/" + file.filename
    app_logger.info(f'processing file - {save_pth}')
    with open(save_pth, "wb") as fid:
        fid.write(file.file.read())

    # Выполнение предсказания
    predictions = model.predict(save_pth)
    response_json = predictions.to_json(orient='records')

    # возвращаем JSON
    return JSONResponse(content={"predictions": response_json})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())
    uvicorn.run(app, **args)