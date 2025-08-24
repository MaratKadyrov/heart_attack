import re
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OneHotEncoder,
                                   OrdinalEncoder,
                                   StandardScaler,
                                   MinMaxScaler,
                                   LabelEncoder)

RANDOM_STATE = 42
MODEL_PATH = "models/model_heart_attack.pkl"
PIPELINE_PATH = "models/pipeline.pkl"

OHE_COLUMNS = [
    'Gender'
]

ORD_COLUMNS = ['Stress Level',
    'Physical Activity Days Per Week',
    'Sleep Hours Per Day'
]

NUM_COLUMNS = [
    'Age',
    'Cholesterol',
    'Heart rate',
    'Exercise Hours Per Week',
    'Sedentary Hours Per Day',
    'Income',
    'BMI',
    'Triglycerides',
    'Blood sugar',
    'CK-MB',
    'Troponin',
    'Systolic blood pressure',
    'Diastolic blood pressure',
    'Diabetes',
    'Family History',
    'Smoking',
    'Obesity',
    'Alcohol Consumption',
    'Diet',
    'Previous Heart Problems',
    'Medication Use'
]

class Model():
    def trans_func(input):
        """
        Преобразует входные данные и заменяет значения.

        Функция заменяет 1.0 и 0.0 на Male и Female соответственно

        Параметры:
        - input: Массив или список данных, которые необходимо обработать.

        Возвращает:
        - Обработанный массив NumPy с замененными значениями.
        """

        if not isinstance(input, np.ndarray):
            input = np.array(input)

        # заменяем ' ' на Nan
        output = np.where(input == '1.0', 'Male', input)
        output = np.where(input == '0.0', 'Female', output)

        return output
    def __init__(self, model_path=MODEL_PATH):
        with open(model_path, 'rb') as f:
            self.model = joblib.load(f)



    def prediction_pipeline(self):
        """
        Загружаем prefitted pipeline
        :return: pipeline
        """

        with open(PIPELINE_PATH, 'rb') as f:
            final_pipe = joblib.load(f)

        return final_pipe

    def predict(self, save_pth):
        # Открытие загруженного файла
        data = pd.read_csv(save_pth)
        data['Gender'] = data['Gender'].replace('1.0', 'Male')
        data['Gender'] = data['Gender'].replace('0.0', 'Female')

        preprocessor = self.prediction_pipeline()
        prediction = self.model.predict(preprocessor.transform(data))
        responce_json = pd.concat(
            [
            data['id'],
            pd.Series(prediction)
            ],
            axis=1
        )
        responce_json.columns = ['id', 'prediction']
        return responce_json