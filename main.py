from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat, Field
import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import List
from datetime import datetime

# --- 1. Инициализация ---
app = FastAPI()
MODEL_PATH = 'xgboost_no_calibration.pkl'
HISTORY_FILE = 'predictions.json'

# --- 2. Загружаем модель ---
pipeline = joblib.load(MODEL_PATH)

# --- 3. Загружаем историю, если файл существует ---
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'r') as f:
        prediction_history = json.load(f)
    request_counter = max([r["id"] for r in prediction_history], default=0)
else:
    prediction_history = []
    request_counter = 0

# --- 4. Pydantic модель ---
class PredictionInput(BaseModel):
    Age: conint(ge=18, le=100)
    AdSpend: confloat(ge=0)
    ClickThroughRate: confloat(ge=0, le=1)
    WebsiteVisits: conint(ge=0)
    TimeOnSite: confloat(ge=0)
    Gender_Male: conint(ge=0, le=1)
    CampaignChannel_PPC: conint(ge=0, le=1)
    CampaignChannel_Referral: conint(ge=0, le=1)
    CampaignChannel_SEO: conint(ge=0, le=1)
    CampaignChannel_Social_Media: conint(ge=0, le=1) = Field(..., alias="CampaignChannel_Social Media")

# --- 5. POST /predict ---
@app.post("/predict")
async def predict(data: PredictionInput):
    global request_counter

    try:
        # DataFrame
        input_df = pd.DataFrame([data.dict(by_alias=True)])
        input_df['CTR_log'] = np.log1p(input_df['ClickThroughRate'])

        # Предикт
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]

        # Увеличиваем ID
        request_counter += 1

        # Запись с ID + timestamp
        record = {
            "id": request_counter,
            "timestamp": datetime.utcnow().isoformat(),
            "input": data.dict(),
            "prediction": int(prediction),
            "probability": round(float(probability), 4)
        }

        prediction_history.append(record)

        # Сохраняем в JSON
        with open(HISTORY_FILE, 'w') as f:
            json.dump(prediction_history, f, indent=4)

        return record

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- 6. GET /predictions (все запросы) ---
@app.get("/predictions")
async def get_all_predictions():
    return prediction_history

# --- 7. GET /predictions/{id} (поиск по ID) ---
@app.get("/predictions/{pred_id}")
async def get_prediction_by_id(pred_id: int):
    for record in prediction_history:
        if record["id"] == pred_id:
            return record
    raise HTTPException(status_code=404, detail="Запрос с таким ID не найден")


# --- 8. GET /download (скачать историю в JSON) ---
from fastapi.responses import FileResponse

@app.get("/download")
async def download_predictions():
    if not os.path.exists(HISTORY_FILE):
        raise HTTPException(status_code=404, detail="Файл истории не найден")
    return FileResponse(HISTORY_FILE, media_type="application/json", filename="predictions.json")