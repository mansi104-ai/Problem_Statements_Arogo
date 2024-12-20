from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle
import os

# Load the trained model
model_path = "../models/optimized_decision_tree.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file was not found at {model_path}. Please check the path and try again.")

with open(model_path, "rb") as file:
    model = pickle.load(file)

# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2 templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint to render the prediction page
@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

# Predict delay endpoint
@app.post("/predict", response_class=HTMLResponse)
def predict_delay(
    request: Request,
    distance_km: float = Form(...),
    planned_delivery_days: int = Form(...),
    actual_delivery_days: int = Form(...),
    weather_conditions: int = Form(...),
    traffic_conditions: int = Form(...),
    is_long_distance: int = Form(...),
    is_bad_weather: int = Form(...),
    is_heavy_traffic: int = Form(...),
):
    # Convert input data to the required format
    input_features = np.array([
        distance_km,
        planned_delivery_days,
        actual_delivery_days,
        weather_conditions,
        traffic_conditions,
        is_long_distance,
        is_bad_weather,
        is_heavy_traffic,
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_features)
    delay_prediction = "Delayed" if prediction[0] == 1 else "On Time"

    # Render the result
    return templates.TemplateResponse(
        "predict.html",
        {"request": request, "result": delay_prediction}
    )
