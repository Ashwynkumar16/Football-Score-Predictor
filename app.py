import requests
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()
#Api configuration
API_KEY=os.getenv("API_FOOTBALL_KEY")

if not API_KEY:
    raise RuntimeError("API_FOOTBALL_KEY is not set")

API_BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {
    "x-apisports-key": API_KEY
}
#Temp Debugging
# response = requests.get(API_BASE_URL, headers=HEADERS)
# print(response.status_code)
# print(response.json())
# print("KEY LENGTH:", len(API_KEY))
# print("KEY PREVIEW:", API_KEY[:4], "****")

LEAGUE_IDS={
    "Premier League": 39,
    "La Liga":140,
    "Serie A":135,
    "Bundeliga":78,
    "Ligue 1": 61
}

#Loading the model 
MODEL_PATH="models/api_features/xgb.pkl"
SCALER_PATH="models/api_features/scaler.pkl"

try: 
    model=joblib.load(MODEL_PATH)
except Exception as e:
    print(f" Error loading model: {0}")
    model=None

try:
    scaler=joblib.load(SCALER_PATH)
except Exception as e:
    print(f"Error loading scaler: {0}")
    #If scaler is not available, we use identity scaler
    class IdentityScaler: 
        def transform(self, X):
            return X
    scaler=IdentityScaler

#Flask Initialisation
app=Flask(__name__)