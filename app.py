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

#Season management functions
def get_current_season():
    today=datetime.today()
    year=today.year
    #If we're before aug, the current season is the prev year
    return year -1 if today.month<8 else year

def get_season_end(season):
    #For a season starting in August, set the end to May 31 of the following year
    return datetime(season+1,month=5, day=31)

#Retrieve Upcoming matches
def get_upcoming_fixtures():
    print("[DEBUG] Retrieving upcoming matches for the next 7 days...")
    matches=[]
    today=datetime.today
    from_date=today.strftime('%Y-%m-%d')
    to_date=(today +timedelta(days=7)).strftime('%Y-%m-%d')
    current_season=get_current_season

    for league_name, league_id in LEAGUE_IDS.items():
        print(f"[DEBUG] {league_name} (ID={league_id}), season={current_season} from {from_date} to {to_date}")
        response=requests.get(
            url=f"{API_BASE_URL}/fixtures",
            headers=HEADERS,
            params={
                "league":league_id,
                "season":current_season,
                "from":from_date,
                "to":to_date
            }
        ).json()

        if response and response.get('response'):
            for fixture in response['response']:
                fixture_id=fixture["fixture"]["id"]
                #Extract only the forst 10 char
                date=fixture['fixture']['date'][:10]
                home_team=fixture['teams']['home']['name']
                away_team=fixture['teams']['away']['name']
                home_team_id=fixture['teams']['home']['id']
                away_team_id=fixture['teams']['away']['id']
                home_logo=fixture['teams']['home'].get("logo","")
                away_logo=fixture['teams']['away'].get("logo","")

                matches.append({
                    "league":league_name,
                    "league_id":league_id,
                    "fixture_id":fixture_id,
                    "date":date,
                    "home_team":home_team,
                    "away_team":away_team,
                    "home_team_id":home_team_id,
                    "away_team_id":away_team_id,
                    "home_logo":home_logo,
                    "away_logo":away_logo,
                    "season":current_season
                })
        
        else:
            print(f"[DEBUG] No matches for {league_name} (ID={league_id})")
    return matches


#Get fixtures played by a team from the start of the season up to a given date

def get_team_fixtures(team_id,league_id,season,fixture_date):
    url=f"{API_BASE_URL}/fixtures"
    season_start=f"{season}-08-01"
    params={
        "team":team_id,
        "league":league_id,
        "season":season,
        "from":season_start,
        "to":fixture_date,
        "status":"FT"   #only finished games
    }

    print(f"[DEBUG] GET request on {url} with params: {params}")
    response=requests.get(url,headers=HEADERS, params=params).json()
    print(f"[DEBUG] Raw response: {response.get('results',0)} results.")
    fixtures=[]
    if response and response.get('response'):
        for fix in response['response']:
            fixtures.append(fix['feature']['id'])
    return fixtures

def get_fixture_statistics(fixture_id,team_id):
    url=f"{API_BASE_URL}/fixtures/statistics"
    params={"fixture":fixture_id,"team":team_id}
    response=requests.get(url,headers=HEADERS,params=params)
    if response and response.get("response") and len(response['response'])>0:
        return response['response'][0].get('statistics',[])
    return []

#aggregate stats match by match (via/fixtures/statistics)

def agg_fixture_stats(team_id,league_id,season,fixture_date):
    mapping={
    "Ball Possession": "possessionPct",
    "Fouls": "foulsCommitted",
    "Yellow Cards": "yellowCards",
    "Red Cards": "redCards",
    "Corner Kicks": "wonCorners",
    "Goalkeeper Saves": "saves",
    "Total Shots": "totalShots",
    "Shots on Goal": "shotsOnTarget",
    "Blocked Shots": "blockedShots",
    "Total passes": "totalPasses",
    "Passes accurate": "accuratePasses"
    }
    agg={
    "possessionPct_sum": 0.0,
    "count_possession": 0,
    "foulsCommitted": 0,
    "yellowCards": 0,
    "redCards": 0,
    "wonCorners": 0,
    "saves": 0,
    "totalShots": 0,
    "shotsOnTarget": 0,
    "blockedShots": 0,
    "totalPasses": 0,
    "accuratePasses": 0     
    }

    fixtures=get_team_fixtures(team_id,league_id,season,fixture_date)
    if not fixtures:
        print(f"[DEBUG] No fixture found for team {team_id} upto {fixture_date}.")
        return agg

    for fix_id in fixtures:
        stats_list=get_fixture_statistics(fix_id,team_id)
        