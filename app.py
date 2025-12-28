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

LEAGUE_IDS={
    "Premier League": 39,
    "La Liga":140,
    "Serie A":135,
    "Bundesiga":78,
    "Ligue 1": 61
}

#Loading the model 
MODEL_PATH="models/api_features/xgb.pkl"
SCALER_PATH="models/api_features/scaler.pkl"

try: 
    model=joblib.load(MODEL_PATH)
except Exception as e:
    print(f" Error loading model: {e}")
    model=None

try:
    scaler=joblib.load(SCALER_PATH) 
except Exception as e:
    print(f"Error loading scaler: {e}")
    #If scaler is not available, we use identity scaler
    class IdentityScaler(): 
        def transform(self, X):
            return X
    scaler=IdentityScaler()

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

def is_season_finished(season):
    return datetime.today()>get_season_end(season)

def get_season_for_fixtures():
    current_season=get_current_season()
    return current_season +1 if is_season_finished(current_season) else current_season

#Retrieve Upcoming matches
def get_upcoming_fixtures():    #prepares date range and season to request upcoming matches
    print("[DEBUG] Retrieving upcoming matches for the next 7 days...") #here we have done 7 days we can also do more
    matches=[]
    today=datetime.today()
    from_date=today.strftime('%Y-%m-%d')
    to_date=(today +timedelta(days=7)).strftime('%Y-%m-%d')
    current_season=get_season_for_fixtures()

    for league_name, league_id in LEAGUE_IDS.items():
        print(f"[DEBUG] {league_name} (ID={league_id}), season={current_season} from {from_date} to {to_date}")
        response=requests.get(
            url=f"{API_BASE_URL}/fixtures",
            headers=HEADERS,
            params={
                "league":league_id,
                "season":current_season,
                "from":from_date,
                "to":to_date,
                "status": "NS"  # Not Started | Filters by status "NS" (Not Started) to exclude ongoing or completed games.
            }
        ).json()
        if response.get("errors"):
            print(response["errors"])


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
    if response.get("errors"):
        print(response["errors"])

    print(f"[DEBUG] Raw response: {response.get('results',0)} results.")
    fixtures=[]
    if response and response.get('response'):
        for fix in response['response']:
            fixtures.append(fix['fixture']['id'])
    return fixtures

def get_fixture_statistics(fixture_id,team_id):
    url=f"{API_BASE_URL}/fixtures/statistics"
    params={"fixture":fixture_id,"team":team_id}
    response=requests.get(url,headers=HEADERS,params=params).json()
    if response.get("errors"):
        print(response["errors"])

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
        for stat in stats_list:
            stat_type=stat.get('type')
            value=stat.get('value')
            if value is None:
                continue
            if isinstance(value,str) and '%' in value:
                try:
                    value=float(value.replace("%",""))
                except Exception:
                    value=0.0
            else:
                try:
                    value=float(value) 
                except Exception:
                    value=0.0

            if stat_type in mapping:
                key=mapping[stat_type]
                if key=='possessionPct':
                    agg['possessionPct_sum']+=value
                    agg['count_possession']+=1
                else:
                      agg[key]+=value
    agg['possessionPct']=(agg["possessionPct_sum"]/agg['count_possession']) if agg['count_possession'] >0 else 50.0
    return agg

#Retrieve stats voa /teams/statistics

def get_team_goals(team_id,league_id,season):
    url=f"{API_BASE_URL}/teams/statistics"
    params={'team':team_id,'league':league_id,'season':season}
    response=requests.get(url,headers=HEADERS,params=params).json()
    if response.get("errors"):
        print(response["errors"])

    goals={'gf':0.0,"ga":0.0}
    if response and response.get('response'):
        resp=response['response']
        gf=resp.get('goals',{}).get('for',{}).get('total',{}).get('home',0)
        ga=resp.get('goals',{}).get('against',{}).get('total',{}).get('away',0)
        try:
            goals["gf"]=float(gf)
            goals['ga']=float(ga)
        except Exception:
            goals['gf']=0.0
            goals['ga']=0.0
    return goals

#Build feature vector for a match

def process_match_data(fixture_id,home_team_id,away_team_id,league_id,season):
    url=f"{API_BASE_URL}/fixtures"
    params={'id':fixture_id}
    fix_data=requests.get(url,headers=HEADERS,params=params).json()
    if fix_data.get("errors"):
        print(fix_data["errors"])

    if not fix_data.get('response'):
        print('[DEBUG] Fixture Non trouvee')
        return None
    fixture_date=fix_data['response'][0]['fixture']['date']
    fixture_date_str=fixture_date[:10]

    home_agg=agg_fixture_stats(home_team_id,league_id,season,fixture_date_str)
    away_agg=agg_fixture_stats(away_team_id,league_id,season,fixture_date_str)

    home_goals=get_team_goals(home_team_id,league_id,season)
    away_goals=get_team_goals(away_team_id,league_id,season)

    home_shotConversion=(home_agg['shotsOnTarget']/home_agg['totalShots']) if home_agg["totalShots"]>0 else 0.0
    away_shotConversion = away_agg["shotsOnTarget"] / away_agg["totalShots"] if away_agg["totalShots"] > 0 else 0.0

    features = {
    "leagueId": float(league_id),

    "home_possessionPct": home_agg.get("possessionPct", 50.0),
    "home_foulsCommitted": home_agg.get("foulsCommitted", 0.0),
    "home_yellowCards": home_agg.get("yellowCards", 0.0),
    "home_redCards": home_agg.get("redCards", 0.0),
    "home_wonCorners": home_agg.get("wonCorners", 0.0),
    "home_saves": home_agg.get("saves", 0.0),
    "home_totalShots": home_agg.get("totalShots", 0.0),
    "home_shotsOnTarget": home_agg.get("shotsOnTarget", 8.0),
    "home_accuratePasses": home_agg.get("accuratePasses", 0.0),
    "home_totalPasses": home_agg.get("totalPasses", 0.0),
    "home_blockedShots": home_agg.get("blockedShots", 0.0),

    "away_possessionPct": away_agg.get("possessionPct", 50.0),
    "away_foulsCommitted": away_agg.get("foulsCommitted", 0.0),
    "away_yellowCards": away_agg.get("yellowCards", 0.0),
    "away_redCards": away_agg.get("redCards", 0.0),
    "away_wonCorners": away_agg.get("wonCorners", 0.0),
    "away_saves": away_agg.get("saves", 0.0),
    "away_totalShots": away_agg.get("totalShots", 0.0),
    "away_shotsOnTarget": away_agg.get("shotsOnTarget", 0.0),
    "away_blockedShots": away_agg.get("blockedShots", 0.0),

    "home_gf": home_goals.get("gf", 0.0),
    "home_gd": home_goals.get("gf", 0.0) - home_goals.get("ga", 0.0),
    "away_ga": away_goals.get("ga", 0.0),
    "away_gd": away_goals.get("gf", 0.0) - away_goals.get("ga", 0.0),

    "home_shotConversion": home_shotConversion,
    "away_shotConversion": away_shotConversion
    }
    return features

#Score Prediction via model

def predict_match(fixture_id,home_team_id,away_team_id,league_id,season):
    if model is None:
        return {"error":"Model unavailable or not loaded"}
    features=process_match_data(fixture_id,home_team_id,away_team_id,league_id,season)
    if features is None:
        return {'error':"Insufficient data or fixture not found"}
    expected_features = [
    "leagueId",
    "home_possessionPct",
    "home_foulsCommitted",
    "home_yellowCards",
    "home_redCards",
    "home_wonCorners",
    "home_saves",
    "home_totalShots",
    "home_shotsOnTarget",
    "home_accuratePasses",
    "home_totalPasses",
    "home_blockedShots",
    "away_possessionPct",
    "away_foulsCommitted",
    "away_yellowCards",
    "away_redCards",
    "away_wonCorners",
    "away_saves",
    "away_totalShots",
    "away_shotsOnTarget",
    "away_blockedShots",
    "home_gf",
    "home_gd",
    "away_ga",
    "away_gd",
    "home_shotConversion",
    "away_shotConversion"
    ]

    ordered_features={key: features.get(key,0.0) for key in expected_features}
    df_features=pd.DataFrame([ordered_features])
    df_features[expected_features]=scaler.transform(df_features[expected_features])
    print('[DEBUG] Features sent to model: ',df_features.iloc[0].to_dict())

    prediction=model.predict(df_features)
    score_home=int(prediction[0][0])
    score_away=int(prediction[0][1])

    return {"predicted_score": f"{score_home} - {score_away}"}

#Flask routes

@app.route("/")
def index():
    #retrieve matches from the past week
    matches=get_upcoming_fixtures()
    return render_template("index.html",matches=matches)

@app.route("/predict",methods=['POST'])
def predict():
    data=request.json
    required_keys=['fixture_id','home_team_id','away_team_id','league_id','season']
    missing=[k for k in required_keys if k not in data]

    if missing:
        return jsonify({'error':f"The following keys are missing: {','.join(missing)} "}),400
    
    try: 
        season=int(data['season'])
    except ValueError:
        return jsonify({'error':'season must be an integer'}),400
    
    result=predict_match(
        data['fixture_id'],
        data['home_team_id'],
        data['away_team_id'],
        data['league_id'],
        season
    )
    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True)