# Football-Score-Predictor

A machine learning powered web application that predicts football match scores using historical data and live API statistics.

## OVERVIEW:

This project is an end-to-end score prediction system designed specifically for football/soccer matches.
It uses a trained machine learning model to predict the final score of upcoming matches by combining historical match data with live team statistics fetched from the API-Football service.

The application is built using:
1. Python & Flask (backend)
2. Machine Learning (XGBoost)
3. API-Football (live football data)
4. HTML/CSS/JavaScript (frontend)

## FEATURES:

1. Displays upcoming football fixtures across major European leagues
2. Fetches real-time team statistics using API-Football
3. Predicts match scores using a trained ML model
4. Interactive Web interface with live predictions

## TECH STACK

**Backend**
- Python
- Flask
- Pandas
- Scikit-learn
- XGBoost

**Frontend**

- HTML
- CSS
- JavaScript(Fetch API)

**Data and APIs**
- API-Football (fixtures, statistics)
- ESPN Soccer Data - kaggle (model training) 

## System Architecture

1. Upcoming fixtures are fetched from API-Football.
2. When the user clicks "Predict Score":
   - The frontend sends match details to the Flask backend.
3. The backend:
   - Fetches historical team statistics
   - Aggregates and processes features
   - Applies scaling
   - Passes data to the trained ML model
4. The predicted score is returned to the frontend and displayed in real time.

## Machine Learning Model

- Model: XGBoost Regressor
- Target: Predict home and away team goals
- Features include:
  - Possession percentage
  - Shots, shots on target
  - Fouls, cards, corners
  - Goals for/against
  - Shot conversion rate
  - Home/Away context

The model was trained on historical match data sourced from Kaggle.


## Installation & Setup

1. Clone the Repository
2. Create a virtual environment
3. Install Required Libraries mentioned in requirements.txt
    - pip install -r requirements.txt
4. Create a .env file and save your API Key
    - API_FOOTBALL_KEY=your_api_key_here
5. Run the application 
    - MainNb.ipynb
    - python app.py
6. Open the browser and go to:
    -http://localhost:5000


## API Key Notice

This project requires an API-Football key. For security reasons, the API key is stored in a .env file and is not included in the repository.

To run the project locally, you must obtain your own API-Football key.
 
A paid subscription of the API-Football is required to access the complete data


## Limitations

- Predictions are based on historical and aggregated statistics and do not account for:
  - Injuries
  - Lineups
  - Tactical changes
  - Weather conditions
- Football outcomes are inherently unpredictable.
- The project is intended for educational and analytical purposes, not betting.

## Demo

A short demo video showing the application running locally is available.
- https://youtu.be/Hn3yNGwso1s?si=9gmqNxJ4NEu6G1Fs


## Future Improvements / Scope of improvement

- Adding win/draw/loss probabilities
- Incorporate player-level statistics
- Improve feature engineering

## Author


**Ashwyn Kumar**  

**B.Tech** Undergraduate at Netaji Subhas University of Technology **(NSUT)**, Delhi
- Linkedin: www.linkedin.com/in/ashwyn-kumar-948162359
- Github: Ashwynkumar16
