from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from prophet import Prophet

app = FastAPI()

# Load the models and preprocessing tools
gpr_model = joblib.load('best_gpr_model.pkl')
poly_transformer = joblib.load('poly_transformer.pkl')
scaler = joblib.load('scaler.pkl')

# Load the Prophet model
df_prophet = pd.read_csv('kc_house_data.csv', encoding='ISO-8859-1')
df_prophet['date'] = pd.to_datetime(df_prophet['date'])
df_prophet = df_prophet[['date', 'price']].dropna()
df_prophet = df_prophet.rename(columns={'date': 'ds', 'price': 'y'})
prophet_model = Prophet()
prophet_model.fit(df_prophet)

# Pydantic model for input validation
class GPRInput(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int
    year: int
    month: int
    day: int

class ProphetInput(BaseModel):
    periods: int

@app.post("/predict/gpr/")
async def predict_gpr(data: GPRInput):
    try:
        # Transform and scale the input features
        inputs = [data.bedrooms, data.bathrooms, data.sqft_living, data.sqft_lot, data.floors, data.waterfront, 
                  data.view, data.condition, data.grade, data.sqft_above, data.sqft_basement, data.yr_built, 
                  data.yr_renovated, data.zipcode, data.lat, data.long, data.sqft_living15, data.sqft_lot15, 
                  data.year, data.month, data.day]
        inputs_poly = poly_transformer.transform([inputs])
        inputs_scaled = scaler.transform(inputs_poly)
        prediction = gpr_model.predict(inputs_scaled)
        return {"predicted_price": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/prophet/")
async def predict_prophet(data: ProphetInput):
    try:
        periods = data.periods
        future_dates = prophet_model.make_future_dataframe(periods=periods)
        forecast = prophet_model.predict(future_dates)
        # Include the entire forecast data for plotting
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
