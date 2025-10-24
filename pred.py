import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTM Model Definition
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Load saved models and scalers
rf_model = joblib.load('rf_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')
lstm_model = LSTMForecaster(3, 64, 3).to(device)
lstm_model.load_state_dict(torch.load('lstm_model.pth', map_location=torch.device('cpu')))

lstm_model.eval()
scaler = joblib.load('scaler.pkl')
ts_scaler = joblib.load('ts_scaler.pkl')
label_encoder_crop = joblib.load('label_encoder_crop.pkl')
label_encoder_state = joblib.load('label_encoder_state.pkl')
label_encoder_district = joblib.load('label_encoder_district.pkl')
label_encoder_season = joblib.load('label_encoder_season.pkl')
label_encoder_soil = joblib.load('label_encoder_soil.pkl')
label_encoder_irrigation = joblib.load('label_encoder_irrigation.pkl')

# Prediction Function
def predict_crops(state, district, area_ha, date_str='2025-09-18'):
    # Load and preprocess data
    weather_df = pd.read_csv('synthetic_weather.csv')
    soil_df = pd.read_csv('synthetic_soil_conditions.csv')
    economic_df = pd.read_csv('synthetic_economic_indicators.csv')
    production_df = pd.read_csv('synthetic_crop_production_yield.csv')
    market_df = pd.read_csv('synthetic_crop_market_prices.csv')

    # Preprocess dates
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    soil_df['Date'] = pd.to_datetime(soil_df['Date'], dayfirst=True)
    economic_df['Date'] = pd.to_datetime(economic_df['Date'])
    production_df['Date'] = pd.to_datetime(production_df['Date'])
    market_df['Date'] = pd.to_datetime(market_df['Date'], format='mixed', dayfirst=True)

    # Merge datasets
    prod_market_df = production_df.merge(market_df, on=['Date', 'State', 'District'], how='outer', suffixes=('_prod', '_market'))
    merged_df = weather_df.merge(soil_df, on=['Date', 'State', 'District'], how='left') \
                          .merge(economic_df, on=['Date', 'State'], how='left') \
                          .merge(prod_market_df, on=['Date', 'State', 'District'], how='left')

    # Handle 'Crop' column
    if 'Crop_prod' in merged_df.columns:
        merged_df['Crop'] = merged_df['Crop_prod'].fillna(merged_df.get('Crop_market', 'Unknown'))
    elif 'Crop_market' in merged_df.columns:
        merged_df['Crop'] = merged_df['Crop_market']
    else:
        merged_df['Crop'] = 'Unknown'

    # Handle missing values
    merged_df = merged_df.fillna(merged_df.mean(numeric_only=True))
    merged_df['Crop'] = merged_df['Crop'].fillna('Unknown')

    # Encode categoricals
    merged_df['Crop_Encoded'] = label_encoder_crop.transform(merged_df['Crop'])
    merged_df['State_Encoded'] = label_encoder_state.transform(merged_df['State'])
    merged_df['District_Encoded'] = label_encoder_district.transform(merged_df['District'])
    merged_df['Season_Encoded'] = label_encoder_season.transform(merged_df['Season'].fillna('Unknown'))
    merged_df['Soil_Type_Encoded'] = label_encoder_soil.transform(merged_df['Soil_Type'].fillna('Unknown'))
    merged_df['Irrigation_Type_Encoded'] = label_encoder_irrigation.transform(merged_df['Irrigation_Type'].fillna('Unknown'))

    # Feature Engineering
    merged_df['Year'] = merged_df['Date'].dt.year
    merged_df['Month'] = merged_df['Date'].dt.month
    merged_df['Rainfall_Lag1'] = merged_df.groupby(['State', 'District'])['Rainfall'].shift(1).fillna(0)
    merged_df['Price_Lag1'] = merged_df.groupby(['Crop'])['Modal_Price'].shift(1).fillna(0)
    merged_df['Yield_Lag1'] = merged_df.groupby(['Crop'])['Yield_kg_ha'].shift(1).fillna(0)
    merged_df['Soil_Productivity'] = merged_df['N'] * merged_df['P'] * merged_df['K'] / 1000
    merged_df['Weather_Suitability'] = merged_df['Avg_Temp'] * merged_df['Rainfall'] / 100

    # Features list
    available_features = [col for col in ['Min_Temp', 'Max_Temp', 'Avg_Temp', 'Rainfall', 'Humidity', 'Wind_Speed', 
                                         'Solar_Radiation', 'pH', 'EC', 'N', 'P', 'K', 'OC', 'CPI', 'WPI', 
                                         'Crude_Oil_Price', 'Exchange_Rate', 'GDP_Growth', 'Imports_tons', 'Exports_tons', 
                                         'Soil_Productivity', 'Weather_Suitability', 'Fertilizer_Use', 'Area_ha', 
                                         'Modal_Price', 'Min_Price', 'Max_Price', 'Arrivals', 'Year', 'Month', 
                                         'Rainfall_Lag1', 'Price_Lag1', 'Yield_Lag1', 'State_Encoded', 'District_Encoded', 
                                         'Season_Encoded', 'Soil_Type_Encoded', 'Irrigation_Type_Encoded'] if col in merged_df.columns]

    # Filter data for prediction
    date = pd.to_datetime(date_str)
    filter_df = merged_df[(merged_df['State'] == state) & (merged_df['District'] == district) & (merged_df['Date'] <= date)]
    if filter_df.empty:
        return "No data for this location."

    # Override Area_ha with user input
    filter_df['Area_ha'] = area_ha

    # Prepare input data
    input_data = filter_df[available_features]
    input_scaled = scaler.transform(input_data)

    # LSTM Forecast
    ts_cols = ['Modal_Price', 'Rainfall', 'Avg_Temp']
    with torch.no_grad():
        recent_ts = filter_df[ts_cols].tail(5).values.astype(float)  # seq_length=5
        recent_ts_scaled = ts_scaler.transform(recent_ts)
        input_tensor = torch.tensor(recent_ts_scaled).float().unsqueeze(0).to(device)
        forecast_scaled = lstm_model(input_tensor).cpu().numpy()
        forecast = ts_scaler.inverse_transform(forecast_scaled)

    # RF and XGBoost Predictions
    suit_pred = rf_model.predict(input_scaled)
    yield_pred = xgb_model.predict(input_scaled)

    # Create recommendation DataFrame
    rec_df = filter_df.copy()
    rec_df['Suitability'] = suit_pred
    rec_df['Predicted_Yield'] = yield_pred
    rec_df['Total_Predicted_Yield'] = rec_df['Predicted_Yield'] * area_ha
    rec_df['Forecast_Price'] = forecast[0, 0] if 'Modal_Price' in ts_cols else rec_df['Modal_Price'].mean()
    rec_df['Profitability'] = rec_df['Total_Predicted_Yield'] * rec_df['Forecast_Price']

    # Generate recommendations
    top_crops = rec_df[rec_df['Suitability'] == 1].sort_values('Profitability', ascending=False).head(5)
    recommendations = []
    for _, row in top_crops.iterrows():
        planting_time = f"Month {row['Month']}, Season: {row.get('Season', 'N/A')}"
        rec = f"Crop: {row['Crop']}, Planting Time: {planting_time}, Predicted Yield per ha: {row['Predicted_Yield']:.2f} kg/ha, Total Predicted Yield: {row['Total_Predicted_Yield']:.2f} kg, Profitability: {row['Profitability']:.2f}"
        recommendations.append(rec)

    return recommendations

# Example prediction
state = 'Kerala'
district = 'Ernakulam'
area_ha = 10.0
date_str = '2026-09-18'
predictions = predict_crops(state, district, area_ha, date_str)
print(f"Predictions for {state}, {district} with {area_ha} ha on {date_str}:")
for pred in predictions:
    print(pred)