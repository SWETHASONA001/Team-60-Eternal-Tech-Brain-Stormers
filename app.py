from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pred import predict_crops  # Import your function from pred.py

# Initialize FastAPI app
app = FastAPI(title="AI Crop Planner API", version="1.0")

# Enable CORS so frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to ["http://localhost:5173"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema
class CropRequest(BaseModel):
    state: str
    district: str
    area_ha: float
    date_str: str = "2025-09-18"  # default if not passed

# Define root route
@app.get("/")
def root():
    return {"message": "AI Crop Planner API is running 🚀"}

# Prediction endpoint
@app.post("/predict")
def get_prediction(req: CropRequest):
    try:
        results = predict_crops(
            state=req.state,
            district=req.district,
            area_ha=req.area_ha,
            date_str=req.date_str,
        )
        return {"recommendations": results}
    except Exception as e:
        return {"error": str(e)}