import os

# Step 30C - Create the Manufacturing Shortage API



from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field

# Step 31B - Load the trained supplier lead-time model


from pathlib import Path
import joblib

# Find the model in the main GitHub project folder
model_path = Path(__file__).resolve().parent.parent / "supplier_lead_time_model_500.joblib"

# Load the trained Linear Regression model
model = joblib.load(model_path)

print("Supplier lead-time model loaded successfully!")

# Step 35B - Identify the trained model version
MODEL_VERSION = "supplier-lead-time-500-v1"


# Create the API application
app = FastAPI(
    title="AI Manufacturing Shortage Agent",
    version="1.0"
)

# Create a simple test endpoint
@app.get("/")
def home():
    return {
        "message": "AI Manufacturing Shortage Agent API is running!"
    }


# Step 32B - Define the supplier information

class SupplierInput(BaseModel):
    Order_Quantity: int = Field(gt=0)
    Supplier_History_Days: float = Field(ge=0)
    Transportation_Days: float = Field(ge=0)
    Supplier_Reliability: float = Field(ge=0, le=100)
    Weather_Delay_Days: float = Field(ge=0)

# Step 32D - Predict supplier delivery time

import pandas as pd

import secrets
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)

def verify_api_key(
    provided_key: str = Security(api_key_header)
):
    expected_key = os.getenv("MANUFACTURING_API_KEY")

    # Local development fallback
    if not expected_key:
        key_path = (
            Path(__file__).resolve().parent.parent
            / "security"
            / ".manufacturing_api_key"
        )

        if key_path.is_file():
            expected_key = key_path.read_text(
                encoding="utf-8"
            ).strip()

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured"
        )

    if (
        not provided_key
        or not secrets.compare_digest(
            provided_key,
            expected_key
        )
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key"
        )

    return True

@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict_delivery(supplier: SupplierInput):

    # Organize the five inputs for our model
    input_data = pd.DataFrame([supplier.model_dump()])

    # Predict delivery time
    prediction = model.predict(input_data)[0]

    return {
    "predicted_lead_time_days": round(float(prediction), 2),
    "model_version": MODEL_VERSION
}


# Step 33 - Add production risk and recommended actions

@app.post("/predict-risk", dependencies=[Depends(verify_api_key)])
def predict_production_risk(supplier: SupplierInput):
    # Organize the five supplier features
    input_data = pd.DataFrame([supplier.model_dump()])

    # Predict delivery time using the trained model
    predicted_days = float(model.predict(input_data)[0])

    # Assume materials are needed within 30 days
    days_until_needed = 30
    delay_days = max(0, predicted_days - days_until_needed)

    # Classify risk and recommend an action
    if delay_days == 0:
        risk_level = "LOW"
        recommended_action = "Continue monitoring the order"
    elif delay_days <= 5:
        risk_level = "MEDIUM"
        recommended_action = "Contact supplier to confirm delivery"
    else:
        risk_level = "HIGH"
        recommended_action = "Escalate risk and review alternate suppliers or expedite shipping"

    return {
        "predicted_lead_time_days": round(predicted_days, 2),
        "days_until_needed": days_until_needed,
        "predicted_delay_days": round(delay_days, 2),
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "model_version": MODEL_VERSION
 }




# Late-delivery risk prediction endpoint

risk_model_path = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "manufacturing_risk_model.joblib"
)

risk_artifact = joblib.load(risk_model_path)

class LateDeliveryInput(BaseModel):
    order_item_quantity: int = Field(ge=0)
    shipping_mode: str
    order_region: str
    scheduled_shipping_days: int = Field(ge=0)

@app.post("/predict-late-delivery", dependencies=[Depends(verify_api_key)])
def predict_late_delivery(order: LateDeliveryInput):
    input_data = pd.DataFrame([{
        "Order Item Quantity": order.order_item_quantity,
        "Shipping Mode": order.shipping_mode,
        "Order Region": order.order_region,
        "Days for shipment (scheduled)": order.scheduled_shipping_days
    }])

    probability = float(
        risk_artifact["model"].predict_proba(input_data)[0, 1]
    )

    threshold = risk_artifact["threshold"]

    return {
        "risk_probability": round(probability, 4),
        "risk_percentage": round(probability * 100, 2),
        "risk_level": (
            "HIGH RISK" if probability >= threshold
            else "LOWER RISK"
        ),
        "threshold": threshold
    }
