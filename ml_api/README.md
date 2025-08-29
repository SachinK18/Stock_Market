# ML API for Stock Prediction

This is the Python Flask API that provides machine learning-based stock predictions using Logistic Regression and Random Forest algorithms.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Install Dependencies**
   ```bash
   cd ml_api
   pip install -r requirements.txt
   ```

2. **Start the API Server**
   ```bash
   python app.py
   ```
   
   Or on Windows, you can double-click `start_ml_api.bat`

3. **API will be available at:** `http://localhost:5002`

## API Endpoints

### POST /api/predict
Predicts stock movement for a given ticker.

**Request Body:**
```json
{
  "ticker": "INFIBEAM.BO"
}
```

**Response:**
```json
{
  "ticker": "INFIBEAM.BO",
  "prediction": {
    "logistic_prediction": "UP",
    "random_forest_prediction": "UP",
    "final_prediction": "UP",
    "confidence": "High"
  },
  "model_accuracies": {
    "logistic_regression": 0.5234,
    "random_forest": 0.5678
  },
  "current_price": 25.45,
  "last_updated": "2025-08-18"
}
```

### GET /api/stocks
Returns list of available stocks.

### GET /health
Health check endpoint.

## Algorithm Logic

The prediction logic works as follows:
- **Both models predict UP** → Final prediction: **UP** (High confidence)
- **Both models predict DOWN** → Final prediction: **DOWN** (High confidence)  
- **Models disagree** → Final prediction: **SLIGHTLY UP** (Medium confidence)

## Features

- Real-time stock data from Yahoo Finance
- Technical indicators: RSI, Moving Averages, Volatility
- Dual model approach for better accuracy
- Feature engineering with price ratios and returns
- Model accuracy reporting
