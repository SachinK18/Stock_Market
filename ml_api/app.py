from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import os
from datetime import datetime
import warnings
import json
import re
import ollama
import requests
import base64

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ==================================================================
# == CHATBOT CODE
# ==================================================================

MODEL_NAME = "llama3"
OLLAMA_OPTIONS = {"temperature": 0.0}

def ask_ollama_for_intent(user_question: str):
    """Ask Ollama to interpret user query into a structured JSON intent."""
    system_prompt = (
        "You are a financial query interpreter. "
        "Given a user's stock market question, respond ONLY in JSON with fields:\n"
        "{"
        "\"intent\": one of [\"price\", \"performance\", \"compare\", \"info\", \"why\"], "
        "\"tickers\": [list of tickers or company names], "
        "\"period\": e.g. \"1d\", \"5d\", \"1mo\", \"3mo\", \"6mo\", \"1y\", "
        "\"metrics\": [list of metrics like close, volume, market_cap, returns, high, low], "
        "\"date\": date if user specifies a particular day (YYYY-MM-DD)"
        "}\n\n"
        "Examples:\n"
        "User: 'What is Apple's stock price today?'\n"
        "→ {\"intent\":\"price\",\"tickers\":[\"AAPL\"],\"period\":\"1d\",\"metrics\":[\"close\"]}\n\n"
        "User: 'Compare Amazon and Google returns over 3 months.'\n"
        "→ {\"intent\":\"compare\",\"tickers\":[\"AMZN\",\"GOOG\"],\"period\":\"3mo\",\"metrics\":[\"returns\"]}\n\n"
        "User: 'How did Microsoft perform this week?'\n"
        "→ {\"intent\":\"performance\",\"tickers\":[\"MSFT\"],\"period\":\"5d\",\"metrics\":[\"close\",\"returns\"]}\n\n"
        "User: 'What affects Tesla’s stock price?'\n"
        "→ {\"intent\":\"why\",\"tickers\":[\"TSLA\"],\"period\":\"1mo\",\"metrics\":[]}\n"
        "Return ONLY JSON, no explanations."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_question}]
    resp = ollama.chat(model=MODEL_NAME, messages=messages, options=OLLAMA_OPTIONS)
    content = resp.get("message", {}).get("content", str(resp))
    try:
        return json.loads(content)
    except Exception:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try: return json.loads(match.group())
            except Exception: pass
    return {"intent": "price", "tickers": [], "period": "1d", "metrics": ["close"], "date": None}

def get_price_by_date(ticker: str, date_str: str):
    """Fetch price of a stock on a specific date."""
    try:
        t = yf.Ticker(ticker)
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        hist = t.history(period="1y")
        if hist.empty: return f"❌ No data found for {ticker}."
        hist.index = hist.index.date
        if target_date in hist.index:
            price = hist.loc[target_date]["Close"]
            return f"💰 {ticker} closing price on {target_date}: {price:.2f}"
        else:
            prev_dates = [d for d in hist.index if d < target_date]
            if prev_dates:
                last_trading = max(prev_dates)
                price = hist.loc[last_trading]["Close"]
                return f"ℹ️ Market was closed on {target_date}. The close on the last trading day ({last_trading}) was: {price:.2f}"
            else:
                return f"❌ No trading data found before {target_date}."
    except Exception as e:
        return f"⚠️ Error fetching price for {ticker}: {e}"

def get_recent_price(ticker: str):
    t = yf.Ticker(ticker)
    hist = t.history(period="1d")
    if hist.empty: return f"❌ No data found for {ticker}."
    return f"💰 {ticker} current close price: {hist['Close'].iloc[-1]:.2f}"

def get_performance(ticker, period="1mo"):
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty: return {"ticker": ticker, "message": "No data"}
    start_price, end_price = hist["Close"].iloc[0], hist["Close"].iloc[-1]
    pct_change = ((end_price - start_price) / start_price) * 100
    return {"ticker": ticker, "start_price": round(start_price, 2), "end_price": round(end_price, 2), "change_pct": round(pct_change, 2)}

def compare_returns(tickers, period="3mo"):
    results = [perf for tk in tickers if "change_pct" in (perf := get_performance(tk, period))]
    if not results: return "No data for comparison."
    best = max(results, key=lambda x: x["change_pct"])
    comparison = " | ".join([f"{p['ticker']}: {p['change_pct']:.2f}%" for p in results])
    return f"📈 Returns over {period}: {comparison}. Best performer: {best['ticker']}."

def get_basic_info(ticker):
    info = yf.Ticker(ticker).info
    return {"ticker": ticker, "market_cap": info.get("marketCap", "N/A"), "volume": info.get("volume", "N/A"), "previous_close": info.get("previousClose", "N/A")}

def handle_question(user_question: str):
    parsed = ask_ollama_for_intent(user_question)
    intent, tickers, period, date = parsed.get("intent", "price"), [t.strip().upper() for t in parsed.get("tickers", []) if t.strip()], parsed.get("period", "1mo"), parsed.get("date")

    if not tickers: return "❌ No ticker symbols found. Try using stock symbols like AAPL, MSFT, TSLA, etc."

    if intent == "price":
        return get_price_by_date(tickers[0], date) if date else get_recent_price(tickers[0])
    elif intent == "performance":
        perf = get_performance(tickers[0], period)
        return perf["message"] if "message" in perf else f"📊 {tickers[0]} performance over {period}:\nStart: {perf['start_price']} → End: {perf['end_price']} ({perf['change_pct']}%)"
    elif intent == "compare":
        return "❌ Need at least two tickers to compare." if len(tickers) < 2 else compare_returns(tickers, period)
    elif intent == "info":
        info = get_basic_info(tickers[0])
        market_cap_str = f"{info['market_cap']:,}" if isinstance(info['market_cap'], (int, float)) else info['market_cap']
        volume_str = f"{info['volume']:,}" if isinstance(info['volume'], (int, float)) else info['volume']
        return f"ℹ️ {tickers[0]} | Market Cap: {market_cap_str} | Volume: {volume_str} | Previous Close: {info['previous_close']}"
    elif intent == "why":
        resp = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": f"Explain briefly what factors influence {tickers[0]}'s stock price. Use general economic and company factors, not speculative advice."}])
        return f"🧠 {tickers[0]} stock factors:\n{resp['message']['content']}"
    else:
        return "❌ Unknown intent. Try asking about price, performance, or comparison."

# ==================================================================
# == STOCK PREDICTION CODE
# ==================================================================
class StockPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.log_model = LogisticRegression(max_iter=1000)
        self.rf_model = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=3, random_state=42)
        self.features = ["Daily_Return", "MA_5", "MA_20", "MA_50", "Price_MA5_ratio", "Price_MA20_ratio", "RSI", "Volatility_20"]

    def prepare_data(self, ticker, end_date):
        data = yf.download(ticker, start="2010-01-01", end=end_date)
        if data.empty: return None, "No data found for ticker"
        if isinstance(data.columns, pd.MultiIndex): data.columns = [col[0] for col in data.columns]
        data = data.dropna()
        if len(data) < 100: return None, "Insufficient data for prediction"
        
        data["Daily_Return"] = data["Close"].pct_change()
        data["MA_5"] = data["Close"].rolling(window=5).mean()
        data["MA_20"] = data["Close"].rolling(window=20).mean()
        data["MA_50"] = data["Close"].rolling(window=50).mean()
        data["Price_MA5_ratio"] = data["Close"] / data["MA_5"]
        data["Price_MA20_ratio"] = data["Close"] / data["MA_20"]
        delta = data["Close"].diff()
        gain, loss = np.where(delta > 0, delta, 0), np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain, index=data.index).rolling(14).mean()
        avg_loss = pd.Series(loss, index=data.index).rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        data["RSI"] = 100 - (100 / (1 + rs))
        data["Volatility_20"] = data["Daily_Return"].rolling(window=20).std()
        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
        return data.dropna(), None

    def train_models(self, data):
        X, y = data[self.features], data["Target"]
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.log_model.fit(X_train_scaled, y_train)
        self.rf_model.fit(X_train, y_train)
        return {'log_accuracy': accuracy_score(y_test, self.log_model.predict(X_test_scaled)), 'rf_accuracy': accuracy_score(y_test, self.rf_model.predict(X_test)), 'latest_features': X.iloc[-1].values}, None

    def predict(self, latest_features):
        features_reshaped = latest_features.reshape(1, -1)
        log_pred = self.log_model.predict(self.scaler.transform(features_reshaped))[0]
        rf_pred = self.rf_model.predict(features_reshaped)[0]
        final_prediction, confidence = ("UP", "High") if log_pred and rf_pred else (("DOWN", "High") if not log_pred and not rf_pred else ("SLIGHTLY UP - RISKY", "Medium"))
        return {'logistic_prediction': "UP" if log_pred else "DOWN", 'random_forest_prediction': "UP" if rf_pred else "DOWN", 'final_prediction': final_prediction, 'confidence': confidence}, None

predictor = StockPredictor()

# ==================================================================
# == FLASK API ROUTES
# ==================================================================

@app.route('/api/predict', methods=['POST'])
def predict_stock():
    try:
        req_data = request.get_json()
        ticker, end_date = req_data.get('ticker'), req_data.get('end_date')
        stock_data, error = predictor.prepare_data(ticker, end_date)
        if error: return jsonify({'error': error}), 400
        training_result, error = predictor.train_models(stock_data)
        if error: return jsonify({'error': str(error)}), 500
        prediction_result, error = predictor.predict(training_result['latest_features'])
        if error: return jsonify({'error': str(error)}), 500
        
        return jsonify({
            'ticker': ticker, 'end_date': end_date, 'prediction': prediction_result,
            'model_accuracies': {'logistic_regression': round(training_result['log_accuracy'], 4), 'random_forest': round(training_result['rf_accuracy'], 4)},
            'current_price': float(stock_data['Close'].iloc[-1]),
            'last_updated': stock_data.index[-1].strftime('%Y-%m-%d'),
            'data_points_used': len(stock_data)
        })
    except Exception as e:
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/api/chatbot', methods=['POST'])
def ask_chatbot():
    try:
        question = request.get_json().get('question')
        if not question: return jsonify({'error': 'No question provided'}), 400
        return jsonify({'answer': handle_question(question)})
    except Exception as e:
        if "Failed to connect" in str(e): return jsonify({'answer': "⚠️ Could not connect to Ollama. Please ensure the Ollama service is running locally."}), 500
        return jsonify({'error': str(e)}), 500

@app.route('/api/search-stocks', methods=['GET'])
def search_stocks():
    """Search for stocks dynamically using Yahoo Finance's real search API."""
    try:
        query = request.args.get('q', '').strip()
        if not query or len(query) < 2: return jsonify({'stocks': []})

        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=20&newsCount=0"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get('quotes', []):
            if item.get('quoteType') == 'EQUITY' and item.get('symbol'):
                exchange = item.get('exchange')
                results.append({
                    'symbol': item.get('symbol'),
                    'name': item.get('shortname', item.get('longname', 'N/A')),
                    'exchange': exchange,
                    'country': get_country_from_exchange(exchange),
                })
        return jsonify({'stocks': results})
    except requests.exceptions.RequestException as e:
        print(f"Yahoo search API request failed: {e}")
        return jsonify({'error': 'Failed to connect to the stock search service.'}), 500
    except Exception as e:
        print(f"An unexpected error occurred in search_stocks: {e}")
        return jsonify({'error': 'An internal error occurred while searching for stocks.'}), 500

@app.route('/api/chart-data', methods=['POST'])
def get_chart_data():
    try:
        data = request.get_json()
        ticker = data.get('ticker')
        end_date = data.get('end_date')

        stock_data, error = predictor.prepare_data(ticker, end_date)
        if error:
            return jsonify({'error': error}), 400

        latest_data = stock_data.iloc[-1]
        current_rsi = stock_data["RSI"].iloc[-1]
        chart_data_points = stock_data.tail(100)
        
        return jsonify({
            'ticker': ticker,
            'current_price': float(latest_data['Close']),
            'current_rsi': float(current_rsi),
            'date_range': {
                'start': stock_data.index[0].strftime('%Y-%m-%d'),
                'end': stock_data.index[-1].strftime('%Y-%m-%d')
            },
            'latest_data': {
                'open': float(latest_data['Open']),
                'high': float(latest_data['High']),
                'low': float(latest_data['Low']),
                'close': float(latest_data['Close']),
                'volume': float(latest_data['Volume']) if 'Volume' in latest_data else 0
            },
            'chart_data': {
                'labels': [date.strftime('%Y-%m-%d') for date in chart_data_points.index],
                'close_prices': chart_data_points['Close'].tolist(),
                'open_prices': chart_data_points['Open'].tolist(),
                'volumes': chart_data_points['Volume'].tolist() if 'Volume' in chart_data_points.columns else [],
                'ma_5': chart_data_points['MA_5'].dropna().tolist(),
                'ma_20': chart_data_points['MA_20'].dropna().tolist(),
                'ma_50': chart_data_points['MA_50'].dropna().tolist(),
                'rsi': chart_data_points['RSI'].dropna().tolist()
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/heatmap-image', methods=['POST'])
def get_heatmap_image():
    try:
        data = request.get_json()
        ticker, end_date = data.get('ticker'), data.get('end_date')
        stock_data, error = predictor.prepare_data(ticker, end_date)
        if error: return jsonify({'error': error}), 400

        heatmap_buffer = generate_chart_image(stock_data, 'correlation', ticker)
        img_base64 = base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')
        return jsonify({'image': img_base64, 'ticker': ticker})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-pdf-report', methods=['POST'])
def generate_pdf_report():
    try:
        data = request.get_json()
        ticker, end_date = data.get('ticker'), data.get('end_date')
        stock_data, error = predictor.prepare_data(ticker, end_date)
        if error: return jsonify({'error': error}), 400
        training_result, error = predictor.train_models(stock_data)
        if error: return jsonify({'error': str(error)}), 500
        prediction_result, error = predictor.predict(training_result['latest_features'])
        if error: return jsonify({'error': str(error)}), 500

        filename = f"{ticker.replace('.BO', '')}_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join('reports', filename)
        
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        # ... (full PDF generation logic would go here)
        # This is a simplified version for demonstration
        story.append(Paragraph(f"Analysis Report for {ticker}", styles['h1']))
        story.append(Paragraph(f"Prediction: {prediction_result['final_prediction']}", styles['body']))

        doc.build(story)
        
        return send_file(filepath, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================================================================
# == HELPER FUNCTIONS
# ==================================================================
def generate_chart_image(stock_data, chart_type, ticker):
    try:
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 6))
        chart_data = stock_data.tail(100)
        
        if chart_type == 'price_and_ma':
            ax.plot(chart_data.index, chart_data['Close'], label='Close Price', color='blue', linewidth=2)
            ax.plot(chart_data.index, chart_data['MA_5'], label='MA 5', color='green', linewidth=1.5)
            ax.plot(chart_data.index, chart_data['MA_20'], label='MA 20', color='orange', linewidth=1.5)
            ax.plot(chart_data.index, chart_data['MA_50'], label='MA 50', color='purple', linewidth=1.5)
            ax.set_title(f'{ticker} - Price with Moving Averages', fontsize=14)
        elif chart_type == 'rsi':
            ax.plot(chart_data.index, chart_data['RSI'], label='RSI', color='purple', linewidth=2)
            ax.axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
            ax.axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
            ax.set_title(f'{ticker} - RSI Indicator', fontsize=14)
            ax.set_ylim(0, 100)
        elif chart_type == 'correlation':
             features = ['Close', 'MA_5', 'MA_20', 'MA_50', 'RSI', 'Daily_Return', 'Volatility_20']
             available = [f for f in features if f in chart_data.columns]
             correlation_matrix = chart_data[available].corr()
             sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
             ax.set_title(f'{ticker} - Feature Correlation Heatmap', fontsize=14)

        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150)
        img_buffer.seek(0)
        plt.close(fig)
        return img_buffer
    except Exception as e:
        plt.close('all')
        raise e

def get_country_from_exchange(exchange):
    """Map common exchange codes to country names."""
    exchange_country_map = {
        'NMS': 'United States', 'NYQ': 'United States', 'PCX': 'United States', 'ASE': 'United States',
        'LSE': 'United Kingdom', 'LON': 'United Kingdom',
        'GER': 'Germany', 'ETR': 'Germany', 'FRA': 'Germany',
        'PAR': 'France', 'EPA': 'France',
        'TSX': 'Canada', 'TOR': 'Canada',
        'ASX': 'Australia',
        'JPX': 'Japan', 'TKS': 'Japan',
        'HKG': 'Hong Kong',
        'SHH': 'China', 'SHZ': 'China',
        'BSE': 'India', 'NSI': 'India',
        'AMS': 'Netherlands',
        'SWX': 'Switzerland',
    }
    return exchange_country_map.get(exchange, 'International')


if __name__ == '__main__':
    if not os.path.exists('reports'):
        os.makedirs('reports')
    app.run(debug=True, port=5002)

