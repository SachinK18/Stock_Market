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
matplotlib.use('Agg')  # Use non-interactive backend
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
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

class StockPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.log_model = LogisticRegression(max_iter=1000)
        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42
        )
        self.features = [
            "Daily_Return", "MA_5", "MA_20", "MA_50",
            "Price_MA5_ratio", "Price_MA20_ratio", "RSI", "Volatility_20"
        ]
        
    def prepare_data(self, ticker, end_date="2025-08-28"):
        """Download and prepare stock data"""
        try:
            # Download data
            data = yf.download(ticker, start="2010-01-01", end=end_date)
            
            if data.empty:
                return None, "No data found for ticker"
            
            # Handle MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            
            # Remove missing values
            data = data.dropna()
            
            if len(data) < 100:  # Need sufficient data
                return None, "Insufficient data for prediction"
            
            # Feature engineering
            data["Daily_Return"] = data["Close"].pct_change()
            data["MA_5"] = data["Close"].rolling(window=5).mean()
            data["MA_20"] = data["Close"].rolling(window=20).mean()
            data["MA_50"] = data["Close"].rolling(window=50).mean()
            
            data["Price_MA5_ratio"] = data["Close"] / data["MA_5"]
            data["Price_MA20_ratio"] = data["Close"] / data["MA_20"]
            
            # RSI calculation
            delta = data["Close"].diff()
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain, index=data.index).rolling(14).mean()
            avg_loss = pd.Series(loss, index=data.index).rolling(14).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            data["RSI"] = 100 - (100 / (1 + rs))
            
            # Volatility
            data["Volatility_20"] = data["Daily_Return"].rolling(window=20).std()
            
            # Target variable
            data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
            
            # Drop NA values
            data = data.dropna()
            
            return data, None
            
        except Exception as e:
            return None, str(e)
    
    def train_models(self, data):
        """Train both models"""
        try:
            X = data[self.features]
            y = data["Target"]
            
            # Split data (80% train, 20% test)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features for logistic regression
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            self.log_model.fit(X_train_scaled, y_train)
            self.rf_model.fit(X_train, y_train)
            
            # Calculate accuracies
            log_pred = self.log_model.predict(X_test_scaled)
            rf_pred = self.rf_model.predict(X_test)
            
            log_accuracy = accuracy_score(y_test, log_pred)
            rf_accuracy = accuracy_score(y_test, rf_pred)
            
            return {
                'log_accuracy': log_accuracy,
                'rf_accuracy': rf_accuracy,
                'latest_features': X.iloc[-1].values
            }, None
            
        except Exception as e:
            return None, str(e)
    
    def predict(self, latest_features):
        """Make predictions using both models"""
        try:
            # Reshape for prediction
            features_reshaped = latest_features.reshape(1, -1)
            
            # Logistic Regression prediction
            log_pred = self.log_model.predict(self.scaler.transform(features_reshaped))[0]
            
            # Random Forest prediction
            rf_pred = self.rf_model.predict(features_reshaped)[0]
            
            # Determine final prediction
            if log_pred == 1 and rf_pred == 1:
                final_prediction = "UP"
                confidence = "High"
            elif log_pred == 0 and rf_pred == 0:
                final_prediction = "DOWN"
                confidence = "High"
            else:
                final_prediction = "SLIGHTLY UP - RISKY"
                confidence = "Medium"
            
            return {
                'logistic_prediction': "UP" if log_pred == 1 else "DOWN",
                'random_forest_prediction': "UP" if rf_pred == 1 else "DOWN",
                'final_prediction': final_prediction,
                'confidence': confidence
            }, None
            
        except Exception as e:
            return None, str(e)

# Global predictor instance
predictor = StockPredictor()

@app.route('/api/predict', methods=['POST'])
def predict_stock():
    try:
        data = request.get_json()
        ticker = data.get('ticker', 'INFIBEAM.BO')
        end_date = data.get('end_date', '2025-08-28')

        # Prepare data
        stock_data, error = predictor.prepare_data(ticker, end_date)
        if error:
            return jsonify({'error': error}), 400
        
        # Train models
        training_result, error = predictor.train_models(stock_data)
        if error:
            return jsonify({'error': error}), 400
        
        # Make prediction
        prediction_result, error = predictor.predict(training_result['latest_features'])
        if error:
            return jsonify({'error': error}), 400
        
        # Combine results
        result = {
            'ticker': ticker,
            'end_date': end_date,
            'prediction': prediction_result,
            'model_accuracies': {
                'logistic_regression': round(training_result['log_accuracy'], 4),
                'random_forest': round(training_result['rf_accuracy'], 4)
            },
            'current_price': float(stock_data['Close'].iloc[-1]),
            'last_updated': stock_data.index[-1].strftime('%Y-%m-%d'),
            'data_points_used': len(stock_data)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/heatmap-image', methods=['POST'])
def get_heatmap_image():
    """Generate heatmap image for frontend display"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', 'INFIBEAM.BO')
        end_date = data.get('end_date', '2025-08-28')

        # Get stock data
        stock_data, error = predictor.prepare_data(ticker, end_date)
        if error:
            return jsonify({'error': error}), 400

        # Generate heatmap image
        heatmap_buffer = generate_chart_image(stock_data, 'correlation', ticker)

        # Convert to base64 for frontend
        import base64
        img_base64 = base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')

        return jsonify({
            'image': img_base64,
            'ticker': ticker
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart-data', methods=['POST'])
def get_chart_data():
    """Generate chart data based on the ML code provided"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', 'INFIBEAM.BO')
        chart_type = data.get('chart_type', 'close_price')
        end_date = data.get('end_date', '2025-08-28')

        # Download stock data (same as ML code)
        stock_data = yf.download(ticker, start="2010-01-01", end=end_date)

        if stock_data.empty:
            return jsonify({'error': 'No data found for ticker'}), 400

        # Handle MultiIndex (same as ML code)
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = [col[0] for col in stock_data.columns]

        # Remove missing values (same as ML code)
        stock_data = stock_data.dropna()

        # Feature engineering (same as ML code)
        stock_data["Daily_Return"] = stock_data["Close"].pct_change()
        stock_data["MA_5"] = stock_data["Close"].rolling(window=5).mean()
        stock_data["MA_20"] = stock_data["Close"].rolling(window=20).mean()
        stock_data["MA_50"] = stock_data["Close"].rolling(window=50).mean()

        # Price ratios (same as ML code)
        stock_data["Price_MA5_ratio"] = stock_data["Close"] / stock_data["MA_5"]
        stock_data["Price_MA20_ratio"] = stock_data["Close"] / stock_data["MA_20"]

        # RSI calculation (same as ML code)
        delta = stock_data["Close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain, index=stock_data.index).rolling(14).mean()
        avg_loss = pd.Series(loss, index=stock_data.index).rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        stock_data["RSI"] = 100 - (100 / (1 + rs))

        # Volatility (same as ML code)
        stock_data["Volatility_20"] = stock_data["Daily_Return"].rolling(window=20).std()

        # Drop NA from features (same as ML code)
        stock_data = stock_data.dropna()

        # Get latest values
        latest_data = stock_data.iloc[-1]
        current_rsi = stock_data["RSI"].iloc[-1]

        # Prepare chart data based on type
        chart_info = {
            'close_price': 'Close Price Chart',
            'moving_averages': 'Close Price with Moving Averages',
            'candlestick': 'Candlestick Chart (OHLC)',
            'volume': 'Volume Chart',
            'correlation_heatmap': 'Feature Correlation Heatmap'
        }

        # Get last 100 data points for charts (or all if less than 100)
        chart_data_points = stock_data.tail(100)

        # Calculate feature correlations (same features as ML code)
        features = ["Daily_Return", "MA_5", "MA_20", "MA_50", "Price_MA5_ratio", "Price_MA20_ratio", "RSI", "Volatility_20"]
        available_features = [f for f in features if f in stock_data.columns]
        correlation_matrix = stock_data[available_features].corr()

        # Convert correlation matrix to heatmap data format
        heatmap_data = []
        for i, feature1 in enumerate(available_features):
            for j, feature2 in enumerate(available_features):
                heatmap_data.append({
                    'x': j,
                    'y': i,
                    'v': float(correlation_matrix.iloc[i, j])
                })

        # Also create formatted display data for PDF
        heatmap_display_data = []
        for i, feature1 in enumerate(available_features):
            for j, feature2 in enumerate(available_features):
                value = float(correlation_matrix.iloc[i, j])
                heatmap_display_data.append({
                    'x': j,
                    'y': i,
                    'v': value,
                    'display': f'{value:.2f}'
                })

        # Also keep simple correlation data for display
        correlation_data = {}
        for i, feature1 in enumerate(available_features):
            for j, feature2 in enumerate(available_features):
                if i < j:  # Only upper triangle to avoid duplicates
                    key = f"{feature1} vs {feature2}"
                    correlation_data[key] = float(correlation_matrix.iloc[i, j])

        result = {
            'ticker': ticker,
            'chart_type': chart_type,
            'chart_title': chart_info.get(chart_type, 'Stock Chart'),
            'current_price': float(latest_data['Close']),
            'current_rsi': float(current_rsi),
            'data_points': len(stock_data),
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
            'moving_averages': {
                'ma_5': float(stock_data["MA_5"].iloc[-1]),
                'ma_20': float(stock_data["MA_20"].iloc[-1]),
                'ma_50': float(stock_data["MA_50"].iloc[-1])
            },
            'correlation_data': correlation_data,
            'heatmap_data': heatmap_data,
            'feature_labels': available_features,
            'chart_data': {
                'labels': [date.strftime('%Y-%m-%d') for date in chart_data_points.index],
                'close_prices': chart_data_points['Close'].tolist(),
                'open_prices': chart_data_points['Open'].tolist(),
                'high_prices': chart_data_points['High'].tolist(),
                'low_prices': chart_data_points['Low'].tolist(),
                'volumes': chart_data_points['Volume'].tolist() if 'Volume' in chart_data_points.columns else [],
                'ma_5': chart_data_points['MA_5'].dropna().tolist(),
                'ma_20': chart_data_points['MA_20'].dropna().tolist(),
                'ma_50': chart_data_points['MA_50'].dropna().tolist(),
                'rsi': chart_data_points['RSI'].dropna().tolist()
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_chart_image(stock_data, chart_type, ticker):
    """Generate chart images for PDF report"""
    try:
        # Clear any existing plots
        plt.clf()
        plt.close('all')

        # Create figure with explicit backend
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get last 100 data points for charts
        chart_data = stock_data.tail(100)

        if chart_type == 'price_and_ma':
            ax.plot(chart_data.index, chart_data['Close'], label='Close Price', color='blue', linewidth=2)
            ax.plot(chart_data.index, chart_data['MA_5'], label='MA 5', color='green', linewidth=1.5)
            ax.plot(chart_data.index, chart_data['MA_20'], label='MA 20', color='orange', linewidth=1.5)
            ax.plot(chart_data.index, chart_data['MA_50'], label='MA 50', color='purple', linewidth=1.5)
            ax.set_title(f'{ticker} - Price with Moving Averages', fontsize=14, fontweight='bold')
            ax.set_ylabel('Price (₹)', fontsize=12)

        elif chart_type == 'rsi':
            ax.plot(chart_data.index, chart_data['RSI'], label='RSI', color='purple', linewidth=2)
            ax.axhline(y=70, color='red', linestyle='--', label='Overbought (70)', linewidth=1.5)
            ax.axhline(y=30, color='green', linestyle='--', label='Oversold (30)', linewidth=1.5)
            ax.set_title(f'{ticker} - RSI Indicator', fontsize=14, fontweight='bold')
            ax.set_ylabel('RSI', fontsize=12)
            ax.set_ylim(0, 100)

        elif chart_type == 'correlation':
            # Create correlation heatmap
            features = ['Close', 'MA_5', 'MA_20', 'MA_50', 'RSI', 'Daily_Return', 'Volatility_20']
            available_features = [f for f in features if f in chart_data.columns]
            correlation_matrix = chart_data[available_features].corr()

            # Create heatmap with proper square aspect ratio
            im = ax.imshow(correlation_matrix.values, cmap='coolwarm', aspect='equal', vmin=-1, vmax=1)
            ax.set_xticks(range(len(available_features)))
            ax.set_yticks(range(len(available_features)))
            ax.set_xticklabels(available_features, rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(available_features, fontsize=10)
            ax.set_title(f'{ticker} - Feature Correlation Heatmap', fontsize=14, fontweight='bold')

            # Add correlation values as text with better formatting
            for i in range(len(available_features)):
                for j in range(len(available_features)):
                    value = correlation_matrix.iloc[i, j]
                    text_color = 'white' if abs(value) > 0.5 else 'black'
                    ax.text(j, i, f'{value:.2f}',
                           ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')

            # Add colorbar with better positioning
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
            cbar.set_label('Correlation Coefficient', fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)

        ax.set_xlabel('Date', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)

        # Clean up
        plt.close(fig)
        plt.clf()

        return img_buffer

    except Exception as e:
        # Ensure cleanup even on error
        plt.close('all')
        plt.clf()
        raise e

@app.route('/api/generate-pdf-report', methods=['POST'])
def generate_pdf_report():
    """Generate PDF report with charts and analysis"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', 'INFIBEAM.BO')
        end_date = data.get('end_date', '2025-08-28')

        # Get stock data and predictions
        stock_data, error = predictor.prepare_data(ticker, end_date)
        if error:
            return jsonify({'error': error}), 400

        training_result, error = predictor.train_models(stock_data)
        if error:
            return jsonify({'error': error}), 400

        prediction_result, error = predictor.predict(training_result['latest_features'])
        if error:
            return jsonify({'error': error}), 400

        # Create PDF
        filename = f"{ticker.replace('.BO', '')}_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join('reports', filename)

        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)

        doc = SimpleDocTemplate(filepath, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.darkblue
        )
        story.append(Paragraph(f"Stock Analysis Report", title_style))
        story.append(Paragraph(f"{ticker.replace('.BO', '')} ({ticker})", title_style))
        story.append(Spacer(1, 20))

        # Report Info
        info_style = styles['Normal']
        story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", info_style))
        story.append(Paragraph(f"<b>Analysis Date:</b> {end_date}", info_style))
        story.append(Spacer(1, 20))

        # Prediction Results
        pred_style = ParagraphStyle(
            'PredictionStyle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.darkgreen
        )
        story.append(Paragraph("PREDICTION RESULTS", pred_style))
        story.append(Spacer(1, 10))

        # Prediction table
        pred_data = [
            ['Metric', 'Value'],
            ['Overall Prediction', prediction_result['final_prediction']],
            ['Confidence Level', prediction_result['confidence']],
            ['Logistic Regression', prediction_result['logistic_prediction']],
            ['Random Forest', prediction_result['random_forest_prediction']],
            ['LR Accuracy', f"{training_result['log_accuracy']:.2%}"],
            ['RF Accuracy', f"{training_result['rf_accuracy']:.2%}"]
        ]

        pred_table = Table(pred_data, colWidths=[2*inch, 2*inch])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(pred_table)
        story.append(Spacer(1, 20))

        # Current Market Data
        story.append(Paragraph("CURRENT MARKET DATA", pred_style))
        story.append(Spacer(1, 10))

        current_price = float(stock_data['Close'].iloc[-1])
        current_rsi = float(stock_data['RSI'].iloc[-1])

        market_data = [
            ['Metric', 'Value'],
            ['Current Price', f"₹{current_price:.2f}"],
            ['Current RSI', f"{current_rsi:.2f}"],
            ['RSI Status', 'Overbought' if current_rsi > 70 else 'Oversold' if current_rsi < 30 else 'Neutral'],
            ['MA 5', f"₹{stock_data['MA_5'].iloc[-1]:.2f}"],
            ['MA 20', f"₹{stock_data['MA_20'].iloc[-1]:.2f}"],
            ['MA 50', f"₹{stock_data['MA_50'].iloc[-1]:.2f}"]
        ]

        market_table = Table(market_data, colWidths=[2*inch, 2*inch])
        market_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(market_table)
        story.append(Spacer(1, 30))

        # Generate and add charts
        story.append(Paragraph("PRICE ANALYSIS CHART", pred_style))
        story.append(Spacer(1, 10))

        # Price chart
        price_chart = generate_chart_image(stock_data, 'price_and_ma', ticker)
        price_img_path = f"reports/price_chart_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        with open(price_img_path, 'wb') as f:
            f.write(price_chart.getvalue())
        story.append(Image(price_img_path, width=6*inch, height=3.6*inch))
        story.append(Spacer(1, 20))

        # RSI chart
        story.append(Paragraph("RSI INDICATOR CHART", pred_style))
        story.append(Spacer(1, 10))

        rsi_chart = generate_chart_image(stock_data, 'rsi', ticker)
        rsi_img_path = f"reports/rsi_chart_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        with open(rsi_img_path, 'wb') as f:
            f.write(rsi_chart.getvalue())
        story.append(Image(rsi_img_path, width=6*inch, height=3.6*inch))
        story.append(Spacer(1, 20))

        # Correlation heatmap
        story.append(Paragraph("FEATURE CORRELATION HEATMAP", pred_style))
        story.append(Spacer(1, 10))

        correlation_chart = generate_chart_image(stock_data, 'correlation', ticker)
        correlation_img_path = f"reports/correlation_chart_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        with open(correlation_img_path, 'wb') as f:
            f.write(correlation_chart.getvalue())
        story.append(Image(correlation_img_path, width=6*inch, height=4.5*inch))
        story.append(Spacer(1, 30))

        # Disclaimer
        disclaimer_style = ParagraphStyle(
            'DisclaimerStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.red,
            alignment=1
        )
        story.append(Paragraph("DISCLAIMER", pred_style))
        story.append(Paragraph(
            "This report is for educational purposes only and should not be considered as financial advice. "
            "Please consult with a qualified financial advisor before making investment decisions. "
            "Past performance does not guarantee future results.",
            disclaimer_style
        ))

        # Build PDF
        doc.build(story)

        # Clean up temporary image files and matplotlib
        try:
            os.remove(price_img_path)
            os.remove(rsi_img_path)
            os.remove(correlation_img_path)
        except:
            pass

        # Final matplotlib cleanup
        plt.close('all')
        plt.clf()

        return send_file(filepath, as_attachment=True, download_name=filename)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stocks', methods=['GET'])
def get_available_stocks():
    """Return list of available stocks"""
    stocks = [
        {'symbol': 'INFIBEAM.BO', 'name': 'Infibeam Avenues Limited'},
        {'symbol': 'SUZLON.BO', 'name': 'Suzlon Energy Limited'},
        {'symbol': 'TTML.BO', 'name': 'Tata Teleservices (Maharashtra) Limited'}
    ]
    return jsonify(stocks)

@app.route('/api/search-stocks', methods=['GET'])
def search_stocks():
    """Search for stocks using yfinance from all nations"""
    try:
        query = request.args.get('q', '').strip()

        if not query or len(query) < 2:
            return jsonify({'stocks': []})

        # Use yfinance to search for stocks across all markets
        try:
            # First try to get ticker info directly if it's a valid symbol
            ticker = yf.Ticker(query)
            ticker_info = ticker.info

            # If we get valid ticker info, add it to results
            search_results = []
            if ticker_info and 'symbol' in ticker_info and 'shortName' in ticker_info:
                exchange = ticker_info.get('exchange', 'Unknown')
                currency = ticker_info.get('currency', 'Unknown')

                # Determine country/region based on exchange or other info
                country = get_country_from_exchange(exchange)

                search_results.append({
                    'symbol': ticker_info['symbol'],
                    'name': ticker_info['shortName'],
                    'exchange': exchange,
                    'country': country,
                    'currency': currency
                })

            # Also search for similar symbols/tickers using yfinance search
            try:
                # Use yfinance's search functionality for broader results
                search_tickers = []
                common_suffixes = ['', '.NS', '.BO', '.NSE', '.BSE', '.TO', '.L', '.PA', '.DE', '.F', '.MI', '.AS', '.AX', '.CO', '.SA']

                # Try common symbol variations
                for suffix in common_suffixes[:5]:  # Limit to avoid too many requests
                    try:
                        test_ticker = yf.Ticker(query + suffix)
                        test_info = test_ticker.info
                        if (test_info and 'symbol' in test_info and
                            test_info['symbol'] not in [s['symbol'] for s in search_results]):
                            exchange = test_info.get('exchange', 'Unknown')
                            country = get_country_from_exchange(exchange)
                            search_results.append({
                                'symbol': test_info['symbol'],
                                'name': test_info.get('shortName', test_info['symbol']),
                                'exchange': exchange,
                                'country': country,
                                'currency': test_info.get('currency', 'Unknown')
                            })
                    except:
                        continue

            except Exception as e:
                print(f"Search error: {e}")

        except Exception as e:
            print(f"Ticker lookup error: {e}")
            search_results = []

        # Also include some popular international stocks for better coverage
        popular_stocks = get_popular_international_stocks(query)

        # Combine results and remove duplicates
        all_stocks = search_results + popular_stocks
        seen_symbols = set()
        unique_stocks = []

        for stock in all_stocks:
            if stock['symbol'] not in seen_symbols:
                seen_symbols.add(stock['symbol'])
                unique_stocks.append(stock)

        # Filter based on search query (case-insensitive)
        filtered_stocks = []
        query_lower = query.lower()

        for stock in unique_stocks:
            if (query_lower in stock['symbol'].lower() or
                query_lower in stock['name'].lower() or
                any(word.lower().startswith(query_lower) for word in stock['name'].split()) or
                (stock.get('country', '').lower().startswith(query_lower))):
                filtered_stocks.append(stock)

        # Sort by relevance (prioritize symbol matches, then name matches)
        def sort_key(stock):
            symbol_lower = stock['symbol'].lower()
            name_lower = stock['name'].lower()

            if query_lower == symbol_lower:
                return 0  # Exact symbol match
            elif query_lower in symbol_lower:
                return 1  # Partial symbol match
            elif query_lower in name_lower:
                return 2  # Name match
            else:
                return 3  # Other match

        filtered_stocks.sort(key=sort_key)

        # Limit results to 30 for better performance
        filtered_stocks = filtered_stocks[:30]

        return jsonify({
            'stocks': filtered_stocks,
            'query': query,
            'count': len(filtered_stocks)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_country_from_exchange(exchange):
    """Map exchange codes to country names"""
    exchange_country_map = {
        'NSE': 'India',
        'BSE': 'India',
        'BOM': 'India',
        'NSI': 'India',
        'NYSE': 'United States',
        'NASDAQ': 'United States',
        'LSE': 'United Kingdom',
        'LON': 'United Kingdom',
        'EPA': 'France',
        'PAR': 'France',
        'FRA': 'Germany',
        'ETR': 'Germany',
        'BIT': 'Italy',
        'MIL': 'Italy',
        'AMS': 'Netherlands',
        'AEX': 'Netherlands',
        'BRU': 'Belgium',
        'EBR': 'Belgium',
        'MCE': 'Spain',
        'BME': 'Spain',
        'STO': 'Sweden',
        'OMX': 'Sweden',
        'HEL': 'Finland',
        'CPH': 'Denmark',
        'OSL': 'Norway',
        'ICE': 'Iceland',
        'TSE': 'Japan',
        'TYO': 'Japan',
        'HKEX': 'Hong Kong',
        'HKG': 'Hong Kong',
        'SSE': 'China',
        'SHG': 'China',
        'SZSE': 'China',
        'SHE': 'China',
        'ASX': 'Australia',
        'TSX': 'Canada',
        'TOR': 'Canada',
        'BVMF': 'Brazil',
        'SAO': 'Brazil',
        'BCBA': 'Argentina',
        'BUE': 'Argentina',
        'BMV': 'Mexico',
        'BCS': 'Chile',
        'SGO': 'Chile',
        'BVL': 'Peru',
        'LIM': 'Peru',
        'BSSE': 'Bulgaria',
        'SOF': 'Bulgaria',
        'PSE': 'Philippines',
        'SET': 'Thailand',
        'BKK': 'Thailand',
        'IDX': 'Indonesia',
        'JSE': 'South Africa',
        'KRX': 'South Korea',
        'SEO': 'South Korea',
        'TWSE': 'Taiwan',
        'TAI': 'Taiwan',
        'BIST': 'Turkey',
        'IST': 'Turkey',
        'TASE': 'Israel',
        'TLV': 'Israel',
        'MOEX': 'Russia',
        'MISX': 'Russia',
        'WSE': 'Poland',
        'WAR': 'Poland',
        'BVB': 'Romania',
        'BUC': 'Romania',
        'VSE': 'Austria',
        'VIE': 'Austria',
        'SWX': 'Switzerland',
        'SIX': 'Switzerland',
        'OSE': 'Norway',
        'NOK': 'Norway'
    }

    return exchange_country_map.get(exchange, 'International')

def get_popular_international_stocks(query):
    """Get popular international stocks for better search coverage"""
    popular_stocks = [
        # US Stocks
        {'symbol': 'AAPL', 'name': 'Apple Inc.', 'exchange': 'NASDAQ', 'country': 'United States', 'currency': 'USD'},
        {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'exchange': 'NASDAQ', 'country': 'United States', 'currency': 'USD'},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'exchange': 'NASDAQ', 'country': 'United States', 'currency': 'USD'},
        {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'exchange': 'NASDAQ', 'country': 'United States', 'currency': 'USD'},
        {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'exchange': 'NASDAQ', 'country': 'United States', 'currency': 'USD'},
        {'symbol': 'META', 'name': 'Meta Platforms Inc.', 'exchange': 'NASDAQ', 'country': 'United States', 'currency': 'USD'},
        {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'exchange': 'NASDAQ', 'country': 'United States', 'currency': 'USD'},
        {'symbol': 'NFLX', 'name': 'Netflix Inc.', 'exchange': 'NASDAQ', 'country': 'United States', 'currency': 'USD'},
        {'symbol': 'DIS', 'name': 'The Walt Disney Company', 'exchange': 'NYSE', 'country': 'United States', 'currency': 'USD'},
        {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.', 'exchange': 'NYSE', 'country': 'United States', 'currency': 'USD'},
        {'symbol': 'V', 'name': 'Visa Inc.', 'exchange': 'NYSE', 'country': 'United States', 'currency': 'USD'},
        {'symbol': 'JNJ', 'name': 'Johnson & Johnson', 'exchange': 'NYSE', 'country': 'United States', 'currency': 'USD'},
        {'symbol': 'WMT', 'name': 'Walmart Inc.', 'exchange': 'NYSE', 'country': 'United States', 'currency': 'USD'},
        {'symbol': 'PG', 'name': 'The Procter & Gamble Company', 'exchange': 'NYSE', 'country': 'United States', 'currency': 'USD'},
        {'symbol': 'UNH', 'name': 'UnitedHealth Group Incorporated', 'exchange': 'NYSE', 'country': 'United States', 'currency': 'USD'},
        {'symbol': 'HD', 'name': 'The Home Depot Inc.', 'exchange': 'NYSE', 'country': 'United States', 'currency': 'USD'},
        {'symbol': 'BAC', 'name': 'Bank of America Corporation', 'exchange': 'NYSE', 'country': 'United States', 'currency': 'USD'},
        {'symbol': 'KO', 'name': 'The Coca-Cola Company', 'exchange': 'NYSE', 'country': 'United States', 'currency': 'USD'},
        {'symbol': 'PFE', 'name': 'Pfizer Inc.', 'exchange': 'NYSE', 'country': 'United States', 'currency': 'USD'},
        {'symbol': 'PEP', 'name': 'PepsiCo Inc.', 'exchange': 'NASDAQ', 'country': 'United States', 'currency': 'USD'},

        # UK Stocks
        {'symbol': 'AZN.L', 'name': 'AstraZeneca PLC', 'exchange': 'LSE', 'country': 'United Kingdom', 'currency': 'GBP'},
        {'symbol': 'HSBA.L', 'name': 'HSBC Holdings PLC', 'exchange': 'LSE', 'country': 'United Kingdom', 'currency': 'GBP'},
        {'symbol': 'ULVR.L', 'name': 'Unilever PLC', 'exchange': 'LSE', 'country': 'United Kingdom', 'currency': 'GBP'},
        {'symbol': 'DGE.L', 'name': 'Diageo PLC', 'exchange': 'LSE', 'country': 'United Kingdom', 'currency': 'GBP'},
        {'symbol': 'RIO.L', 'name': 'Rio Tinto PLC', 'exchange': 'LSE', 'country': 'United Kingdom', 'currency': 'GBP'},
        {'symbol': 'BATS.L', 'name': 'British American Tobacco PLC', 'exchange': 'LSE', 'country': 'United Kingdom', 'currency': 'GBP'},

        # European Stocks
        {'symbol': 'SAP.DE', 'name': 'SAP SE', 'exchange': 'XETRA', 'country': 'Germany', 'currency': 'EUR'},
        {'symbol': 'SIE.DE', 'name': 'Siemens AG', 'exchange': 'XETRA', 'country': 'Germany', 'currency': 'EUR'},
        {'symbol': 'ALV.DE', 'name': 'Allianz SE', 'exchange': 'XETRA', 'country': 'Germany', 'currency': 'EUR'},
        {'symbol': 'MC.PA', 'name': 'LVMH Moët Hennessy Louis Vuitton SE', 'exchange': 'EPA', 'country': 'France', 'currency': 'EUR'},
        {'symbol': 'OR.PA', 'name': 'L\'Oréal S.A.', 'exchange': 'EPA', 'country': 'France', 'currency': 'EUR'},
        {'symbol': 'ASML.AS', 'name': 'ASML Holding N.V.', 'exchange': 'AMS', 'country': 'Netherlands', 'currency': 'EUR'},
        {'symbol': 'NESN.SW', 'name': 'Nestlé S.A.', 'exchange': 'SIX', 'country': 'Switzerland', 'currency': 'CHF'},
        {'symbol': 'ROG.SW', 'name': 'Roche Holding AG', 'exchange': 'SIX', 'country': 'Switzerland', 'currency': 'CHF'},
        {'symbol': 'NOVN.SW', 'name': 'Novartis AG', 'exchange': 'SIX', 'country': 'Switzerland', 'currency': 'CHF'},

        # Asian Stocks
        {'symbol': '005930.KS', 'name': 'Samsung Electronics Co., Ltd.', 'exchange': 'KRX', 'country': 'South Korea', 'currency': 'KRW'},
        {'symbol': '000660.KS', 'name': 'SK Hynix Inc.', 'exchange': 'KRX', 'country': 'South Korea', 'currency': 'KRW'},
        {'symbol': '7203.T', 'name': 'Toyota Motor Corporation', 'exchange': 'TSE', 'country': 'Japan', 'currency': 'JPY'},
        {'symbol': '6758.T', 'name': 'Sony Group Corporation', 'exchange': 'TSE', 'country': 'Japan', 'currency': 'JPY'},
        {'symbol': '9984.T', 'name': 'SoftBank Group Corp.', 'exchange': 'TSE', 'country': 'Japan', 'currency': 'JPY'},
        {'symbol': '0700.HK', 'name': 'Tencent Holdings Limited', 'exchange': 'HKEX', 'country': 'Hong Kong', 'currency': 'HKD'},
        {'symbol': '1398.HK', 'name': 'Industrial and Commercial Bank of China Limited', 'exchange': 'HKEX', 'country': 'Hong Kong', 'currency': 'HKD'},
        {'symbol': 'BABA', 'name': 'Alibaba Group Holding Limited', 'exchange': 'NYSE', 'country': 'China', 'currency': 'USD'},
        {'symbol': 'NIO', 'name': 'NIO Inc.', 'exchange': 'NYSE', 'country': 'China', 'currency': 'USD'},

        # Canadian Stocks
        {'symbol': 'SHOP.TO', 'name': 'Shopify Inc.', 'exchange': 'TSX', 'country': 'Canada', 'currency': 'CAD'},
        {'symbol': 'TD.TO', 'name': 'The Toronto-Dominion Bank', 'exchange': 'TSX', 'country': 'Canada', 'currency': 'CAD'},
        {'symbol': 'ENB.TO', 'name': 'Enbridge Inc.', 'exchange': 'TSX', 'country': 'Canada', 'currency': 'CAD'},
        {'symbol': 'BNS.TO', 'name': 'The Bank of Nova Scotia', 'exchange': 'TSX', 'country': 'Canada', 'currency': 'CAD'},
        {'symbol': 'BMO.TO', 'name': 'Bank of Montreal', 'exchange': 'TSX', 'country': 'Canada', 'currency': 'CAD'},

        # Australian Stocks
        {'symbol': 'CBA.AX', 'name': 'Commonwealth Bank of Australia', 'exchange': 'ASX', 'country': 'Australia', 'currency': 'AUD'},
        {'symbol': 'BHP.AX', 'name': 'BHP Group Limited', 'exchange': 'ASX', 'country': 'Australia', 'currency': 'AUD'},
        {'symbol': 'CSL.AX', 'name': 'CSL Limited', 'exchange': 'ASX', 'country': 'Australia', 'currency': 'AUD'},
        {'symbol': 'NAB.AX', 'name': 'National Australia Bank Limited', 'exchange': 'ASX', 'country': 'Australia', 'currency': 'AUD'},
        {'symbol': 'WBC.AX', 'name': 'Westpac Banking Corporation', 'exchange': 'ASX', 'country': 'Australia', 'currency': 'AUD'}
    ]

    # Filter popular stocks based on query
    filtered_popular = []
    query_lower = query.lower()

    for stock in popular_stocks:
        if (query_lower in stock['symbol'].lower() or
            query_lower in stock['name'].lower() or
            any(word.lower().startswith(query_lower) for word in stock['name'].split()) or
            (stock.get('country', '').lower().startswith(query_lower))):
            filtered_popular.append(stock)

    return filtered_popular[:15]  # Limit popular stock results

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5002)
