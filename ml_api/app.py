
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
        print('DEBUG: Correlation matrix values:')
        print(correlation_matrix)
        for i, feature1 in enumerate(available_features):
            for j, feature2 in enumerate(available_features):
                value = float(correlation_matrix.iloc[i, j])
                print(f'DEBUG: {feature1} vs {feature2} = {value}')
                heatmap_data.append({
                    'x': j,
                    'y': i,
                    'v': value
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


@app.route('/api/correlation-heatmap-image', methods=['GET'])
def correlation_heatmap_image():
    try:
        ticker = request.args.get('ticker', 'INFIBEAM.BO')
        end_date = request.args.get('end_date', None)
        print(f'DEBUG: correlation_heatmap_image called for ticker={ticker}, end_date={end_date}')

        stock_data, error = predictor.prepare_data(ticker, end_date)
        # If data couldn't be prepared, return a small PNG image with the error text
        if error:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, f"Error: {error}", ha='center', va='center', fontsize=14, color='red')
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
            buf.seek(0)
            plt.close(fig)
            return send_file(buf, mimetype='image/png')

        # Use last 100 data points
        chart_data = stock_data.tail(100)
        features = [
            "Daily_Return", "MA_5", "MA_20", "MA_50",
            "Price_MA5_ratio", "Price_MA20_ratio", "RSI", "Volatility_20"
        ]
        available_features = [f for f in features if f in chart_data.columns]
        if not available_features:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, f"Error: No features available for heatmap", ha='center', va='center', fontsize=14, color='red')
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
            buf.seek(0)
            plt.close(fig)
            return send_file(buf, mimetype='image/png')

        corr_matrix = chart_data[available_features].corr()

        plt.clf()
        # Larger figure and higher DPI for clarity
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            vmin=-1, vmax=1,
            linewidths=0.5,
            cbar=True,
            ax=ax,
            fmt=".2f",
            annot_kws={"size": 14, 'weight': 'bold'},
            square=True
        )
        ax.set_title(f"{ticker} - Feature Correlation Heatmap", fontsize=20, fontweight='bold')
        ax.set_xticklabels(available_features, rotation=45, ha='right', fontsize=14)
        ax.set_yticklabels(available_features, fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.tight_layout()

        img_buffer = io.BytesIO()
        # Save at higher DPI so it appears crisp when enlarged in the frontend
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        plt.close()
        return send_file(img_buffer, mimetype='image/png')

    except Exception as e:
        # On unexpected exception, return an image describing the error so frontend can display it
        try:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, f"Server error: {str(e)}", ha='center', va='center', fontsize=12, color='red')
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
            buf.seek(0)
            plt.close(fig)
            return send_file(buf, mimetype='image/png')
        except Exception:
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

                # Create correlation heatmap using seaborn for better visual
                features = ["Daily_Return", "MA_5", "MA_20", "MA_50", "Price_MA5_ratio", "Price_MA20_ratio", "RSI", "Volatility_20"]
                available_features = [f for f in features if f in chart_data.columns]
                correlation_matrix = chart_data[available_features].corr()

                # Exact match to provided image
                sns.heatmap(
                    correlation_matrix,
                    annot=True,
                    cmap='coolwarm',
                    vmin=-1, vmax=1,
                    linewidths=0.5,
                    cbar=True,
                    ax=ax,
                    fmt='.2f',
                    annot_kws={"size": 12},
                    square=True
                )
                ax.set_title('Feature Correlation Heatmap', fontsize=18)
                ax.set_xticklabels(available_features, rotation=45, ha='right', fontsize=12)
                ax.set_yticklabels(available_features, fontsize=12)
                # Remove axis label for x/y to match image
                ax.set_xlabel("")
                ax.set_ylabel("")
                plt.tight_layout()

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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5002)
