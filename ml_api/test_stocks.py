import yfinance as yf
import pandas as pd

def test_stock_data(ticker):
    """Test if stock data is available for a given ticker"""
    try:
        print(f"\n=== Testing {ticker} ===")
        
        # Download recent data
        data = yf.download(ticker, start="2020-01-01", end="2025-08-18")
        
        if data.empty:
            print(f"❌ No data found for {ticker}")
            return False
        
        # Handle MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        print(f"✅ Data available for {ticker}")
        print(f"   Data points: {len(data)}")
        print(f"   Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Latest close price: ₹{data['Close'].iloc[-1]:.2f}")
        
        # Check if we have enough data for ML
        if len(data) < 100:
            print(f"⚠️  Warning: Only {len(data)} data points (need at least 100 for ML)")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing {ticker}: {str(e)}")
        return False

if __name__ == "__main__":
    stocks = [
        "INFIBEAM.BO",
        "SUZLON.BO", 
        "TTML.BO"
    ]
    
    print("Testing stock data availability on Yahoo Finance...")
    
    results = {}
    for ticker in stocks:
        results[ticker] = test_stock_data(ticker)
    
    print("\n=== SUMMARY ===")
    for ticker, available in results.items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"{ticker}: {status}")
    
    available_count = sum(results.values())
    print(f"\nTotal available stocks: {available_count}/{len(stocks)}")
