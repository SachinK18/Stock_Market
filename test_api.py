import requests
import json

def test_stock_search():
    """Test the updated stock search API"""
    try:
        # Test 1: Search for Apple
        print("Testing search for 'apple'...")
        response = requests.get('http://localhost:5002/api/search-stocks?q=apple')
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Found {data.get('count', 0)} stocks for 'apple'")
            if data.get('stocks'):
                print("Sample results:")
                for stock in data['stocks'][:3]:
                    print(f"  - {stock.get('symbol', 'N/A')}: {stock.get('name', 'N/A')} ({stock.get('country', 'N/A')})")
        else:
            print(f"✗ Error: {response.status_code}")

        print()

        # Test 2: Search for Microsoft
        print("Testing search for 'microsoft'...")
        response = requests.get('http://localhost:5002/api/search-stocks?q=microsoft')
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Found {data.get('count', 0)} stocks for 'microsoft'")
            if data.get('stocks'):
                print("Sample results:")
                for stock in data['stocks'][:3]:
                    print(f"  - {stock.get('symbol', 'N/A')}: {stock.get('name', 'N/A')} ({stock.get('country', 'N/A')})")
        else:
            print(f"✗ Error: {response.status_code}")

        print()

        # Test 3: Search for Reliance (Indian stock)
        print("Testing search for 'reliance'...")
        response = requests.get('http://localhost:5002/api/search-stocks?q=reliance')
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Found {data.get('count', 0)} stocks for 'reliance'")
            if data.get('stocks'):
                print("Sample results:")
                for stock in data['stocks'][:3]:
                    print(f"  - {stock.get('symbol', 'N/A')}: {stock.get('name', 'N/A')} ({stock.get('country', 'N/A')})")
        else:
            print(f"✗ Error: {response.status_code}")

    except Exception as e:
        print(f"Error testing API: {e}")

if __name__ == "__main__":
    test_stock_search()
