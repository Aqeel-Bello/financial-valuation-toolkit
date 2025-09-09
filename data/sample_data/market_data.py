# fix_market_data.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def create_market_data():
    """Create comprehensive market data CSV with actual values"""
    
    print("Downloading market data...")
    
    # Define tickers for market data
    tickers = {
        '^GSPC': 'SP500',      # S&P 500 Index
        'XLU': 'Utilities',     # Utilities Sector ETF
        'XLE': 'Energy',        # Energy Sector ETF
        '^TNX': 'Treasury_10Y'  # 10-Year Treasury Yield
    }
    
    # Get 2 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    # Download all data at once
    try:
        data = yf.download(
            list(tickers.keys()), 
            start=start_date, 
            end=end_date,
            progress=False
        )
        
        # Get closing prices
        if len(tickers) > 1:
            closing_prices = data['Adj Close']
        else:
            closing_prices = data[['Adj Close']]
        
        # Rename columns to be more readable
        closing_prices.columns = [tickers.get(col, col) for col in closing_prices.columns]
        
        # Remove any rows with all NaN values
        closing_prices = closing_prices.dropna(how='all')
        
        # Save to CSV
        closing_prices.to_csv('data/sample_data/market_data.csv')
        
        print("✅ Market data saved successfully!")
        print(f"Data range: {closing_prices.index[0].strftime('%Y-%m-%d')} to {closing_prices.index[-1].strftime('%Y-%m-%d')}")
        print(f"Total rows: {len(closing_prices)}")
        
        # Show sample of the data
        print("\nSample data:")
        print(closing_prices.head())
        
        # Show latest values
        print("\nLatest values:")
        latest = closing_prices.iloc[-1]
        for column, value in latest.items():
            if pd.notna(value):
                if 'Treasury' in column:
                    print(f"{column}: {value:.2f}%")
                else:
                    print(f"{column}: ${value:.2f}")
        
        return closing_prices
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None

def create_simple_market_data():
    """Fallback: Create basic market data if download fails"""
    
    print("Creating basic market data...")
    
    # Get just S&P 500 data
    try:
        sp500 = yf.Ticker("^GSPC")
        data = sp500.history(period="2y")
        
        # Create simple DataFrame with just closing prices
        market_data = pd.DataFrame({
            'SP500': data['Close']
        })
        
        # Add some basic sector data
        try:
            utilities = yf.Ticker("XLU").history(period="2y")['Close']
            market_data['Utilities'] = utilities
        except:
            pass
            
        try:
            energy = yf.Ticker("XLE").history(period="2y")['Close']
            market_data['Energy'] = energy
        except:
            pass
        
        # Save to CSV
        market_data.to_csv('data/sample_data/market_data.csv')
        
        print("✅ Basic market data created!")
        print(f"Latest S&P 500: ${market_data['SP500'].iloc[-1]:.2f}")
        
        return market_data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    # Try comprehensive data first, fallback to simple if needed
    market_data = create_market_data()
    
    if market_data is None:
        print("\nTrying simpler approach...")
        market_data = create_simple_market_data()
    
    if market_data is not None:
        print("\n" + "="*50)
        print("Market data ready for your financial models!")
        print("="*50)
    else:
        print("\n❌ Could not create market data. Check your internet connection.")