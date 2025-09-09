# setup_data.py
import yfinance as yf
import pandas as pd
import os

def setup_sample_data():
    """Create sample data files for the financial modeling toolkit"""
    
    # Ensure directories exist
    os.makedirs('data/sample_data', exist_ok=True)
    
    # Sample companies across sectors
    companies = {
        'AAPL': 'Technology',
        'MSFT': 'Technology', 
        'NEE': 'Utilities',
        'AMT': 'Infrastructure',
        'JPM': 'Financial Services'
    }
    
    financials_data = []
    
    for ticker, sector in companies.items():
        try:
            company = yf.Ticker(ticker)
            info = company.info
            
            financials_data.append({
                'ticker': ticker,
                'company_name': info.get('shortName', ''),
                'sector': sector,
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'revenue_ttm': info.get('totalRevenue', 0),
                'ebitda_ttm': info.get('ebitda', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'ev_revenue': info.get('enterpriseToRevenue', 0),
                'ev_ebitda': info.get('enterpriseToEbitda', 0)
            })
            print(f"✅ Got data for {ticker}")
            
        except Exception as e:
            print(f"❌ Error getting {ticker}: {e}")
    
    # Save to CSV
    df = pd.DataFrame(financials_data)
    df.to_csv('data/sample_data/company_financials.csv', index=False)
    print(f"📊 Saved data for {len(df)} companies")

if __name__ == "__main__":
    setup_sample_data()