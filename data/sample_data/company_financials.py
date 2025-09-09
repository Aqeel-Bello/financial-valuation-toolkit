import pandas as pd
import yfinance as yf

# Create sample financials
companies = ['AAPL', 'MSFT', 'NEE', 'AMT']
financials_data = []

for ticker in companies:
    company = yf.Ticker(ticker)
    info = company.info
    financials_data.append({
        'ticker': ticker,
        'market_cap': info.get('marketCap', 0),
        'revenue': info.get('totalRevenue', 0),
        'ebitda': info.get('ebitda', 0),
        'enterprise_value': info.get('enterpriseValue', 0)
    })

df = pd.DataFrame(financials_data)
df.to_csv('data/sample_data/company_financials.csv', index=False)