# models/comps/comps_analysis.py
"""
Comparable Company Analysis - Utilities Sector
Professional-grade trading multiples analysis for renewable energy utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ComparableAnalysis:
    def __init__(self, target_ticker="NEE"):
        """Initialise comparable analysis with target company and peer universe"""
        self.target_ticker = target_ticker
        self.setup_peer_universe()
        self.load_financial_data()
        
    def setup_peer_universe(self):
        """Define peer company universe for analysis"""
        
        # Utility peer companies with renewable/clean energy focus
        self.peer_universe = {
            'NEE': {
                'name': 'NextEra Energy',
                'category': 'Renewable Leader',
                'description': 'Largest renewable energy generator'
            },
            'SO': {
                'name': 'Southern Company', 
                'category': 'Regulated Utility',
                'description': 'Southeast regulated utility'
            },
            'DUK': {
                'name': 'Duke Energy',
                'category': 'Regulated Utility', 
                'description': 'Carolinas regulated utility'
            },
            'XEL': {
                'name': 'Xcel Energy',
                'category': 'Clean Energy Focus',
                'description': 'Midwest utility with clean energy goals'
            },
            'AEP': {
                'name': 'American Electric Power',
                'category': 'Regulated Utility',
                'description': 'Multi-state regulated utility'
            },
            'EXC': {
                'name': 'Exelon Corporation',
                'category': 'Nuclear/Regulated',
                'description': 'Nuclear and regulated utility'
            },
            'SRE': {
                'name': 'Sempra Energy',
                'category': 'Gas/Electric Utility',
                'description': 'California-based utility'
            },
            'D': {
                'name': 'Dominion Energy',
                'category': 'Regulated Utility',
                'description': 'Virginia-based regulated utility'
            }
        }
        
        self.tickers = list(self.peer_universe.keys())
        print(f"Peer Universe: {len(self.tickers)} utilities including target {self.target_ticker}")
        
    def load_financial_data(self):
        """Load financial data for all companies in peer universe"""
        
        print("Loading financial data for peer companies...")
        
        company_data = []
        successful_loads = 0
        
        for ticker in self.tickers:
            try:
                # Get company info from yfinance
                company = yf.Ticker(ticker)
                info = company.info
                
                # Extract key financial metrics
                financial_data = {
                    'ticker': ticker,
                    'company_name': self.peer_universe[ticker]['name'],
                    'category': self.peer_universe[ticker]['category'],
                    'market_cap': info.get('marketCap', 0) / 1e6,  # Convert to millions
                    'enterprise_value': info.get('enterpriseValue', 0) / 1e6,
                    'revenue_ttm': info.get('totalRevenue', 0) / 1e6,
                    'ebitda_ttm': info.get('ebitda', 0) / 1e6,
                    'net_income_ttm': info.get('netIncomeToCommon', 0) / 1e6,
                    'total_debt': info.get('totalDebt', 0) / 1e6,
                    'cash': info.get('totalCash', 0) / 1e6,
                    'book_value': info.get('totalStockholderEquity', 0) / 1e6,
                    'shares_outstanding': info.get('sharesOutstanding', 0) / 1e6,
                    'current_price': info.get('currentPrice', 0),
                    'beta': info.get('beta', 1.0),
                    'dividend_yield': info.get('dividendYield', 0) if info.get('dividendYield') else 0,
                    'roe': info.get('returnOnEquity', 0) if info.get('returnOnEquity') else 0
                }
                
                company_data.append(financial_data)
                successful_loads += 1
                print(f" Loaded data for {ticker} - {self.peer_universe[ticker]['name']}")
                
            except Exception as e:
                print(f" Error loading {ticker}: {e}")
                continue
        
        # Convert to DataFrame
        self.company_data = pd.DataFrame(company_data)
        
        # Calculate trading multiples
        self.calculate_multiples()
        
        print(f"\n Successfully loaded data for {successful_loads}/{len(self.tickers)} companies")
        
    def calculate_multiples(self):
        """Calculate key trading multiples for all companies"""
        
        df = self.company_data.copy()
        
        # Valuation Multiples
        df['ev_revenue'] = np.where(df['revenue_ttm'] > 0, df['enterprise_value'] / df['revenue_ttm'], np.nan)
        df['ev_ebitda'] = np.where(df['ebitda_ttm'] > 0, df['enterprise_value'] / df['ebitda_ttm'], np.nan)
        df['p_e'] = np.where(df['net_income_ttm'] > 0, df['market_cap'] / df['net_income_ttm'], np.nan)
        df['p_b'] = np.where(df['book_value'] > 0, df['market_cap'] / df['book_value'], np.nan)
        
        # Financial Metrics
        df['debt_to_ebitda'] = np.where(df['ebitda_ttm'] > 0, df['total_debt'] / df['ebitda_ttm'], np.nan)
        df['net_debt'] = df['total_debt'] - df['cash']
        df['net_debt_to_ebitda'] = np.where(df['ebitda_ttm'] > 0, df['net_debt'] / df['ebitda_ttm'], np.nan)
        df['ebitda_margin'] = np.where(df['revenue_ttm'] > 0, df['ebitda_ttm'] / df['revenue_ttm'], np.nan)
        df['net_margin'] = np.where(df['revenue_ttm'] > 0, df['net_income_ttm'] / df['revenue_ttm'], np.nan)
        
        # Quality Metrics
        df['roic_proxy'] = np.where((df['total_debt'] + df['book_value']) > 0, 
                                   df['ebitda_ttm'] / (df['total_debt'] + df['book_value']), np.nan)
        
        self.company_data = df
        
        print(" Calculated trading multiples and financial metrics")
        
    def generate_peer_analysis(self):
        """Generate comprehensive peer analysis"""
        
        # Key multiples for analysis
        key_multiples = ['ev_revenue', 'ev_ebitda', 'p_e', 'p_b']
        key_metrics = ['ebitda_margin', 'net_margin', 'roe', 'dividend_yield', 'debt_to_ebitda']
        
        # Statistical analysis
        stats_summary = {}
        
        for multiple in key_multiples + key_metrics:
            if multiple in self.company_data.columns:
                data = self.company_data[multiple].dropna()
                if len(data) > 0:
                    stats_summary[multiple] = {
                        'count': len(data),
                        'mean': data.mean(),
                        'median': data.median(),
                        'min': data.min(),
                        'max': data.max(),
                        '25th_percentile': data.quantile(0.25),
                        '75th_percentile': data.quantile(0.75),
                        'std': data.std()
                    }
        
        self.peer_stats = stats_summary
        return stats_summary
    
    def analyze_target_vs_peers(self):
        """Analyse target company relative to peer universe"""
        
        target_data = self.company_data[self.company_data['ticker'] == self.target_ticker]
        
        if len(target_data) == 0:
            print(f" Target company {self.target_ticker} not found in dataset")
            return None
            
        target_metrics = target_data.iloc[0]
        
        # Compare target to peer statistics
        comparison_analysis = {}
        
        key_multiples = ['ev_revenue', 'ev_ebitda', 'p_e', 'p_b']
        
        for multiple in key_multiples:
            if multiple in self.peer_stats:
                target_value = target_metrics[multiple]
                peer_median = self.peer_stats[multiple]['median']
                peer_mean = self.peer_stats[multiple]['mean']
                
                if pd.notna(target_value) and pd.notna(peer_median):
                    premium_to_median = (target_value - peer_median) / peer_median
                    premium_to_mean = (target_value - peer_mean) / peer_mean
                    
                    # Percentile ranking
                    peer_values = self.company_data[multiple].dropna()
                    percentile = (peer_values < target_value).sum() / len(peer_values)
                    
                    comparison_analysis[multiple] = {
                        'target_value': target_value,
                        'peer_median': peer_median,
                        'peer_mean': peer_mean,
                        'premium_to_median': premium_to_median,
                        'premium_to_mean': premium_to_mean,
                        'percentile_ranking': percentile
                    }
        
        self.target_analysis = comparison_analysis
        return comparison_analysis
    
    def generate_valuation_ranges(self):
        """Generate valuation ranges using peer multiples"""
        
        target_data = self.company_data[self.company_data['ticker'] == self.target_ticker].iloc[0]
        
        valuation_ranges = {}
        
        # EV/Revenue multiple range
        if 'ev_revenue' in self.peer_stats and pd.notna(target_data['revenue_ttm']):
            ev_rev_stats = self.peer_stats['ev_revenue']
            
            valuation_ranges['ev_revenue'] = {
                'target_revenue': target_data['revenue_ttm'],
                'low_multiple': ev_rev_stats['25th_percentile'],
                'median_multiple': ev_rev_stats['median'],
                'high_multiple': ev_rev_stats['75th_percentile'],
                'low_ev': target_data['revenue_ttm'] * ev_rev_stats['25th_percentile'],
                'median_ev': target_data['revenue_ttm'] * ev_rev_stats['median'],
                'high_ev': target_data['revenue_ttm'] * ev_rev_stats['75th_percentile'],
                'current_ev': target_data['enterprise_value']
            }
        
        # EV/EBITDA multiple range
        if 'ev_ebitda' in self.peer_stats and pd.notna(target_data['ebitda_ttm']):
            ev_ebitda_stats = self.peer_stats['ev_ebitda']
            
            valuation_ranges['ev_ebitda'] = {
                'target_ebitda': target_data['ebitda_ttm'],
                'low_multiple': ev_ebitda_stats['25th_percentile'],
                'median_multiple': ev_ebitda_stats['median'],
                'high_multiple': ev_ebitda_stats['75th_percentile'],
                'low_ev': target_data['ebitda_ttm'] * ev_ebitda_stats['25th_percentile'],
                'median_ev': target_data['ebitda_ttm'] * ev_ebitda_stats['median'],
                'high_ev': target_data['ebitda_ttm'] * ev_ebitda_stats['75th_percentile'],
                'current_ev': target_data['enterprise_value']
            }
        
        # P/E multiple range
        if 'p_e' in self.peer_stats and pd.notna(target_data['net_income_ttm']):
            pe_stats = self.peer_stats['p_e']
            
            valuation_ranges['p_e'] = {
                'target_earnings': target_data['net_income_ttm'],
                'low_multiple': pe_stats['25th_percentile'],
                'median_multiple': pe_stats['median'],
                'high_multiple': pe_stats['75th_percentile'],
                'low_market_cap': target_data['net_income_ttm'] * pe_stats['25th_percentile'],
                'median_market_cap': target_data['net_income_ttm'] * pe_stats['median'],
                'high_market_cap': target_data['net_income_ttm'] * pe_stats['75th_percentile'],
                'current_market_cap': target_data['market_cap']
            }
        
        self.valuation_ranges = valuation_ranges
        return valuation_ranges
    
    def create_peer_visualizations(self):
        """Create comprehensive peer analysis charts"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. EV/EBITDA Multiple Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ev_ebitda_data = self.company_data.dropna(subset=['ev_ebitda'])
        bars = ax1.bar(ev_ebitda_data['ticker'], ev_ebitda_data['ev_ebitda'], 
                      color=['#1f77b4' if x == self.target_ticker else 'lightblue' for x in ev_ebitda_data['ticker']])
        ax1.set_title('EV/EBITDA Multiples')
        ax1.set_ylabel('EV/EBITDA (x)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Highlight target company
        if self.target_ticker in ev_ebitda_data['ticker'].values:
            target_idx = ev_ebitda_data['ticker'].tolist().index(self.target_ticker)
            bars[target_idx].set_color('#ff7f0e')
        
        # 2. P/E Multiple Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        pe_data = self.company_data.dropna(subset=['p_e'])
        # Filter out extreme outliers (P/E > 50)
        pe_data_filtered = pe_data[pe_data['p_e'] <= 50]
        bars2 = ax2.bar(pe_data_filtered['ticker'], pe_data_filtered['p_e'],
                       color=['#ff7f0e' if x == self.target_ticker else '#ffcc99' for x in pe_data_filtered['ticker']])
        ax2.set_title('P/E Multiples')
        ax2.set_ylabel('P/E (x)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. EBITDA Margin Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        margin_data = self.company_data.dropna(subset=['ebitda_margin'])
        bars3 = ax3.bar(margin_data['ticker'], margin_data['ebitda_margin'] * 100,
                       color=['#2ca02c' if x == self.target_ticker else '#98df8a' for x in margin_data['ticker']])
        ax3.set_title('EBITDA Margins')
        ax3.set_ylabel('EBITDA Margin (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Market Cap vs Enterprise Value Scatter
        ax4 = fig.add_subplot(gs[1, :2])
        scatter_data = self.company_data.dropna(subset=['market_cap', 'enterprise_value'])
        colors = ['#ff7f0e' if x == self.target_ticker else '#1f77b4' for x in scatter_data['ticker']]
        sizes = [100 if x == self.target_ticker else 60 for x in scatter_data['ticker']]
        
        ax4.scatter(scatter_data['market_cap'], scatter_data['enterprise_value'], 
                   c=colors, s=sizes, alpha=0.7)
        
        # Add company labels
        for idx, row in scatter_data.iterrows():
            ax4.annotate(row['ticker'], 
                        (row['market_cap'], row['enterprise_value']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
        
        ax4.set_title('Market Cap vs Enterprise Value')
        ax4.set_xlabel('Market Cap ($M)')
        ax4.set_ylabel('Enterprise Value ($M)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Dividend Yield vs ROE
        ax5 = fig.add_subplot(gs[1, 2])
        div_roe_data = self.company_data.dropna(subset=['dividend_yield', 'roe'])
        colors = ['#ff7f0e' if x == self.target_ticker else '#1f77b4' for x in div_roe_data['ticker']]
        sizes = [100 if x == self.target_ticker else 60 for x in div_roe_data['ticker']]
        
        ax5.scatter(div_roe_data['dividend_yield'] * 100, div_roe_data['roe'] * 100,
                   c=colors, s=sizes, alpha=0.7)
        
        for idx, row in div_roe_data.iterrows():
            ax5.annotate(row['ticker'], 
                        (row['dividend_yield'] * 100, row['roe'] * 100),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
        
        ax5.set_title('Dividend Yield vs ROE')
        ax5.set_xlabel('Dividend Yield (%)')
        ax5.set_ylabel('ROE (%)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Leverage Analysis
        ax6 = fig.add_subplot(gs[2, :])
        leverage_data = self.company_data.dropna(subset=['debt_to_ebitda'])
        bars6 = ax6.bar(leverage_data['ticker'], leverage_data['debt_to_ebitda'],
                       color=['#d62728' if x == self.target_ticker else '#ff9999' for x in leverage_data['ticker']])
        ax6.set_title('Debt-to-EBITDA Ratios')
        ax6.set_ylabel('Debt/EBITDA (x)')
        ax6.tick_params(axis='x', rotation=45)
        ax6.axhline(y=4.0, color='red', linestyle='--', alpha=0.7, label='High Leverage (4x)')
        ax6.legend()
        
        plt.suptitle(f'{self.target_ticker} vs Utility Sector Peers - Trading Analysis', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def generate_comps_report(self):
        """Generate comprehensive comparable analysis report"""
        
        print("\n" + "="*80)
        print(f"         COMPARABLE COMPANY ANALYSIS - {self.target_ticker}")
        print("="*80)
        
        # Generate analysis
        peer_stats = self.generate_peer_analysis()
        target_analysis = self.analyze_target_vs_peers()
        valuation_ranges = self.generate_valuation_ranges()
        
        # Display peer universe
        print(f"\n PEER UNIVERSE ({len(self.company_data)} companies)")
        print("-" * 60)
        for _, company in self.company_data.iterrows():
            print(f"{company['ticker']:>4}: {company['company_name']:<25} ({company['category']})")
        
        # Display key statistics
        print(f"\n SECTOR TRADING MULTIPLES")
        print("-" * 60)
        
        multiples_display = ['ev_revenue', 'ev_ebitda', 'p_e', 'p_b']
        multiple_names = ['EV/Revenue', 'EV/EBITDA', 'P/E Ratio', 'Price/Book']
        
        for multiple, name in zip(multiples_display, multiple_names):
            if multiple in peer_stats:
                stats = peer_stats[multiple]
                print(f"{name:>12}: {stats['median']:>6.1f}x median | "
                      f"{stats['25th_percentile']:>6.1f}x - {stats['75th_percentile']:>6.1f}x range | "
                      f"{stats['count']:>2} companies")
        
        # Target company analysis
        if target_analysis:
            print(f"\n {self.target_ticker} RELATIVE VALUATION")
            print("-" * 60)
            
            target_data = self.company_data[self.company_data['ticker'] == self.target_ticker].iloc[0]
            
            print(f"Company: {target_data['company_name']}")
            print(f"Category: {target_data['category']}")
            print(f"Market Cap: ${target_data['market_cap']:,.0f}M")
            print(f"Enterprise Value: ${target_data['enterprise_value']:,.0f}M")
            print(f"Revenue (TTM): ${target_data['revenue_ttm']:,.0f}M")
            print(f"EBITDA (TTM): ${target_data['ebitda_ttm']:,.0f}M")
            
            print(f"\nMULTIPLE ANALYSIS:")
            for multiple, analysis in target_analysis.items():
                name = {'ev_revenue': 'EV/Revenue', 'ev_ebitda': 'EV/EBITDA', 
                       'p_e': 'P/E Ratio', 'p_b': 'Price/Book'}.get(multiple, multiple)
                
                print(f"{name:>12}: {analysis['target_value']:>6.1f}x | "
                      f"Peer Median: {analysis['peer_median']:>6.1f}x | "
                      f"Premium: {analysis['premium_to_median']:>+6.1%} | "
                      f"Percentile: {analysis['percentile_ranking']:>6.1%}")
        
        # Valuation ranges
        if valuation_ranges:
            print(f"\n PEER-BASED VALUATION RANGES")
            print("-" * 60)
            
            for method, ranges in valuation_ranges.items():
                if 'ev' in method:
                    method_name = method.replace('_', '/').upper()
                    current_val = ranges['current_ev']
                    low_val = ranges['low_ev']
                    median_val = ranges['median_ev']
                    high_val = ranges['high_ev']
                    
                    print(f"\n{method_name} Method:")
                    print(f"  Current EV: ${current_val:,.0f}M")
                    print(f"  Peer Range: ${low_val:,.0f}M - ${high_val:,.0f}M")
                    print(f"  Peer Median: ${median_val:,.0f}M")
                    
                    if current_val > 0:
                        median_premium = (current_val - median_val) / median_val
                        print(f"  Premium to Median: {median_premium:+.1%}")
        
        # Investment perspective
        print(f"\n INVESTMENT IMPLICATIONS")
        print("-" * 40)
        
        target_data = self.company_data[self.company_data['ticker'] == self.target_ticker].iloc[0]
        
        # Quality metrics
        print(f"Quality Metrics:")
        print(f"  EBITDA Margin: {target_data['ebitda_margin']:.1%}")
        print(f"  ROE: {target_data['roe']:.1%}")
        print(f"  Dividend Yield: {target_data['dividend_yield']:.1%}")
        print(f"  Debt/EBITDA: {target_data['debt_to_ebitda']:.1f}x")
        
        # Relative positioning
        if target_analysis and 'ev_ebitda' in target_analysis:
            ev_ebitda_premium = target_analysis['ev_ebitda']['premium_to_median']
            ev_ebitda_percentile = target_analysis['ev_ebitda']['percentile_ranking']
            
            if ev_ebitda_premium > 0.15:  # >15% premium
                valuation_assessment = "PREMIUM VALUATION"
                reasoning = "Trading above peer median, requires justification"
            elif ev_ebitda_premium > -0.15:  # Within ±15%
                valuation_assessment = "FAIR VALUATION"
                reasoning = "In-line with peer group multiples"
            else:
                valuation_assessment = "DISCOUNT VALUATION"
                reasoning = "Trading below peer median, potential opportunity"
            
            print(f"\nValuation Assessment: {valuation_assessment}")
            print(f"Reasoning: {reasoning}")
            print(f"Peer Percentile: {ev_ebitda_percentile:.0%}")
        
        return {
            'peer_stats': peer_stats,
            'target_analysis': target_analysis,
            'valuation_ranges': valuation_ranges
        }

def main():
    """Main function to run comparable analysis"""
    
    print(" Starting Comparable Company Analysis...")
    
    # Initialize comparable analysis for NextEra Energy
    comps = ComparableAnalysis("NEE")
    
    # Generate comprehensive report
    results = comps.generate_comps_report()
    
    # Create visualizations
    comps.create_peer_visualizations()
    
    print(f"\n Comparable Analysis Complete!")
    print(f" Analys ed {len(comps.company_data)} utility companies")
    
    return comps

if __name__ == "__main__":
    # Run the comparable analysis
    nee_comps = main()