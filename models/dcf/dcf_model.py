# models/dcf/dcf_model.py
"""
NextEra Energy (NEE) DCF Valuation Model
Professional-grade DCF analysis for renewable energy infrastructure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DCFModel:
    def __init__(self, ticker="NEE"):
        """Initialize DCF model with company data"""
        self.ticker = ticker
        self.load_data()
        self.set_assumptions()
        
    def load_data(self):
        """Load company financial data"""
        try:
            # Load your CSV data
            df = pd.read_csv('data/sample_data/company_financials.csv')
            company_data = df[df['ticker'] == self.ticker].iloc[0]
            
            # Extract key metrics (convert to millions for easier reading)
            self.market_cap = company_data['market_cap'] / 1e6
            self.enterprise_value = company_data['enterprise_value'] / 1e6  
            self.revenue_ttm = company_data['revenue'] / 1e6
            self.ebitda_ttm = company_data['ebitda'] / 1e6
            
            print(f" Loaded data for {self.ticker}")
            print(f"   Revenue (TTM): ${self.revenue_ttm:,.0f}M")
            print(f"   EBITDA (TTM):  ${self.ebitda_ttm:,.0f}M")
            print(f"   Market Cap:    ${self.market_cap:,.0f}M")
            
        except Exception as e:
            print(f" Error loading data: {e}")
            # Use NextEra Energy defaults if CSV fails
            self.market_cap = 143677  # From your data
            self.enterprise_value = 247600
            self.revenue_ttm = 25900
            self.ebitda_ttm = 14643
            print("Using fallback NEE data")
    
    def set_assumptions(self):
        """Set DCF modeling assumptions"""
        
        # Revenue Growth (Renewable energy sector)
        self.revenue_growth = [0.08, 0.07, 0.06, 0.05, 0.04]  # 8% to 4% over 5 years
        
        # EBITDA Margin (Based on current margin + efficiency improvements)
        current_margin = self.ebitda_ttm / self.revenue_ttm
        self.ebitda_margins = [current_margin + 0.005 * i for i in range(5)]  # Gradual improvement
        
        # Capital Expenditure (% of revenue - heavy for renewable infrastructure)
        self.capex_pct_revenue = [0.12, 0.11, 0.10, 0.09, 0.08]  # Decreasing as growth matures
        
        # Working Capital (% of revenue)
        self.wc_pct_revenue = 0.02  # Low for utilities
        
        # Tax Rate
        self.tax_rate = 0.25  # Corporate tax rate
        
        # WACC Components
        self.risk_free_rate = 0.042  # 10Y Treasury (~4.2%)
        self.market_risk_premium = 0.06  # Historical equity risk premium
        self.beta = 0.8  # Utilities typically have low beta
        self.cost_of_equity = self.risk_free_rate + self.beta * self.market_risk_premium
        
        # Debt assumptions
        self.cost_of_debt = 0.045  # Based on utility debt spreads
        self.debt_to_equity = 1.2  # Utilities are leveraged
        self.wacc = self.calculate_wacc()
        
        # Terminal Value
        self.terminal_growth_rate = 0.025  # 2.5% long-term growth
        
        print(f"\n Key Assumptions:")
        print(f"   WACC: {self.wacc:.1%}")
        print(f"   Terminal Growth: {self.terminal_growth_rate:.1%}")
        print(f"   Revenue Growth (Yr 1): {self.revenue_growth[0]:.1%}")
        
    def calculate_wacc(self):
        """Calculate Weighted Average Cost of Capital"""
        
        # Weight calculations
        debt_weight = self.debt_to_equity / (1 + self.debt_to_equity)
        equity_weight = 1 / (1 + self.debt_to_equity)
        
        # WACC formula
        wacc = (equity_weight * self.cost_of_equity + 
                debt_weight * self.cost_of_debt * (1 - self.tax_rate))
        
        return wacc
    
    def project_financials(self):
        """Project 5-year financial statements"""
        
        years = list(range(2024, 2029))  # 5-year projection
        projections = []
        
        # Base year (TTM)
        base_revenue = self.revenue_ttm
        
        for i, year in enumerate(years):
            
            # Revenue projection
            if i == 0:
                revenue = base_revenue * (1 + self.revenue_growth[i])
            else:
                revenue = projections[i-1]['revenue'] * (1 + self.revenue_growth[i])
            
            # EBITDA calculation
            ebitda = revenue * self.ebitda_margins[i]
            
            # Depreciation (estimated as % of revenue for capital-intensive business)
            depreciation = revenue * 0.08
            
            # EBIT
            ebit = ebitda - depreciation
            
            # Taxes
            taxes = ebit * self.tax_rate
            
            # NOPAT (Net Operating Profit After Tax)
            nopat = ebit - taxes
            
            # Capital Expenditure
            capex = revenue * self.capex_pct_revenue[i]
            
            # Working Capital Change
            if i == 0:
                wc_change = (revenue * self.wc_pct_revenue) - (base_revenue * self.wc_pct_revenue)
            else:
                current_wc = revenue * self.wc_pct_revenue
                prior_wc = projections[i-1]['revenue'] * self.wc_pct_revenue
                wc_change = current_wc - prior_wc
            
            # Unlevered Free Cash Flow
            fcf = nopat + depreciation - capex - wc_change
            
            # Present Value of FCF
            pv_fcf = fcf / ((1 + self.wacc) ** (i + 1))
            
            projections.append({
                'year': year,
                'revenue': revenue,
                'ebitda': ebitda,
                'ebitda_margin': ebitda / revenue,
                'depreciation': depreciation,
                'ebit': ebit,
                'taxes': taxes,
                'nopat': nopat,
                'capex': capex,
                'wc_change': wc_change,
                'fcf': fcf,
                'pv_fcf': pv_fcf
            })
        
        self.projections = pd.DataFrame(projections)
        return self.projections
    
    def calculate_terminal_value(self):
        """Calculate terminal value using Gordon Growth Model"""
        
        # Terminal year FCF
        terminal_fcf = self.projections.iloc[-1]['fcf'] * (1 + self.terminal_growth_rate)
        
        # Terminal value
        terminal_value = terminal_fcf / (self.wacc - self.terminal_growth_rate)
        
        # Present value of terminal value
        pv_terminal_value = terminal_value / ((1 + self.wacc) ** 5)
        
        self.terminal_value = terminal_value
        self.pv_terminal_value = pv_terminal_value
        
        return pv_terminal_value
    
    def calculate_valuation(self):
        """Calculate enterprise and equity value"""
        
        # Project financials
        projections = self.project_financials()
        
        # Calculate terminal value
        pv_terminal_value = self.calculate_terminal_value()
        
        # Sum of PV of FCF
        sum_pv_fcf = projections['pv_fcf'].sum()
        
        # Enterprise Value
        enterprise_value = sum_pv_fcf + pv_terminal_value
        
        # Assume net debt (you could enhance this with actual balance sheet data)
        net_debt = self.enterprise_value - self.market_cap  # Approximate
        
        # Equity Value
        equity_value = enterprise_value - net_debt
        
        # Current share price and implied share price
        # (You'd need shares outstanding data for this - using approximation)
        current_price_proxy = self.market_cap / 1000  # Rough approximation
        implied_price = equity_value / 1000
        
        # Upside/Downside
        upside_downside = (implied_price - current_price_proxy) / current_price_proxy
        
        self.valuation_summary = {
            'sum_pv_fcf': sum_pv_fcf,
            'pv_terminal_value': pv_terminal_value,
            'enterprise_value': enterprise_value,
            'net_debt': net_debt,
            'equity_value': equity_value,
            'current_market_cap': self.market_cap,
            'upside_downside': upside_downside
        }
        
        return self.valuation_summary
    
    def sensitivity_analysis(self):
        """Perform sensitivity analysis on key variables"""
        
        # Define sensitivity ranges
        wacc_range = np.arange(self.wacc - 0.01, self.wacc + 0.015, 0.005)
        terminal_growth_range = np.arange(self.terminal_growth_rate - 0.01, 
                                        self.terminal_growth_rate + 0.015, 0.005)
        
        sensitivity_results = []
        
        for wacc in wacc_range:
            for terminal_growth in terminal_growth_range:
                # Temporarily update assumptions
                original_wacc = self.wacc
                original_terminal = self.terminal_growth_rate
                
                self.wacc = wacc
                self.terminal_growth_rate = terminal_growth
                
                # Recalculate valuation
                valuation = self.calculate_valuation()
                
                sensitivity_results.append({
                    'wacc': wacc,
                    'terminal_growth': terminal_growth,
                    'equity_value': valuation['equity_value'],
                    'upside_downside': valuation['upside_downside']
                })
                
                # Restore original assumptions
                self.wacc = original_wacc
                self.terminal_growth_rate = original_terminal
        
        self.sensitivity_df = pd.DataFrame(sensitivity_results)
        return self.sensitivity_df
    
    def generate_report(self):
        """Generate comprehensive DCF analysis report"""
        
        print("\n" + "="*60)
        print(f"         {self.ticker} DCF VALUATION ANALYSIS")
        print("="*60)
        
        # Run full analysis
        projections = self.project_financials()
        valuation = self.calculate_valuation()
        
        # Display projections
        print(f"\n FINANCIAL PROJECTIONS ($ Millions)")
        print("-" * 60)
        display_cols = ['year', 'revenue', 'ebitda', 'ebitda_margin', 'fcf', 'pv_fcf']
        proj_display = projections[display_cols].copy()
        proj_display['ebitda_margin'] = proj_display['ebitda_margin'].apply(lambda x: f"{x:.1%}")
        
        for _, row in proj_display.iterrows():
            print(f"{row['year']}: Revenue ${row['revenue']:,.0f}M | "
                  f"EBITDA ${row['ebitda']:,.0f}M ({row['ebitda_margin']}) | "
                  f"FCF ${row['fcf']:,.0f}M | PV ${row['pv_fcf']:,.0f}M")
        
        # Display valuation
        print(f"\n VALUATION SUMMARY")
        print("-" * 40)
        print(f"Sum of PV of FCF (2024-2028):    ${valuation['sum_pv_fcf']:,.0f}M")
        print(f"PV of Terminal Value:           ${valuation['pv_terminal_value']:,.0f}M")
        print(f"Enterprise Value:               ${valuation['enterprise_value']:,.0f}M")
        print(f"Less: Net Debt:                 ${valuation['net_debt']:,.0f}M")
        print(f"Equity Value:                   ${valuation['equity_value']:,.0f}M")
        print(f"Current Market Cap:             ${valuation['current_market_cap']:,.0f}M")
        print(f"\n INVESTMENT RECOMMENDATION")
        print(f"Upside/Downside:                {valuation['upside_downside']:+.1%}")
        
        if valuation['upside_downside'] > 0.15:
            recommendation = "STRONG BUY"
        elif valuation['upside_downside'] > 0.05:
            recommendation = "BUY"
        elif valuation['upside_downside'] > -0.05:
            recommendation = "HOLD"
        else:
            recommendation = "SELL"
            
        print(f"Recommendation:                 {recommendation}")
        
        return valuation
    
    def create_visualizations(self):
        """Create charts for the DCF analysis"""
        
        # Create subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.ticker} DCF Analysis - Key Metrics', fontsize=16, fontweight='bold')
        
        # 1. Revenue and EBITDA Growth
        ax1.plot(self.projections['year'], self.projections['revenue'], 
                marker='o', linewidth=2, label='Revenue', color='#2E86AB')
        ax1.plot(self.projections['year'], self.projections['ebitda'], 
                marker='s', linewidth=2, label='EBITDA', color='#A23B72')
        ax1.set_title('Revenue & EBITDA Projections')
        ax1.set_ylabel('$ Millions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. EBITDA Margins
        ax2.bar(self.projections['year'], self.projections['ebitda_margin'], 
               color='#F18F01', alpha=0.7)
        ax2.set_title('EBITDA Margin Trend')
        ax2.set_ylabel('EBITDA Margin %')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax2.grid(True, alpha=0.3)
        
        # 3. Free Cash Flow
        ax3.bar(self.projections['year'], self.projections['fcf'], 
               color='#C73E1D', alpha=0.7)
        ax3.set_title('Unlevered Free Cash Flow')
        ax3.set_ylabel('FCF ($ Millions)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Valuation Breakdown
        valuation_components = ['PV of FCF', 'PV Terminal Value']
        valuation_values = [self.valuation_summary['sum_pv_fcf'], 
                           self.valuation_summary['pv_terminal_value']]
        
        ax4.pie(valuation_values, labels=valuation_components, autopct='%1.1f%%', 
               colors=['#2E86AB', '#A23B72'])
        ax4.set_title('Enterprise Value Components')
        
        plt.tight_layout()
        plt.show()
        
        return fig

def main():
    """Main function to run DCF analysis"""
    
    print(" Starting DCF Analysis...")
    
    # Initialize and run DCF model
    dcf = DCFModel("NEE")
    
    # Generate comprehensive report
    valuation_results = dcf.generate_report()
    
    # Create visualizations
    dcf.create_visualizations()
    
    # Optional: Run sensitivity analysis
    print(f"\n Running sensitivity analysis...")
    sensitivity = dcf.sensitivity_analysis()
    
    print(f"\n DCF Analysis Complete!")
    print(f"Enterprise Value Range: ${sensitivity['equity_value'].min():,.0f}M - ${sensitivity['equity_value'].max():,.0f}M")
    
    return dcf

if __name__ == "__main__":
    # Run the analysis
    nee_dcf = main()