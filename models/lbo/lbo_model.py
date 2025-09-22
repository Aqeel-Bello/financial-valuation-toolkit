# models/lbo/lbo_model.py
"""
LBO Model - NextEra Energy Subsidiary Buyout
Professional-grade leveraged buyout analysis for renewable energy infrastructure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class LBOModel:
    def __init__(self, company_name="NEE Energy Subsidiary"):
        """Initialise LBO model with target company assumptions"""
        self.company_name = company_name
        self.set_deal_assumptions()
        self.set_operating_assumptions()
        self.set_financing_assumptions()
        
    def set_deal_assumptions(self):
        """Set transaction and deal structure assumptions"""
        
        # Transaction Details (Based on NEE subsidiary - assume 40% of NEE's business)
        self.transaction_date = "2024"
        self.revenue_ltm = 10500  # $10.5B (40% of NEE's ~$26B)
        self.ebitda_ltm = 5850   # $5.85B (40% of NEE's ~$14.6B) 
        
        # Purchase Price and Multiples
        self.entry_multiple = 12.0  # 12x EBITDA (typical for infrastructure)
        self.purchase_price = self.ebitda_ltm * self.entry_multiple  # $70.2B
        
        # Exit Assumptions (5-year hold)
        self.exit_multiple = 11.5  # Slightly lower exit multiple
        self.hold_period = 5  # Years
        
        # Management Assumptions
        self.mgmt_rollover_pct = 0.15  # 15% management rollover
        self.mgmt_option_pool = 0.05   # 5% option pool for management
        
        print(f"🏗️ LBO Deal Structure - {self.company_name}")
        print(f"   Revenue (LTM): ${self.revenue_ltm:,.0f}M")
        print(f"   EBITDA (LTM):  ${self.ebitda_ltm:,.0f}M")
        print(f"   Entry Multiple: {self.entry_multiple:.1f}x")
        print(f"   Purchase Price: ${self.purchase_price:,.0f}M")
        
    def set_operating_assumptions(self):
        """Set operating performance assumptions over hold period"""
        
        # Revenue Growth (Infrastructure growth - more stable than tech)
        self.revenue_growth = [0.06, 0.05, 0.04, 0.04, 0.03]  # 6% to 3%
        
        # EBITDA Margin Improvement (Operational improvements)
        base_margin = self.ebitda_ltm / self.revenue_ltm  # ~55.7%
        self.ebitda_margins = [base_margin + 0.005 * i for i in range(5)]  # Gradual improvement
        
        # Capex as % of Revenue (Infrastructure maintenance + growth)
        self.capex_pct_revenue = [0.10, 0.09, 0.08, 0.08, 0.07]  # Decreasing
        
        # Working Capital (% of Revenue - low for utilities)
        self.wc_pct_revenue = 0.01
        
        # Tax Rate
        self.tax_rate = 0.25
        
        # Depreciation (% of Revenue - capital intensive business)
        self.depreciation_pct = 0.08
        
        print(f"\n📈 Operating Assumptions:")
        print(f"   Revenue Growth (Yr 1): {self.revenue_growth[0]:.1%}")
        print(f"   EBITDA Margin (Base): {base_margin:.1%}")
        print(f"   Target EBITDA Margin: {self.ebitda_margins[-1]:.1%}")
        
    def set_financing_assumptions(self):
        """Set debt financing structure and terms"""
        
        # Sources of Funds
        self.sponsor_equity_pct = 0.35  # 35% equity (typical for infrastructure LBO)
        self.debt_pct = 1 - self.sponsor_equity_pct - self.mgmt_rollover_pct  # Remaining debt
        
        # Debt Structure (Multiple tranches typical in large LBOs)
        self.senior_debt_pct = 0.40    # 40% of purchase price
        self.subordinated_debt_pct = 0.10  # 10% of purchase price
        
        # Debt Terms
        self.senior_debt_rate = 0.055   # 5.5% (current market for infrastructure)
        self.sub_debt_rate = 0.085      # 8.5% (higher rate for sub debt)
        self.senior_debt_term = 7       # Years
        self.sub_debt_term = 8          # Years
        
        # Cash Sweep (% of excess cash used to pay down debt)
        self.cash_sweep_pct = 0.75      # 75% cash sweep
        self.minimum_cash = 500         # $500M minimum cash balance
        
        # Calculate debt amounts
        self.senior_debt_amount = self.purchase_price * self.senior_debt_pct
        self.sub_debt_amount = self.purchase_price * self.subordinated_debt_pct
        self.total_debt = self.senior_debt_amount + self.sub_debt_amount
        self.sponsor_equity = self.purchase_price * self.sponsor_equity_pct
        
        print(f"\n💰 Financing Structure:")
        print(f"   Purchase Price: ${self.purchase_price:,.0f}M")
        print(f"   Sponsor Equity: ${self.sponsor_equity:,.0f}M ({self.sponsor_equity_pct:.1%})")
        print(f"   Senior Debt:    ${self.senior_debt_amount:,.0f}M ({self.senior_debt_rate:.1%})")
        print(f"   Sub Debt:       ${self.sub_debt_amount:,.0f}M ({self.sub_debt_rate:.1%})")
        print(f"   Total Debt:     ${self.total_debt:,.0f}M")
        
    def build_operating_model(self):
        """Build 5-year operating projections"""
        
        years = list(range(2024, 2029))
        projections = []
        
        base_revenue = self.revenue_ltm
        
        for i, year in enumerate(years):
            
            # Revenue Growth
            if i == 0:
                revenue = base_revenue * (1 + self.revenue_growth[i])
            else:
                revenue = projections[i-1]['revenue'] * (1 + self.revenue_growth[i])
            
            # EBITDA
            ebitda = revenue * self.ebitda_margins[i]
            
            # Depreciation & Amortization
            depreciation = revenue * self.depreciation_pct
            
            # EBIT
            ebit = ebitda - depreciation
            
            # Interest Expense (calculated later in debt schedule)
            # Placeholder for now - will be updated in debt scheduling
            interest_expense = 0
            
            # EBT (Earnings Before Tax)
            ebt = ebit - interest_expense
            
            # Taxes
            taxes = max(0, ebt * self.tax_rate)  # No tax benefit if negative EBT
            
            # Net Income
            net_income = ebt - taxes
            
            # Cash Flow Calculations
            # Add back non-cash items
            cash_from_operations = net_income + depreciation
            
            # Capital Expenditure
            capex = revenue * self.capex_pct_revenue[i]
            
            # Working Capital Change
            if i == 0:
                wc_change = (revenue * self.wc_pct_revenue) - (base_revenue * self.wc_pct_revenue)
            else:
                current_wc = revenue * self.wc_pct_revenue
                prior_wc = projections[i-1]['revenue'] * self.wc_pct_revenue
                wc_change = current_wc - prior_wc
            
            # Free Cash Flow (before debt service)
            free_cash_flow = cash_from_operations - capex - wc_change
            
            projections.append({
                'year': year,
                'revenue': revenue,
                'ebitda': ebitda,
                'ebitda_margin': ebitda / revenue,
                'depreciation': depreciation,
                'ebit': ebit,
                'interest_expense': interest_expense,  # Will be updated
                'ebt': ebt,
                'taxes': taxes,
                'net_income': net_income,
                'capex': capex,
                'wc_change': wc_change,
                'free_cash_flow': free_cash_flow
            })
        
        self.operating_projections = pd.DataFrame(projections)
        return self.operating_projections
    
    def build_debt_schedule(self):
        """Build debt paydown schedule with cash sweep"""
        
        # Initialize debt balances
        senior_debt_balance = self.senior_debt_amount
        sub_debt_balance = self.sub_debt_amount
        total_debt_balance = senior_debt_balance + sub_debt_balance
        
        debt_schedule = []
        
        for i in range(5):  # 5-year projection
            year = 2024 + i
            
            # Beginning debt balances
            beg_senior_debt = senior_debt_balance
            beg_sub_debt = sub_debt_balance
            beg_total_debt = total_debt_balance
            
            # Interest calculations
            senior_interest = beg_senior_debt * self.senior_debt_rate
            sub_interest = beg_sub_debt * self.sub_debt_rate
            total_interest = senior_interest + sub_interest
            
            # Update operating projections with interest expense
            self.operating_projections.loc[i, 'interest_expense'] = total_interest
            self.operating_projections.loc[i, 'ebt'] = (
                self.operating_projections.loc[i, 'ebit'] - total_interest
            )
            self.operating_projections.loc[i, 'taxes'] = max(
                0, self.operating_projections.loc[i, 'ebt'] * self.tax_rate
            )
            self.operating_projections.loc[i, 'net_income'] = (
                self.operating_projections.loc[i, 'ebt'] - 
                self.operating_projections.loc[i, 'taxes']
            )
            
            # Cash available for debt service
            fcf_after_interest = self.operating_projections.loc[i, 'free_cash_flow'] - total_interest
            
            # Cash sweep calculation
            excess_cash = max(0, fcf_after_interest - self.minimum_cash)
            cash_for_paydown = excess_cash * self.cash_sweep_pct
            
            # Debt paydown priority: Senior debt first
            senior_paydown = min(cash_for_paydown, beg_senior_debt)
            remaining_cash = cash_for_paydown - senior_paydown
            sub_paydown = min(remaining_cash, beg_sub_debt)
            total_paydown = senior_paydown + sub_paydown
            
            # Ending debt balances
            senior_debt_balance = beg_senior_debt - senior_paydown
            sub_debt_balance = beg_sub_debt - sub_paydown
            total_debt_balance = senior_debt_balance + sub_debt_balance
            
            debt_schedule.append({
                'year': year,
                'beg_senior_debt': beg_senior_debt,
                'beg_sub_debt': beg_sub_debt,
                'beg_total_debt': beg_total_debt,
                'senior_interest': senior_interest,
                'sub_interest': sub_interest,
                'total_interest': total_interest,
                'fcf_after_interest': fcf_after_interest,
                'cash_for_paydown': cash_for_paydown,
                'senior_paydown': senior_paydown,
                'sub_paydown': sub_paydown,
                'total_paydown': total_paydown,
                'end_senior_debt': senior_debt_balance,
                'end_sub_debt': sub_debt_balance,
                'end_total_debt': total_debt_balance
            })
        
        self.debt_schedule = pd.DataFrame(debt_schedule)
        return self.debt_schedule
    
    def calculate_returns(self):
        """Calculate IRR and MOIC for different exit scenarios"""
        
        # Exit year financials (Year 5)
        exit_ebitda = self.operating_projections.iloc[-1]['ebitda']
        exit_debt = self.debt_schedule.iloc[-1]['end_total_debt']
        
        # Exit scenarios with different multiples
        exit_scenarios = [
            {'scenario': 'Downside', 'multiple': self.exit_multiple - 1.0},
            {'scenario': 'Base', 'multiple': self.exit_multiple},
            {'scenario': 'Upside', 'multiple': self.exit_multiple + 1.0}
        ]
        
        returns_analysis = []
        
        for scenario in exit_scenarios:
            exit_multiple = scenario['multiple']
            
            # Enterprise Value at exit
            exit_enterprise_value = exit_ebitda * exit_multiple
            
            # Equity Value (EV minus debt)
            exit_equity_value = exit_enterprise_value - exit_debt
            
            # Sponsor equity returns
            gross_equity_proceeds = exit_equity_value * (1 - self.mgmt_option_pool)
            
            # IRR Calculation (approximate using compound annual growth rate)
            initial_investment = self.sponsor_equity
            cagr = (gross_equity_proceeds / initial_investment) ** (1/self.hold_period) - 1
            
            # MOIC (Multiple of Invested Capital)
            moic = gross_equity_proceeds / initial_investment
            
            returns_analysis.append({
                'scenario': scenario['scenario'],
                'exit_multiple': exit_multiple,
                'exit_enterprise_value': exit_enterprise_value,
                'exit_debt': exit_debt,
                'exit_equity_value': exit_equity_value,
                'gross_proceeds': gross_equity_proceeds,
                'initial_investment': initial_investment,
                'irr': cagr,
                'moic': moic
            })
        
        self.returns_analysis = pd.DataFrame(returns_analysis)
        return self.returns_analysis
    
    def generate_lbo_report(self):
        """Generate comprehensive LBO analysis report"""
        
        print("\n" + "="*70)
        print(f"         {self.company_name.upper()} - LBO ANALYSIS")
        print("="*70)
        
        # Build complete model
        operating = self.build_operating_model()
        debt_sched = self.build_debt_schedule()
        returns = self.calculate_returns()
        
        # Sources & Uses
        print(f"\n💰 SOURCES & USES OF FUNDS")
        print("-" * 50)
        print("USES:")
        print(f"  Purchase Price:           ${self.purchase_price:,.0f}M")
        print(f"  Transaction Fees:         ${self.purchase_price * 0.02:,.0f}M")
        print(f"  Total Uses:               ${self.purchase_price * 1.02:,.0f}M")
        print("\nSOURCES:")
        print(f"  Sponsor Equity:           ${self.sponsor_equity:,.0f}M")
        print(f"  Senior Debt:              ${self.senior_debt_amount:,.0f}M")
        print(f"  Subordinated Debt:        ${self.sub_debt_amount:,.0f}M")
        print(f"  Management Rollover:      ${self.purchase_price * self.mgmt_rollover_pct:,.0f}M")
        print(f"  Total Sources:            ${self.purchase_price * 1.02:,.0f}M")
        
        # Operating Performance
        print(f"\n📈 PROJECTED OPERATING PERFORMANCE")
        print("-" * 50)
        for _, row in operating.iterrows():
            print(f"{int(row['year'])}: Revenue ${row['revenue']:,.0f}M | "
                  f"EBITDA ${row['ebitda']:,.0f}M ({row['ebitda_margin']:.1%}) | "
                  f"FCF ${row['free_cash_flow']:,.0f}M")
        
        # Debt Paydown
        print(f"\n💳 DEBT PAYDOWN SCHEDULE")
        print("-" * 50)
        for _, row in debt_sched.iterrows():
            print(f"{int(row['year'])}: Beginning Debt ${row['beg_total_debt']:,.0f}M | "
                  f"Interest ${row['total_interest']:,.0f}M | "
                  f"Paydown ${row['total_paydown']:,.0f}M | "
                  f"Ending Debt ${row['end_total_debt']:,.0f}M")
        
        # Returns Analysis
        print(f"\n🎯 RETURNS ANALYSIS (5-Year Hold)")
        print("-" * 50)
        for _, row in returns.iterrows():
            print(f"{row['scenario']:>8}: {row['exit_multiple']:.1f}x Exit | "
                  f"IRR {row['irr']:>6.1%} | "
                  f"MOIC {row['moic']:>4.1f}x | "
                  f"Proceeds ${row['gross_proceeds']:>7,.0f}M")
        
        # Investment Recommendation
        base_case_irr = returns[returns['scenario'] == 'Base']['irr'].iloc[0]
        base_case_moic = returns[returns['scenario'] == 'Base']['moic'].iloc[0]
        
        print(f"\n⭐ INVESTMENT RECOMMENDATION")
        print("-" * 30)
        print(f"Base Case IRR:        {base_case_irr:.1%}")
        print(f"Base Case MOIC:       {base_case_moic:.1f}x")
        
        if base_case_irr >= 0.20:  # 20%+ IRR target
            recommendation = "STRONG BUY - Attractive Returns"
        elif base_case_irr >= 0.15:
            recommendation = "BUY - Adequate Returns"
        else:
            recommendation = "PASS - Below Target Returns"
            
        print(f"Recommendation:       {recommendation}")
        
        return {
            'operating': operating,
            'debt_schedule': debt_sched,
            'returns': returns
        }
    
    def create_lbo_visualizations(self):
        """Create charts for LBO analysis"""
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.company_name} - LBO Analysis', fontsize=16, fontweight='bold')
        
        # 1. Revenue and EBITDA Growth
        ax1.plot(self.operating_projections['year'], self.operating_projections['revenue'], 
                marker='o', linewidth=3, label='Revenue', color='#1f77b4')
        ax1.plot(self.operating_projections['year'], self.operating_projections['ebitda'], 
                marker='s', linewidth=3, label='EBITDA', color='#ff7f0e')
        ax1.set_title('Revenue & EBITDA Projections')
        ax1.set_ylabel('$ Millions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Debt Paydown
        ax2.bar(self.debt_schedule['year'], self.debt_schedule['beg_total_debt'], 
               color='#d62728', alpha=0.7, label='Total Debt')
        ax2.set_title('Debt Paydown Schedule')
        ax2.set_ylabel('Debt Balance ($ Millions)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Free Cash Flow
        ax3.bar(self.operating_projections['year'], self.operating_projections['free_cash_flow'], 
               color='#2ca02c', alpha=0.7)
        ax3.set_title('Unlevered Free Cash Flow')
        ax3.set_ylabel('FCF ($ Millions)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Returns by Scenario
        scenarios = self.returns_analysis['scenario']
        irrs = self.returns_analysis['irr'] * 100  # Convert to percentage
        moics = self.returns_analysis['moic']
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        ax4_twin = ax4.twinx()
        bars1 = ax4.bar(x - width/2, irrs, width, label='IRR (%)', color='#9467bd', alpha=0.7)
        bars2 = ax4_twin.bar(x + width/2, moics, width, label='MOIC (x)', color='#8c564b', alpha=0.7)
        
        ax4.set_title('Returns Analysis by Scenario')
        ax4.set_ylabel('IRR (%)')
        ax4_twin.set_ylabel('MOIC (x)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(scenarios)
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        return fig

def main():
    """Main function to run LBO analysis"""
    
    print("🚀 Starting LBO Analysis - NextEra Energy Subsidiary...")
    
    # Initialize LBO model
    lbo = LBOModel("NEE Energy Infrastructure")
    
    # Generate comprehensive report
    results = lbo.generate_lbo_report()
    
    # Create visualizations
    lbo.create_lbo_visualizations()
    
    print(f"\n✅ LBO Analysis Complete!")
    
    return lbo

if __name__ == "__main__":
    # Run the LBO analysis
    nee_lbo = main()