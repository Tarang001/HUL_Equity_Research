import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the analysis parameters
class HULEquityAnalysis:
    def __init__(self):
        self.ticker = "HINDUNILVR.NS"  # NSE symbol for HUL
        self.company_name = "Hindustan Unilever Limited"
        
    def fetch_stock_data(self, period="2y"):
        """Fetch historical stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(self.ticker)
            hist_data = stock.history(period=period)
            info = stock.info
            return hist_data, info
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None, None
    
    def get_financial_metrics(self):
        """Extract key financial metrics"""
        stock = yf.Ticker(self.ticker)
        info = stock.info
        
        metrics = {
            'Market Cap (Cr)': info.get('marketCap', 0) / 1e7,  # Convert to crores
            'Enterprise Value (Cr)': info.get('enterpriseValue', 0) / 1e7,
            'P/E Ratio': info.get('trailingPE', 0),
            'P/B Ratio': info.get('priceToBook', 0),
            'Debt/Equity': info.get('debtToEquity', 0) / 100,
            'ROE (%)': info.get('returnOnEquity', 0) * 100,
            'Profit Margin (%)': info.get('profitMargins', 0) * 100,
            'Revenue (Cr)': info.get('totalRevenue', 0) / 1e7,
        }
        return metrics
    
    def simple_dcf_model(self, initial_fcf, growth_rates, discount_rate=0.10, terminal_growth=0.03):
        """
        Simple DCF model for HUL
        
        Parameters:
        - initial_fcf: Starting free cash flow (in crores)
        - growth_rates: List of growth rates for projection years
        - discount_rate: WACC (default 10%)
        - terminal_growth: Long-term growth rate (default 3%)
        """
        
        # Project cash flows
        projected_fcf = []
        fcf = initial_fcf
        
        for growth in growth_rates:
            fcf = fcf * (1 + growth)
            projected_fcf.append(fcf)
        
        # Calculate present values
        pv_fcf = []
        for i, cash_flow in enumerate(projected_fcf):
            pv = cash_flow / ((1 + discount_rate) ** (i + 1))
            pv_fcf.append(pv)
        
        # Terminal value
        terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / ((1 + discount_rate) ** len(growth_rates))
        
        # Enterprise value
        enterprise_value = sum(pv_fcf) + pv_terminal
        
        return {
            'Projected FCF': projected_fcf,
            'PV of FCF': pv_fcf,
            'Terminal Value': terminal_value,
            'PV of Terminal': pv_terminal,
            'Enterprise Value': enterprise_value,
            'Sum PV FCF': sum(pv_fcf)
        }
    
    def peer_comparison(self):
        """Compare HUL with peer companies"""
        peers = {
            'HUL': 'HINDUNILVR.NS',
            'Nestle': 'NESTLEIND.NS',
            'Dabur': 'DABUR.NS',
            'ITC': 'ITC.NS'
        }
        
        comparison_data = []
        
        for name, ticker in peers.items():
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                comparison_data.append({
                    'Company': name,
                    'Market Cap (₹Cr)': info.get('marketCap', 0) / 1e7,
                    'P/E Ratio': info.get('trailingPE', 0),
                    'P/B Ratio': info.get('priceToBook', 0),
                    'Profit Margin (%)': info.get('profitMargins', 0) * 100,
                    'ROE (%)': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
                })
            except:
                print(f"Could not fetch data for {name}")
        
        return pd.DataFrame(comparison_data)
    
    def plot_stock_performance(self, hist_data):
        """Plot stock price performance"""
        plt.figure(figsize=(12, 6))
        plt.plot(hist_data.index, hist_data['Close'], linewidth=2)
        plt.title(f'{self.company_name} - Stock Price Performance')
        plt.xlabel('Date')
        plt.ylabel('Price (₹)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def calculate_returns(self, hist_data):
        """Calculate various return metrics"""
        current_price = hist_data['Close'].iloc[-1]
        
        returns = {
            '1M Return (%)': ((current_price / hist_data['Close'].iloc[-22]) - 1) * 100,
            '3M Return (%)': ((current_price / hist_data['Close'].iloc[-66]) - 1) * 100,
            '6M Return (%)': ((current_price / hist_data['Close'].iloc[-132]) - 1) * 100,
            '1Y Return (%)': ((current_price / hist_data['Close'].iloc[-252]) - 1) * 100,
        }
        
        return returns

# Example usage and analysis
def run_hul_analysis():
    """Main function to run the complete HUL analysis"""
    
    print("=" * 60)
    print("HINDUSTAN UNILEVER LIMITED - EQUITY ANALYSIS")
    print("=" * 60)
    
    # Initialize analysis
    hul = HULEquityAnalysis()
    
    # 1. Fetch stock data
    print("\n1. Fetching Stock Data...")
    hist_data, info = hul.fetch_stock_data()
    
    if hist_data is not None:
        print(f"✓ Successfully fetched {len(hist_data)} days of data")
        current_price = hist_data['Close'].iloc[-1]
        print(f"Current Price: ₹{current_price:.2f}")
    
    # 2. Get financial metrics
    print("\n2. Key Financial Metrics:")
    print("-" * 40)
    metrics = hul.get_financial_metrics()
    for key, value in metrics.items():
        if 'Cr' in key:
            print(f"{key}: ₹{value:,.0f}")
        elif '%' in key:
            print(f"{key}: {value:.2f}%")
        else:
            print(f"{key}: {value:.2f}")
    
    # 3. DCF Analysis
    print("\n3. DCF Valuation Model:")
    print("-" * 40)
    
    # Assumptions (approximate based on research)
    initial_fcf = 8500  # Approximate FCF in crores
    growth_rates = [0.08, 0.07, 0.06, 0.06, 0.05]  # 5-year projections
    
    dcf_results = hul.simple_dcf_model(initial_fcf, growth_rates)
    
    print(f"Enterprise Value: ₹{dcf_results['Enterprise Value']:,.0f} crores")
    print(f"PV of FCF (5 years): ₹{dcf_results['Sum PV FCF']:,.0f} crores")
    print(f"PV of Terminal Value: ₹{dcf_results['PV of Terminal']:,.0f} crores")
    
    # Approximate share count (240 crores shares)
    shares_outstanding = 240  # crores
    fair_value_per_share = (dcf_results['Enterprise Value'] * 10) / shares_outstanding
    print(f"Implied Fair Value per Share: ₹{fair_value_per_share:.0f}")
    
    # 4. Peer Comparison
    print("\n4. Peer Comparison:")
    print("-" * 40)
    peer_df = hul.peer_comparison()
    print(peer_df.to_string(index=False))
    
    # 5. Return Analysis
    if hist_data is not None:
        print("\n5. Return Analysis:")
        print("-" * 40)
        returns = hul.calculate_returns(hist_data)
        for period, return_val in returns.items():
            print(f"{period}: {return_val:.2f}%")
    
    # 6. Investment Recommendation
    print("\n6. Investment Recommendation:")
    print("-" * 40)
    target_price = 2650
    current_price_approx = 2550  # Approximate current price
    upside = ((target_price / current_price_approx) - 1) * 100
    
    print(f"Current Price: ₹{current_price_approx}")
    print(f"Target Price: ₹{target_price}")
    print(f"Upside/Downside: {upside:.1f}%")
    print(f"Recommendation: {'BUY' if upside > 15 else 'HOLD' if upside > -10 else 'SELL'}")
    
    return hul, hist_data

# Excel alternative functions
def export_to_excel(data_dict, filename="hul_analysis.xlsx"):
    """Export analysis results to Excel"""
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for sheet_name, data in data_dict.items():
            if isinstance(data, pd.DataFrame):
                data.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                pd.DataFrame([data]).to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Analysis exported to {filename}")

# Run the analysis
if __name__ == "__main__":
    hul_analyzer, historical_data = run_hul_analysis()
    
    # Optional: Create visualizations
    if historical_data is not None:
        hul_analyzer.plot_stock_performance(historical_data)

print("""
USAGE INSTRUCTIONS:
1. Install required packages: pip install yfinance pandas numpy matplotlib seaborn openpyxl
2. Run this script to get automated analysis
3. Modify DCF assumptions in the run_hul_analysis() function
4. Add more peers in peer_comparison() method
5. Export results to Excel using export_to_excel() function

KEY FEATURES:
- Automated data fetching from Yahoo Finance
- Simple DCF valuation model
- Peer comparison analysis
- Return calculations and performance metrics
- Exportable results for presentations
""")