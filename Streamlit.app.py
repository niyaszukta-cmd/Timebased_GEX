# ============================================================================
# ADVANCED GEX + DEX ANALYSIS - STREAMLIT DASHBOARD
# WITH TIME RANGE SLIDER FOR BACKTESTING
# Created by NYZTrade - Options Analytics
# ============================================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import warnings
import time
import pickle
import os

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="GEX + DEX Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling including time slider
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .bullish {
        color: #00c853;
        font-weight: bold;
    }
    .bearish {
        color: #ff1744;
        font-weight: bold;
    }
    .neutral {
        color: #ff9800;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    
    /* Time Slider Styling */
    .time-slider-container {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        padding: 20px 30px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .time-label {
        color: #a0a0a0;
        font-size: 14px;
        margin-bottom: 5px;
    }
    .time-value {
        color: #ffffff;
        font-size: 18px;
        font-weight: bold;
    }
    .preset-btn {
        background-color: transparent;
        border: 1px solid #6c5ce7;
        color: #6c5ce7;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 2px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .preset-btn:hover {
        background-color: #6c5ce7;
        color: white;
    }
    .preset-btn-active {
        background-color: #6c5ce7;
        color: white;
    }
    
    /* Live indicator */
    .live-indicator {
        display: inline-flex;
        align-items: center;
        background-color: #1a1a2e;
        padding: 5px 15px;
        border-radius: 20px;
        margin-left: 10px;
    }
    .live-dot {
        width: 10px;
        height: 10px;
        background-color: #00ff88;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Historical mode indicator */
    .historical-indicator {
        display: inline-flex;
        align-items: center;
        background-color: #1a1a2e;
        padding: 5px 15px;
        border-radius: 20px;
        margin-left: 10px;
        color: #ffa500;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'data_snapshots' not in st.session_state:
    st.session_state.data_snapshots = {}  # {timestamp: {df, futures_ltp, market_info, atm_info, flow_metrics}}

if 'snapshot_times' not in st.session_state:
    st.session_state.snapshot_times = []

if 'selected_time_index' not in st.session_state:
    st.session_state.selected_time_index = None

if 'is_live_mode' not in st.session_state:
    st.session_state.is_live_mode = True

if 'last_capture_time' not in st.session_state:
    st.session_state.last_capture_time = None

if 'auto_capture' not in st.session_state:
    st.session_state.auto_capture = True

if 'capture_interval' not in st.session_state:
    st.session_state.capture_interval = 3  # minutes


# ============================================================================
# BLACK-SCHOLES CALCULATOR (GAMMA + DELTA)
# ============================================================================

class BlackScholesCalculator:
    """Calculate accurate gamma and delta using Black-Scholes formula"""

    @staticmethod
    def calculate_d1(S, K, T, r, sigma):
        """Calculate d1 for Black-Scholes"""
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))

    @staticmethod
    def calculate_gamma(S, K, T, r, sigma):
        """Calculate Black-Scholes Gamma"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0

        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            n_prime_d1 = norm.pdf(d1)
            gamma = n_prime_d1 / (S * sigma * np.sqrt(T))
            return gamma
        except Exception as e:
            return 0

    @staticmethod
    def calculate_call_delta(S, K, T, r, sigma):
        """Calculate Call Delta = N(d1)"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0

        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            return norm.cdf(d1)
        except:
            return 0

    @staticmethod
    def calculate_put_delta(S, K, T, r, sigma):
        """Calculate Put Delta = N(d1) - 1"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0

        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            return norm.cdf(d1) - 1
        except:
            return 0


# ============================================================================
# ENHANCED NSE DATA FETCHER WITH GEX + DEX CALCULATIONS
# ============================================================================

class EnhancedGEXDEXCalculator:
    """Advanced GEX + DEX calculations with improved futures fetching"""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.nseindia.com/',
            'Host': 'www.nseindia.com',
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.base_url = "https://www.nseindia.com"
        self.option_chain_url = "https://www.nseindia.com/api/option-chain-indices"
        self.risk_free_rate = 0.07
        self.bs_calc = BlackScholesCalculator()
        self.use_demo_data = False

    def initialize_session(self):
        """Initialize session with NSE with multiple retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # First hit the main page to get cookies
                response = self.session.get(self.base_url, timeout=15)
                if response.status_code == 200:
                    # Small delay before API call
                    time.sleep(1)
                    return True, "Connected to NSE"
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    self.use_demo_data = True
                    return True, "Using demo data (NSE connection failed)"
        
        self.use_demo_data = True
        return True, "Using demo data (NSE blocked cloud IP)"

    def fetch_futures_ltp_method1(self, symbol, expiry_date=None):
        """Method 1: Fetch from Groww.in (Most reliable!)"""
        try:
            symbol_map = {
                'NIFTY': 'nifty',
                'BANKNIFTY': 'bank-nifty',
                'FINNIFTY': 'finnifty',
                'MIDCPNIFTY': 'midcpnifty'
            }

            groww_symbol = symbol_map.get(symbol, 'nifty')
            url = f"https://groww.in/futures/{groww_symbol}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0'
            }

            response = self.session.get(url, headers=headers, timeout=15)

            if response.status_code == 200:
                html_content = response.text
                import re

                script_patterns = [
                    r'"ltp":\s*([0-9.]+)',
                    r'"lastPrice":\s*([0-9.]+)',
                    r'"close":\s*([0-9.]+)',
                    r'"currentPrice":\s*([0-9.]+)',
                    r'ltp.*?([0-9]{5,6}\.[0-9]{1,2})',
                ]

                for pattern in script_patterns:
                    matches = re.findall(pattern, html_content)
                    if matches:
                        for match in matches:
                            price = float(match)
                            if symbol == 'NIFTY' and 15000 < price < 35000:
                                return price, expiry_date
                            elif symbol == 'BANKNIFTY' and 35000 < price < 70000:
                                return price, expiry_date
                            elif symbol == 'FINNIFTY' and 15000 < price < 35000:
                                return price, expiry_date
                            elif symbol == 'MIDCPNIFTY' and 5000 < price < 25000:
                                return price, expiry_date

            return None, None
        except Exception as e:
            return None, None

    def fetch_futures_ltp_method2(self, symbol, spot_price, expiry_date):
        """Method 2: Calculate from ATM options using Put-Call Parity"""
        try:
            url = f"{self.option_chain_url}?symbol={symbol}"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                records = data['records']

                atm_strike = None
                min_diff = float('inf')

                for item in records.get('data', []):
                    if expiry_date and item.get('expiryDate') != expiry_date:
                        continue

                    strike = item.get('strikePrice', 0)
                    diff = abs(strike - spot_price)

                    if diff < min_diff:
                        min_diff = diff
                        atm_strike = strike

                if atm_strike:
                    for item in records.get('data', []):
                        if item.get('strikePrice') == atm_strike:
                            if expiry_date and item.get('expiryDate') != expiry_date:
                                continue

                            ce = item.get('CE', {})
                            pe = item.get('PE', {})

                            call_ltp = ce.get('lastPrice', 0)
                            put_ltp = pe.get('lastPrice', 0)

                            if call_ltp > 0 and put_ltp > 0:
                                futures_price = atm_strike + call_ltp - put_ltp
                                return futures_price, expiry_date

            return None, None
        except Exception as e:
            return None, None

    def fetch_futures_ltp_method3(self, symbol, spot_price, days_to_expiry):
        """Method 3: Theoretical futures price using cost of carry"""
        try:
            T = days_to_expiry / 365.0
            futures_price = spot_price * np.exp(self.risk_free_rate * T)
            return futures_price, None
        except Exception as e:
            return None, None

    def fetch_futures_ltp_comprehensive(self, symbol, spot_price, expiry_date, days_to_expiry):
        """Comprehensive futures fetching with multiple fallback methods"""
        # Method 1: Groww.in
        futures_ltp, futures_expiry = self.fetch_futures_ltp_method1(symbol, expiry_date)
        if futures_ltp and futures_ltp > 0:
            return futures_ltp, futures_expiry, "Groww.in"

        # Method 2: Put-Call Parity
        futures_ltp, futures_expiry = self.fetch_futures_ltp_method2(symbol, spot_price, expiry_date)
        if futures_ltp and futures_ltp > 0:
            return futures_ltp, futures_expiry, "Put-Call Parity"

        # Method 3: Cost of Carry
        futures_ltp, _ = self.fetch_futures_ltp_method3(symbol, spot_price, days_to_expiry)
        if futures_ltp and futures_ltp > 0:
            return futures_ltp, expiry_date, "Cost of Carry Model"

        # Fallback to Spot
        return spot_price, expiry_date, "Spot (Fallback)"

    def calculate_time_to_expiry(self, expiry_date_str):
        """Calculate time to expiry in years"""
        try:
            expiry_date = datetime.strptime(expiry_date_str, "%d-%b-%Y")
            today = datetime.now()
            days_to_expiry = (expiry_date - today).days
            time_to_expiry = max(days_to_expiry / 365, 0.001)
            return time_to_expiry, days_to_expiry
        except:
            return 7/365, 7

    def generate_demo_data(self, symbol="NIFTY", strikes_range=10):
        """Generate demo data when NSE is not accessible"""
        
        # Demo spot prices based on symbol
        demo_prices = {
            'NIFTY': 24250.50,
            'BANKNIFTY': 51850.75,
            'FINNIFTY': 23150.25,
            'MIDCPNIFTY': 12450.50
        }
        
        spot_price = demo_prices.get(symbol, 24250.50)
        futures_ltp = spot_price * 1.001  # Small premium
        
        # Contract specifications
        if 'BANKNIFTY' in symbol:
            contract_size = 15
            strike_interval = 100
        elif 'FINNIFTY' in symbol:
            contract_size = 40
            strike_interval = 50
        elif 'MIDCPNIFTY' in symbol:
            contract_size = 75
            strike_interval = 25
        else:
            contract_size = 25
            strike_interval = 50
        
        # Generate ATM strike
        atm_strike = round(spot_price / strike_interval) * strike_interval
        
        # Generate strikes around ATM
        all_strikes = []
        time_to_expiry = 7 / 365
        
        for i in range(-strikes_range, strikes_range + 1):
            strike = atm_strike + (i * strike_interval)
            
            # Generate realistic demo OI and premiums
            distance_from_atm = abs(i)
            
            # OI peaks at OTM strikes
            base_oi = 500000
            if i < 0:  # Below ATM - Put OI higher
                call_oi = int(base_oi * (0.5 + 0.3 * np.random.random()) * max(0.3, 1 - distance_from_atm * 0.1))
                put_oi = int(base_oi * (1 + 0.5 * np.random.random()) * max(0.3, 1 - distance_from_atm * 0.05))
            else:  # Above ATM - Call OI higher
                call_oi = int(base_oi * (1 + 0.5 * np.random.random()) * max(0.3, 1 - distance_from_atm * 0.05))
                put_oi = int(base_oi * (0.5 + 0.3 * np.random.random()) * max(0.3, 1 - distance_from_atm * 0.1))
            
            # OI changes
            call_oi_change = int((np.random.random() - 0.5) * call_oi * 0.1)
            put_oi_change = int((np.random.random() - 0.5) * put_oi * 0.1)
            
            # Volume
            call_volume = int(call_oi * (0.1 + 0.2 * np.random.random()))
            put_volume = int(put_oi * (0.1 + 0.2 * np.random.random()))
            
            # IV (higher for OTM)
            base_iv = 15
            call_iv = base_iv + distance_from_atm * 0.5 + np.random.random() * 2
            put_iv = base_iv + distance_from_atm * 0.5 + np.random.random() * 2
            
            # Premiums (simplified Black-Scholes approximation)
            if strike < spot_price:
                call_ltp = max(5, spot_price - strike + np.random.random() * 50)
                put_ltp = max(5, np.random.random() * 50 * (1 + distance_from_atm * 0.2))
            else:
                call_ltp = max(5, np.random.random() * 50 * (1 + distance_from_atm * 0.2))
                put_ltp = max(5, strike - spot_price + np.random.random() * 50)
            
            call_iv_decimal = call_iv / 100
            put_iv_decimal = put_iv / 100
            
            # Calculate Greeks
            call_gamma = self.bs_calc.calculate_gamma(
                S=futures_ltp, K=strike, T=time_to_expiry,
                r=self.risk_free_rate, sigma=call_iv_decimal
            )
            put_gamma = self.bs_calc.calculate_gamma(
                S=futures_ltp, K=strike, T=time_to_expiry,
                r=self.risk_free_rate, sigma=put_iv_decimal
            )
            call_delta = self.bs_calc.calculate_call_delta(
                S=futures_ltp, K=strike, T=time_to_expiry,
                r=self.risk_free_rate, sigma=call_iv_decimal
            )
            put_delta = self.bs_calc.calculate_put_delta(
                S=futures_ltp, K=strike, T=time_to_expiry,
                r=self.risk_free_rate, sigma=put_iv_decimal
            )
            
            # Calculate GEX and DEX
            call_gex = (call_oi * call_gamma * futures_ltp * futures_ltp * contract_size) / 1_000_000_000
            put_gex = -(put_oi * put_gamma * futures_ltp * futures_ltp * contract_size) / 1_000_000_000
            call_dex = (call_oi * call_delta * futures_ltp * contract_size) / 1_000_000_000
            put_dex = (put_oi * put_delta * futures_ltp * contract_size) / 1_000_000_000
            
            call_flow_gex = (call_oi_change * call_gamma * futures_ltp * futures_ltp * contract_size) / 1_000_000_000
            put_flow_gex = -(put_oi_change * put_gamma * futures_ltp * futures_ltp * contract_size) / 1_000_000_000
            call_flow_dex = (call_oi_change * call_delta * futures_ltp * contract_size) / 1_000_000_000
            put_flow_dex = (put_oi_change * put_delta * futures_ltp * contract_size) / 1_000_000_000
            
            all_strikes.append({
                'Strike': strike,
                'Call_OI': call_oi,
                'Put_OI': put_oi,
                'Call_OI_Change': call_oi_change,
                'Put_OI_Change': put_oi_change,
                'Call_Volume': call_volume,
                'Put_Volume': put_volume,
                'Call_IV': call_iv,
                'Put_IV': put_iv,
                'Call_LTP': call_ltp,
                'Put_LTP': put_ltp,
                'Call_Gamma': call_gamma,
                'Put_Gamma': put_gamma,
                'Call_Delta': call_delta,
                'Put_Delta': put_delta,
                'Call_GEX': call_gex,
                'Put_GEX': put_gex,
                'Net_GEX': call_gex + put_gex,
                'Call_DEX': call_dex,
                'Put_DEX': put_dex,
                'Net_DEX': call_dex + put_dex,
                'Call_Flow_GEX': call_flow_gex,
                'Put_Flow_GEX': put_flow_gex,
                'Net_Flow_GEX': call_flow_gex + put_flow_gex,
                'Call_Flow_DEX': call_flow_dex,
                'Put_Flow_DEX': put_flow_dex,
                'Net_Flow_DEX': call_flow_dex + put_flow_dex
            })
        
        df = pd.DataFrame(all_strikes)
        df = df.sort_values('Strike').reset_index(drop=True)
        
        # Add calculated columns
        df['Call_GEX_B'] = df['Call_GEX']
        df['Put_GEX_B'] = df['Put_GEX']
        df['Net_GEX_B'] = df['Net_GEX']
        df['Call_DEX_B'] = df['Call_DEX']
        df['Put_DEX_B'] = df['Put_DEX']
        df['Net_DEX_B'] = df['Net_DEX']
        df['Call_Flow_GEX_B'] = df['Call_Flow_GEX']
        df['Put_Flow_GEX_B'] = df['Put_Flow_GEX']
        df['Net_Flow_GEX_B'] = df['Net_Flow_GEX']
        df['Call_Flow_DEX_B'] = df['Call_Flow_DEX']
        df['Put_Flow_DEX_B'] = df['Put_Flow_DEX']
        df['Net_Flow_DEX_B'] = df['Net_Flow_DEX']
        df['Total_Volume'] = df['Call_Volume'] + df['Put_Volume']
        
        max_net_gex = df['Net_GEX_B'].abs().max()
        if max_net_gex > 0:
            df['Hedging_Pressure'] = (df['Net_GEX_B'] / max_net_gex) * 100
        else:
            df['Hedging_Pressure'] = 0
        
        # ATM info
        atm_row = df[df['Strike'] == atm_strike].iloc[0] if len(df[df['Strike'] == atm_strike]) > 0 else df.iloc[len(df)//2]
        atm_call_premium = atm_row['Call_LTP']
        atm_put_premium = atm_row['Put_LTP']
        
        atm_info = {
            'atm_strike': atm_strike,
            'atm_call_premium': atm_call_premium,
            'atm_put_premium': atm_put_premium,
            'atm_straddle_premium': atm_call_premium + atm_put_premium
        }
        
        # Calculate next Thursday for expiry
        today = datetime.now()
        days_until_thursday = (3 - today.weekday()) % 7
        if days_until_thursday == 0:
            days_until_thursday = 7
        next_thursday = today + timedelta(days=days_until_thursday)
        selected_expiry = next_thursday.strftime("%d-%b-%Y")
        
        market_info = {
            'spot_price': spot_price,
            'futures_ltp': futures_ltp,
            'basis': futures_ltp - spot_price,
            'basis_pct': ((futures_ltp - spot_price) / spot_price * 100),
            'fetch_method': 'Demo Data',
            'timestamp': datetime.now().strftime('%d-%b-%Y %H:%M:%S'),
            'expiry_dates': [selected_expiry],
            'selected_expiry': selected_expiry,
            'days_to_expiry': days_until_thursday
        }
        
        return df, futures_ltp, market_info, atm_info, None

    def fetch_and_calculate_gex_dex(self, symbol="NIFTY", strikes_range=10, expiry_index=0):
        """
        Fetch option chain and calculate both GEX and DEX
        """
        # Check if we need to use demo data
        if self.use_demo_data:
            return self.generate_demo_data(symbol, strikes_range)
        
        try:
            url = f"{self.option_chain_url}?symbol={symbol}"
            response = self.session.get(url, timeout=15)

            if response.status_code != 200:
                # Try demo data as fallback
                return self.generate_demo_data(symbol, strikes_range)
            
            # Check if response is JSON
            try:
                data = response.json()
            except:
                # Response is not JSON (blocked by NSE)
                return self.generate_demo_data(symbol, strikes_range)
            
            # Check if data has the expected structure
            if 'records' not in data:
                return self.generate_demo_data(symbol, strikes_range)
            
            records = data['records']

            spot_price = records.get('underlyingValue', 0)
            timestamp = records.get('timestamp', '')
            expiry_dates = records.get('expiryDates', [])

            if not expiry_dates:
                selected_expiry = None
                time_to_expiry = 7/365
                days_to_expiry = 7
            elif expiry_index >= len(expiry_dates):
                selected_expiry = expiry_dates[0]
                time_to_expiry, days_to_expiry = self.calculate_time_to_expiry(selected_expiry)
            else:
                selected_expiry = expiry_dates[expiry_index]
                time_to_expiry, days_to_expiry = self.calculate_time_to_expiry(selected_expiry)

            futures_ltp, futures_expiry, fetch_method = self.fetch_futures_ltp_comprehensive(
                symbol, spot_price, selected_expiry, days_to_expiry
            )

            basis = futures_ltp - spot_price
            basis_pct = (basis / spot_price * 100) if spot_price > 0 else 0

            reference_price = futures_ltp

            # Contract specifications
            if 'BANKNIFTY' in symbol:
                contract_size = 15
                strike_interval = 100
            elif 'FINNIFTY' in symbol:
                contract_size = 40
                strike_interval = 50
            elif 'MIDCPNIFTY' in symbol:
                contract_size = 75
                strike_interval = 25
            else:
                contract_size = 25
                strike_interval = 50

            # Process strikes data
            all_strikes = []
            processed_strikes = set()
            atm_strike = None
            min_atm_diff = float('inf')
            atm_call_premium = 0
            atm_put_premium = 0

            for item in records.get('data', []):
                if selected_expiry and item.get('expiryDate') != selected_expiry:
                    continue

                strike = item.get('strikePrice', 0)
                if strike == 0 or strike in processed_strikes:
                    continue

                processed_strikes.add(strike)

                strike_distance = abs(strike - reference_price) / strike_interval
                if strike_distance > strikes_range:
                    continue

                ce = item.get('CE', {})
                pe = item.get('PE', {})

                call_oi = ce.get('openInterest', 0)
                put_oi = pe.get('openInterest', 0)
                call_oi_change = ce.get('changeinOpenInterest', 0)
                put_oi_change = pe.get('changeinOpenInterest', 0)
                call_volume = ce.get('totalTradedVolume', 0)
                put_volume = pe.get('totalTradedVolume', 0)
                call_iv = ce.get('impliedVolatility', 0)
                put_iv = pe.get('impliedVolatility', 0)
                call_ltp = ce.get('lastPrice', 0)
                put_ltp = pe.get('lastPrice', 0)

                # Find ATM strike
                strike_diff = abs(strike - reference_price)
                if strike_diff < min_atm_diff:
                    min_atm_diff = strike_diff
                    atm_strike = strike
                    atm_call_premium = call_ltp
                    atm_put_premium = put_ltp

                call_iv_decimal = call_iv / 100 if call_iv > 0 else 0.15
                put_iv_decimal = put_iv / 100 if put_iv > 0 else 0.15

                # Calculate Gammas
                call_gamma = self.bs_calc.calculate_gamma(
                    S=reference_price, K=strike, T=time_to_expiry,
                    r=self.risk_free_rate, sigma=call_iv_decimal
                )

                put_gamma = self.bs_calc.calculate_gamma(
                    S=reference_price, K=strike, T=time_to_expiry,
                    r=self.risk_free_rate, sigma=put_iv_decimal
                )

                # Calculate Deltas
                call_delta = self.bs_calc.calculate_call_delta(
                    S=reference_price, K=strike, T=time_to_expiry,
                    r=self.risk_free_rate, sigma=call_iv_decimal
                )

                put_delta = self.bs_calc.calculate_put_delta(
                    S=reference_price, K=strike, T=time_to_expiry,
                    r=self.risk_free_rate, sigma=put_iv_decimal
                )

                # Calculate GEX (in Billions)
                call_gex = (call_oi * call_gamma * reference_price * reference_price * contract_size) / 1_000_000_000
                put_gex = -(put_oi * put_gamma * reference_price * reference_price * contract_size) / 1_000_000_000

                # Calculate DEX (in Billions)
                call_dex = (call_oi * call_delta * reference_price * contract_size) / 1_000_000_000
                put_dex = (put_oi * put_delta * reference_price * contract_size) / 1_000_000_000

                # Flow GEX
                call_flow_gex = (call_oi_change * call_gamma * reference_price * reference_price * contract_size) / 1_000_000_000
                put_flow_gex = -(put_oi_change * put_gamma * reference_price * reference_price * contract_size) / 1_000_000_000

                # Flow DEX
                call_flow_dex = (call_oi_change * call_delta * reference_price * contract_size) / 1_000_000_000
                put_flow_dex = (put_oi_change * put_delta * reference_price * contract_size) / 1_000_000_000

                all_strikes.append({
                    'Strike': strike,
                    'Call_OI': call_oi,
                    'Put_OI': put_oi,
                    'Call_OI_Change': call_oi_change,
                    'Put_OI_Change': put_oi_change,
                    'Call_Volume': call_volume,
                    'Put_Volume': put_volume,
                    'Call_IV': call_iv,
                    'Put_IV': put_iv,
                    'Call_LTP': call_ltp,
                    'Put_LTP': put_ltp,
                    'Call_Gamma': call_gamma,
                    'Put_Gamma': put_gamma,
                    'Call_Delta': call_delta,
                    'Put_Delta': put_delta,
                    'Call_GEX': call_gex,
                    'Put_GEX': put_gex,
                    'Net_GEX': call_gex + put_gex,
                    'Call_DEX': call_dex,
                    'Put_DEX': put_dex,
                    'Net_DEX': call_dex + put_dex,
                    'Call_Flow_GEX': call_flow_gex,
                    'Put_Flow_GEX': put_flow_gex,
                    'Net_Flow_GEX': call_flow_gex + put_flow_gex,
                    'Call_Flow_DEX': call_flow_dex,
                    'Put_Flow_DEX': put_flow_dex,
                    'Net_Flow_DEX': call_flow_dex + put_flow_dex
                })

            if not all_strikes:
                # Use demo data if no strikes found
                return self.generate_demo_data(symbol, strikes_range)

            df = pd.DataFrame(all_strikes)
            df = df.sort_values('Strike').reset_index(drop=True)

            df['Call_GEX_B'] = df['Call_GEX']
            df['Put_GEX_B'] = df['Put_GEX']
            df['Net_GEX_B'] = df['Net_GEX']
            df['Call_DEX_B'] = df['Call_DEX']
            df['Put_DEX_B'] = df['Put_DEX']
            df['Net_DEX_B'] = df['Net_DEX']
            df['Call_Flow_GEX_B'] = df['Call_Flow_GEX']
            df['Put_Flow_GEX_B'] = df['Put_Flow_GEX']
            df['Net_Flow_GEX_B'] = df['Net_Flow_GEX']
            df['Call_Flow_DEX_B'] = df['Call_Flow_DEX']
            df['Put_Flow_DEX_B'] = df['Put_Flow_DEX']
            df['Net_Flow_DEX_B'] = df['Net_Flow_DEX']
            df['Total_Volume'] = df['Call_Volume'] + df['Put_Volume']

            # Calculate Hedging Pressure
            max_net_gex = df['Net_GEX_B'].abs().max()
            if max_net_gex > 0:
                df['Hedging_Pressure'] = (df['Net_GEX_B'] / max_net_gex) * 100
            else:
                df['Hedging_Pressure'] = 0

            # Calculate ATM Straddle
            atm_straddle_premium = atm_call_premium + atm_put_premium

            # Return ATM info as dictionary
            atm_info = {
                'atm_strike': atm_strike,
                'atm_call_premium': atm_call_premium,
                'atm_put_premium': atm_put_premium,
                'atm_straddle_premium': atm_straddle_premium
            }

            # Additional info
            market_info = {
                'spot_price': spot_price,
                'futures_ltp': futures_ltp,
                'basis': basis,
                'basis_pct': basis_pct,
                'fetch_method': fetch_method,
                'timestamp': timestamp,
                'expiry_dates': expiry_dates,
                'selected_expiry': selected_expiry,
                'days_to_expiry': days_to_expiry
            }

            return df, reference_price, market_info, atm_info, None

        except Exception as e:
            # Use demo data as fallback on any error
            return self.generate_demo_data(symbol, strikes_range)


# ============================================================================
# MODIFIED GEX + DEX FLOW CALCULATION
# ============================================================================

def calculate_dual_gex_dex_flow(df, futures_ltp):
    """
    MODIFIED: Calculate GEX flow based on 5 positive + 5 negative strikes closest to spot
    """
    df_unique = df.drop_duplicates(subset=['Strike']).sort_values('Strike').reset_index(drop=True)

    # ===== NEW GEX FLOW LOGIC =====
    positive_gex_df = df_unique[df_unique['Net_GEX_B'] > 0].copy()
    positive_gex_df['Distance'] = abs(positive_gex_df['Strike'] - futures_ltp)
    positive_gex_df = positive_gex_df.sort_values('Distance').head(5)

    negative_gex_df = df_unique[df_unique['Net_GEX_B'] < 0].copy()
    negative_gex_df['Distance'] = abs(negative_gex_df['Strike'] - futures_ltp)
    negative_gex_df = negative_gex_df.sort_values('Distance').head(5)

    gex_near_positive = float(positive_gex_df['Net_GEX_B'].sum()) if len(positive_gex_df) > 0 else 0.0
    gex_near_negative = float(negative_gex_df['Net_GEX_B'].sum()) if len(negative_gex_df) > 0 else 0.0
    gex_near_total = gex_near_positive + gex_near_negative

    positive_gex_mask = df_unique['Net_GEX_B'] > 0
    negative_gex_mask = df_unique['Net_GEX_B'] < 0

    gex_total_positive = float(df_unique.loc[positive_gex_mask, 'Net_GEX_B'].sum()) if positive_gex_mask.any() else 0.0
    gex_total_negative = float(df_unique.loc[negative_gex_mask, 'Net_GEX_B'].sum()) if negative_gex_mask.any() else 0.0
    gex_total_all = gex_total_positive + gex_total_negative

    # ===== DEX FLOW =====
    above_futures = df_unique[df_unique['Strike'] > futures_ltp].head(5)
    below_futures = df_unique[df_unique['Strike'] < futures_ltp].tail(5)

    dex_near_positive = float(above_futures['Net_DEX_B'].sum()) if len(above_futures) > 0 else 0.0
    dex_near_negative = float(below_futures['Net_DEX_B'].sum()) if len(below_futures) > 0 else 0.0
    dex_near_total = dex_near_positive + dex_near_negative

    positive_dex_mask = df_unique['Net_DEX_B'] > 0
    negative_dex_mask = df_unique['Net_DEX_B'] < 0

    dex_total_positive = float(df_unique.loc[positive_dex_mask, 'Net_DEX_B'].sum()) if positive_dex_mask.any() else 0.0
    dex_total_negative = float(df_unique.loc[negative_dex_mask, 'Net_DEX_B'].sum()) if negative_dex_mask.any() else 0.0
    dex_total_all = dex_total_positive + dex_total_negative

    # ===== BIAS LOGIC =====
    def get_gex_bias(flow_value):
        if flow_value > 50:
            return "🟢 STRONG BULLISH (Sideways to Bullish)", "green"
        elif flow_value > 0:
            return "🟢 BULLISH (Sideways to Bullish)", "lightgreen"
        elif flow_value < -50:
            return "🔴 STRONG BEARISH (High Volatility)", "red"
        elif flow_value < 0:
            return "🔴 BEARISH (High Volatility)", "lightcoral"
        else:
            return "⚖️ NEUTRAL", "orange"

    def get_dex_bias(flow_value):
        if flow_value > 50:
            return "🟢 BULLISH", "green"
        elif flow_value < -50:
            return "🔴 BEARISH", "red"
        elif flow_value > 0:
            return "🟢 Mild Bullish", "lightgreen"
        elif flow_value < 0:
            return "🔴 Mild Bearish", "lightcoral"
        else:
            return "⚖️ NEUTRAL", "orange"

    gex_near_bias, gex_near_color = get_gex_bias(gex_near_total)
    gex_total_bias, gex_total_color = get_gex_bias(gex_total_all)
    dex_near_bias, dex_near_color = get_dex_bias(dex_near_total)
    dex_total_bias, dex_total_color = get_dex_bias(dex_total_all)

    combined_signal = (gex_near_total + dex_near_total) / 2
    combined_bias, combined_color = get_gex_bias(combined_signal)

    return {
        'gex_near_positive': gex_near_positive,
        'gex_near_negative': gex_near_negative,
        'gex_near_total': gex_near_total,
        'gex_near_bias': gex_near_bias,
        'gex_near_color': gex_near_color,
        'gex_total_positive': gex_total_positive,
        'gex_total_negative': gex_total_negative,
        'gex_total_all': gex_total_all,
        'gex_total_bias': gex_total_bias,
        'gex_total_color': gex_total_color,
        'dex_near_positive': dex_near_positive,
        'dex_near_negative': dex_near_negative,
        'dex_near_total': dex_near_total,
        'dex_near_bias': dex_near_bias,
        'dex_near_color': dex_near_color,
        'dex_total_positive': dex_total_positive,
        'dex_total_negative': dex_total_negative,
        'dex_total_all': dex_total_all,
        'dex_total_bias': dex_total_bias,
        'dex_total_color': dex_total_color,
        'combined_signal': combined_signal,
        'combined_bias': combined_bias,
        'combined_color': combined_color,
        'positive_gex_strikes': positive_gex_df['Strike'].tolist() if len(positive_gex_df) > 0 else [],
        'negative_gex_strikes': negative_gex_df['Strike'].tolist() if len(negative_gex_df) > 0 else [],
        'above_strikes': above_futures['Strike'].tolist(),
        'below_strikes': below_futures['Strike'].tolist(),
    }


# ============================================================================
# TIME SLIDER COMPONENT
# ============================================================================

def render_time_slider():
    """Render the time slider component for backtesting"""
    
    st.markdown("---")
    
    # Time Slider Header
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### ⏰ Time Machine - Backtest Mode")
    
    with col2:
        if st.session_state.is_live_mode:
            st.markdown("""
            <div class="live-indicator">
                <div class="live-dot"></div>
                <span style="color: #00ff88; font-weight: bold;">LIVE</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="historical-indicator">
                <span>📜 HISTORICAL MODE</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if st.button("🔴 Go Live" if not st.session_state.is_live_mode else "✅ Live Mode Active", 
                     type="primary" if not st.session_state.is_live_mode else "secondary"):
            st.session_state.is_live_mode = True
            st.session_state.selected_time_index = None
            st.rerun()
    
    # Check if we have any snapshots
    if len(st.session_state.snapshot_times) == 0:
        st.info("📝 No historical data captured yet. Enable auto-capture or click 'Capture Snapshot' to start recording data for backtesting.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.auto_capture = st.checkbox("🔄 Auto-capture data", value=st.session_state.auto_capture)
        with col2:
            st.session_state.capture_interval = st.selectbox(
                "Capture interval",
                options=[1, 2, 3, 5, 10, 15],
                index=2,
                format_func=lambda x: f"{x} min{'s' if x > 1 else ''}"
            )
        return None
    
    # Time range display
    first_time = st.session_state.snapshot_times[0]
    last_time = st.session_state.snapshot_times[-1]
    
    st.markdown(f"""
    <div style="background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); border-radius: 15px; padding: 20px; margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <div>
                <span style="color: #a0a0a0; font-size: 14px;">Time Range</span><br>
                <span style="color: #ffffff; font-size: 18px; font-weight: bold;">
                    {first_time.strftime('%I:%M %p')} - {last_time.strftime('%I:%M %p')}
                </span>
            </div>
            <div style="color: #6c5ce7; font-size: 14px;">
                📸 {len(st.session_state.snapshot_times)} snapshots captured
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Time Slider
    if len(st.session_state.snapshot_times) > 1:
        # Create time labels for slider
        time_labels = [t.strftime('%I:%M %p') for t in st.session_state.snapshot_times]
        
        # Slider
        selected_idx = st.select_slider(
            "Select Time Point",
            options=list(range(len(st.session_state.snapshot_times))),
            value=st.session_state.selected_time_index if st.session_state.selected_time_index is not None else len(st.session_state.snapshot_times) - 1,
            format_func=lambda x: time_labels[x],
            key="time_slider"
        )
        
        # Update state
        if selected_idx != len(st.session_state.snapshot_times) - 1:
            st.session_state.is_live_mode = False
            st.session_state.selected_time_index = selected_idx
        
        # Preset buttons
        st.markdown("#### ⚡ Quick Jump")
        preset_cols = st.columns(7)
        
        presets = [
            ("5 Mins", 5),
            ("15 Mins", 15),
            ("30 Mins", 30),
            ("1 Hr", 60),
            ("2 Hrs", 120),
            ("3 Hrs", 180),
            ("Full Day", 9999)
        ]
        
        for idx, (label, minutes) in enumerate(presets):
            with preset_cols[idx]:
                if st.button(label, key=f"preset_{minutes}", use_container_width=True):
                    if minutes == 9999:
                        # Full day - go to first snapshot
                        st.session_state.selected_time_index = 0
                        st.session_state.is_live_mode = False
                    else:
                        # Find snapshot closest to X minutes ago
                        target_time = datetime.now() - timedelta(minutes=minutes)
                        closest_idx = 0
                        min_diff = float('inf')
                        
                        for i, snap_time in enumerate(st.session_state.snapshot_times):
                            diff = abs((snap_time - target_time).total_seconds())
                            if diff < min_diff:
                                min_diff = diff
                                closest_idx = i
                        
                        st.session_state.selected_time_index = closest_idx
                        st.session_state.is_live_mode = False
                    st.rerun()
        
        # Show selected time info
        if st.session_state.selected_time_index is not None:
            selected_time = st.session_state.snapshot_times[st.session_state.selected_time_index]
            time_ago = datetime.now() - selected_time
            minutes_ago = int(time_ago.total_seconds() / 60)
            
            if minutes_ago < 60:
                time_ago_str = f"{minutes_ago} min{'s' if minutes_ago != 1 else ''} ago"
            else:
                hours_ago = minutes_ago // 60
                mins = minutes_ago % 60
                time_ago_str = f"{hours_ago} hr{'s' if hours_ago != 1 else ''} {mins} min{'s' if mins != 1 else ''} ago"
            
            st.info(f"📍 **Viewing data from:** {selected_time.strftime('%I:%M:%S %p')} ({time_ago_str})")
    
    # Auto-capture settings
    with st.expander("⚙️ Capture Settings"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.auto_capture = st.checkbox(
                "🔄 Auto-capture enabled",
                value=st.session_state.auto_capture
            )
        
        with col2:
            st.session_state.capture_interval = st.selectbox(
                "Capture interval",
                options=[1, 2, 3, 5, 10, 15],
                index=[1, 2, 3, 5, 10, 15].index(st.session_state.capture_interval) if st.session_state.capture_interval in [1, 2, 3, 5, 10, 15] else 2,
                format_func=lambda x: f"{x} min{'s' if x > 1 else ''}"
            )
        
        with col3:
            if st.button("🗑️ Clear History"):
                st.session_state.data_snapshots = {}
                st.session_state.snapshot_times = []
                st.session_state.selected_time_index = None
                st.session_state.is_live_mode = True
                st.success("History cleared!")
                st.rerun()
    
    # Return selected snapshot data if in historical mode
    if not st.session_state.is_live_mode and st.session_state.selected_time_index is not None:
        selected_time = st.session_state.snapshot_times[st.session_state.selected_time_index]
        return st.session_state.data_snapshots.get(selected_time)
    
    return None


def capture_snapshot(df, futures_ltp, market_info, atm_info, flow_metrics):
    """Capture a data snapshot for backtesting"""
    current_time = datetime.now().replace(microsecond=0)
    
    # Check if enough time has passed since last capture
    if st.session_state.last_capture_time:
        time_diff = (current_time - st.session_state.last_capture_time).total_seconds() / 60
        if time_diff < st.session_state.capture_interval:
            return False
    
    # Store snapshot
    st.session_state.data_snapshots[current_time] = {
        'df': df.copy(),
        'futures_ltp': futures_ltp,
        'market_info': market_info.copy(),
        'atm_info': atm_info.copy(),
        'flow_metrics': flow_metrics.copy()
    }
    
    # Update times list
    if current_time not in st.session_state.snapshot_times:
        st.session_state.snapshot_times.append(current_time)
        st.session_state.snapshot_times.sort()
    
    st.session_state.last_capture_time = current_time
    
    # Keep only last 500 snapshots to manage memory
    if len(st.session_state.snapshot_times) > 500:
        oldest_time = st.session_state.snapshot_times.pop(0)
        if oldest_time in st.session_state.data_snapshots:
            del st.session_state.data_snapshots[oldest_time]
    
    return True


# ============================================================================
# ENHANCED VISUALIZATION WITH 7 CHARTS
# ============================================================================

def create_enhanced_dashboard(df, futures_ltp, symbol, flow_metrics, fetch_method, atm_info, is_historical=False, historical_time=None):
    """Enhanced dashboard with GEX + DEX + ATM Straddle analysis"""

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            '📊 Net GEX Profile with Volume',
            '📈 Delta Exposure (DEX) Profile with Volume',
            '🔄 GEX Flow (OI Changes) with Volume',
            '📉 DEX Flow with Volume',
            '🎯 Hedging Pressure Index with Volume',
            '⚡ Combined GEX+DEX Directional Bias',
            '💰 ATM Straddle Analysis',
            ''
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy", "colspan": 2}, None]],
        vertical_spacing=0.08, horizontal_spacing=0.10,
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )

    # Prepare volume scaling
    max_gex = df['Net_GEX_B'].abs().max()
    max_dex = df['Net_DEX_B'].abs().max()
    max_vol = df['Total_Volume'].max()

    if max_vol > 0:
        vol_scale_gex = (max_gex * 0.3) / max_vol
        vol_scale_dex = (max_dex * 0.3) / max_vol
        scaled_volume_gex = df['Total_Volume'] * vol_scale_gex
        scaled_volume_dex = df['Total_Volume'] * vol_scale_dex
    else:
        scaled_volume_gex = df['Total_Volume']
        scaled_volume_dex = df['Total_Volume']

    # S&R levels
    positive_gex_mask = df['Net_GEX_B'] > 0
    positive_gex = df[positive_gex_mask].nlargest(3, 'Net_GEX_B')

    # CHART 1: Net GEX Profile
    colors = ['green' if x > 0 else 'red' for x in df['Net_GEX_B']]
    fig.add_trace(go.Bar(y=df['Strike'], x=df['Net_GEX_B'], name='Net GEX',
                         orientation='h', marker_color=colors,
                         hovertemplate='<b>Strike:</b> %{y}<br><b>Net GEX:</b> %{x:.4f} B<extra></extra>'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(y=df['Strike'], x=scaled_volume_gex, name='Volume',
                             mode='lines+markers', line=dict(color='blue', width=2),
                             marker=dict(size=4),
                             hovertemplate='<b>Strike:</b> %{y}<br><b>Volume:</b> %{customdata:,.0f}<extra></extra>',
                             customdata=df['Total_Volume']), row=1, col=1)

    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="blue",
                  line_width=3, annotation_text="FUTURES", row=1, col=1)

    for idx, (_, row) in enumerate(positive_gex.iterrows()):
        if row['Strike'] < futures_ltp:
            fig.add_hline(y=row['Strike'], line_dash="dot", line_color="green",
                         line_width=1, opacity=0.5, annotation_text=f"S{idx+1}",
                         annotation_position="left", row=1, col=1)
        elif row['Strike'] > futures_ltp:
            fig.add_hline(y=row['Strike'], line_dash="dot", line_color="red",
                         line_width=1, opacity=0.5, annotation_text=f"R{idx+1}",
                         annotation_position="right", row=1, col=1)

    # CHART 2: DEX Profile
    dex_colors = ['green' if x > 0 else 'red' for x in df['Net_DEX_B']]
    fig.add_trace(go.Bar(y=df['Strike'], x=df['Net_DEX_B'], name='Net DEX',
                         orientation='h', marker_color=dex_colors,
                         hovertemplate='<b>Strike:</b> %{y}<br><b>Net DEX:</b> %{x:.4f} B<extra></extra>'),
                  row=1, col=2)

    fig.add_trace(go.Scatter(y=df['Strike'], x=scaled_volume_dex, name='Volume',
                             mode='lines+markers', line=dict(color='purple', width=2),
                             marker=dict(size=4),
                             hovertemplate='<b>Strike:</b> %{y}<br><b>Volume:</b> %{customdata:,.0f}<extra></extra>',
                             customdata=df['Total_Volume']), row=1, col=2)

    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="blue",
                  line_width=3, annotation_text="FUTURES", row=1, col=2)

    # CHART 3: GEX Flow
    flow_colors = ['green' if x > 0 else 'red' for x in df['Net_Flow_GEX_B']]
    fig.add_trace(go.Bar(y=df['Strike'], x=df['Net_Flow_GEX_B'], name='GEX Flow',
                         orientation='h', marker_color=flow_colors,
                         hovertemplate='<b>Strike:</b> %{y}<br><b>Flow GEX:</b> %{x:.4f} B<extra></extra>'),
                  row=2, col=1)

    max_flow = df['Net_Flow_GEX_B'].abs().max()
    vol_scale_flow = (max_flow * 0.3) / max_vol if max_vol > 0 and max_flow > 0 else 1
    scaled_volume_flow = df['Total_Volume'] * vol_scale_flow

    fig.add_trace(go.Scatter(y=df['Strike'], x=scaled_volume_flow, name='Volume',
                             mode='lines+markers', line=dict(color='orange', width=2),
                             marker=dict(size=4),
                             hovertemplate='<b>Strike:</b> %{y}<br><b>Volume:</b> %{customdata:,.0f}<extra></extra>',
                             customdata=df['Total_Volume']), row=2, col=1)

    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="blue",
                  line_width=3, annotation_text="FUTURES", row=2, col=1)

    # CHART 4: DEX Flow
    dex_flow_colors = ['green' if x > 0 else 'red' for x in df['Net_Flow_DEX_B']]
    fig.add_trace(go.Bar(y=df['Strike'], x=df['Net_Flow_DEX_B'], name='DEX Flow',
                         orientation='h', marker_color=dex_flow_colors,
                         hovertemplate='<b>Strike:</b> %{y}<br><b>Flow DEX:</b> %{x:.4f} B<extra></extra>'),
                  row=2, col=2)

    max_dex_flow = df['Net_Flow_DEX_B'].abs().max()
    vol_scale_dex_flow = (max_dex_flow * 0.3) / max_vol if max_vol > 0 and max_dex_flow > 0 else 1
    scaled_volume_dex_flow = df['Total_Volume'] * vol_scale_dex_flow

    fig.add_trace(go.Scatter(y=df['Strike'], x=scaled_volume_dex_flow, name='Volume',
                             mode='lines+markers', line=dict(color='cyan', width=2),
                             marker=dict(size=4),
                             hovertemplate='<b>Strike:</b> %{y}<br><b>Volume:</b> %{customdata:,.0f}<extra></extra>',
                             customdata=df['Total_Volume']), row=2, col=2)

    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="blue",
                  line_width=3, annotation_text="FUTURES", row=2, col=2)

    # CHART 5: Hedging Pressure
    fig.add_trace(go.Bar(y=df['Strike'], x=df['Hedging_Pressure'], orientation='h',
                         marker=dict(color=df['Hedging_Pressure'], colorscale='RdYlGn',
                                   showscale=True, colorbar=dict(title="Pressure", x=0.46, y=0.22, len=0.25)),
                         name='Hedge Pressure',
                         hovertemplate='<b>Strike:</b> %{y}<br><b>Pressure:</b> %{x:.2f}<extra></extra>'),
                  row=3, col=1)

    max_pressure = df['Hedging_Pressure'].abs().max()
    vol_scale_pressure = (max_pressure * 0.3) / max_vol if max_vol > 0 and max_pressure > 0 else 1
    scaled_volume_pressure = df['Total_Volume'] * vol_scale_pressure

    fig.add_trace(go.Scatter(y=df['Strike'], x=scaled_volume_pressure, name='Volume',
                             mode='lines+markers', line=dict(color='magenta', width=2),
                             marker=dict(size=4),
                             hovertemplate='<b>Strike:</b> %{y}<br><b>Volume:</b> %{customdata:,.0f}<extra></extra>',
                             customdata=df['Total_Volume']), row=3, col=1)

    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="blue",
                  line_width=3, annotation_text="FUTURES", row=3, col=1)

    # CHART 6: Combined
    df['Combined_Signal'] = (df['Net_GEX_B'] + df['Net_DEX_B']) / 2
    combined_colors = ['green' if x > 0 else 'red' for x in df['Combined_Signal']]

    fig.add_trace(go.Bar(y=df['Strike'], x=df['Combined_Signal'], orientation='h',
                         marker_color=combined_colors, name='Combined Signal',
                         hovertemplate='<b>Strike:</b> %{y}<br><b>Combined:</b> %{x:.4f} B<extra></extra>'),
                  row=3, col=2)

    fig.add_trace(go.Scatter(y=df['Strike'], x=df['Net_Flow_DEX_B'],
                             name='DEX Flow Curve',
                             mode='lines', line=dict(color='yellow', width=3, dash='dash'),
                             hovertemplate='<b>Strike:</b> %{y}<br><b>DEX Flow:</b> %{x:.4f} B<extra></extra>'),
                  row=3, col=2)

    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="blue",
                  line_width=3, annotation_text="FUTURES", row=3, col=2)

    # CHART 7: ATM Straddle
    atm_strike = atm_info['atm_strike']
    atm_call_premium = atm_info['atm_call_premium']
    atm_put_premium = atm_info['atm_put_premium']
    atm_straddle_premium = atm_info['atm_straddle_premium']

    strike_range = np.linspace(atm_strike * 0.90, atm_strike * 1.10, 100)
    call_payoff = np.maximum(strike_range - atm_strike, 0) - atm_call_premium
    put_payoff = np.maximum(atm_strike - strike_range, 0) - atm_put_premium
    straddle_payoff = call_payoff + put_payoff

    upper_breakeven = atm_strike + atm_straddle_premium
    lower_breakeven = atm_strike - atm_straddle_premium

    fig.add_trace(go.Scatter(x=strike_range, y=straddle_payoff,
                             name='Straddle P&L',
                             mode='lines', line=dict(color='purple', width=3),
                             hovertemplate='<b>Price:</b> %{x:.0f}<br><b>P&L:</b> ₹%{y:.2f}<extra></extra>'),
                  row=4, col=1)

    fig.add_trace(go.Scatter(x=strike_range, y=call_payoff,
                             name='Call P&L',
                             mode='lines', line=dict(color='green', width=2, dash='dot'),
                             hovertemplate='<b>Price:</b> %{x:.0f}<br><b>Call P&L:</b> ₹%{y:.2f}<extra></extra>'),
                  row=4, col=1)

    fig.add_trace(go.Scatter(x=strike_range, y=put_payoff,
                             name='Put P&L',
                             mode='lines', line=dict(color='red', width=2, dash='dot'),
                             hovertemplate='<b>Price:</b> %{x:.0f}<br><b>Put P&L:</b> ₹%{y:.2f}<extra></extra>'),
                  row=4, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=4, col=1)
    fig.add_vline(x=atm_strike, line_dash="solid", line_color="blue",
                  line_width=2, annotation_text=f"ATM: {atm_strike}", row=4, col=1)
    fig.add_vline(x=upper_breakeven, line_dash="dash", line_color="orange",
                  line_width=2, annotation_text=f"Upper BE: {upper_breakeven:.0f}", row=4, col=1)
    fig.add_vline(x=lower_breakeven, line_dash="dash", line_color="orange",
                  line_width=2, annotation_text=f"Lower BE: {lower_breakeven:.0f}", row=4, col=1)
    fig.add_vline(x=futures_ltp, line_dash="solid", line_color="red",
                  line_width=2, annotation_text=f"Current: {futures_ltp:.0f}", row=4, col=1)

    # Update axes
    fig.update_xaxes(title_text="Net GEX (B)", row=1, col=1)
    fig.update_xaxes(title_text="Net DEX (B)", row=1, col=2)
    fig.update_xaxes(title_text="GEX Flow (B)", row=2, col=1)
    fig.update_xaxes(title_text="DEX Flow (B)", row=2, col=2)
    fig.update_xaxes(title_text="Pressure", row=3, col=1)
    fig.update_xaxes(title_text="Combined (B)", row=3, col=2)
    fig.update_xaxes(title_text="Underlying Price", row=4, col=1)

    fig.update_yaxes(title_text="Strike", row=1, col=1)
    fig.update_yaxes(title_text="Strike", row=1, col=2)
    fig.update_yaxes(title_text="Strike", row=2, col=1)
    fig.update_yaxes(title_text="Strike", row=2, col=2)
    fig.update_yaxes(title_text="Strike", row=3, col=1)
    fig.update_yaxes(title_text="Strike", row=3, col=2)
    fig.update_yaxes(title_text="Profit/Loss (₹)", row=4, col=1)

    # Title with mode indicator
    if is_historical and historical_time:
        timestamp = historical_time.strftime('%I:%M:%S %p')
        mode_indicator = f"📜 HISTORICAL DATA - {timestamp}"
    else:
        timestamp = datetime.now().strftime('%H:%M:%S')
        mode_indicator = f"🔴 LIVE - {timestamp}"
    
    fig.update_layout(
        title=dict(
            text=f'<b>{symbol} - GEX + DEX Analysis (Futures: {futures_ltp:,.2f} via {fetch_method})</b><br>' +
                 f'<sup>{mode_indicator} | GEX: {flow_metrics["gex_near_bias"]} | DEX: {flow_metrics["dex_near_bias"]} | ' +
                 f'Combined: {flow_metrics["combined_bias"]} | ATM Straddle: ₹{atm_straddle_premium:.2f}</sup>',
            font=dict(size=14)
        ),
        height=1600, showlegend=True, template='plotly_white', hovermode='closest'
    )

    return fig


# ============================================================================
# TRADING STRATEGIES GENERATOR
# ============================================================================

def generate_trading_strategies(df, futures_ltp, flow_metrics, atm_info):
    """Generate option trading strategies based on GEX+DEX analysis"""
    
    strategies = []
    
    positive_gex_mask = df['Net_GEX_B'] > 0
    positive_gex = df[positive_gex_mask].nlargest(5, 'Net_GEX_B')

    supports_below = positive_gex[positive_gex['Strike'] < futures_ltp]
    resistances_above = positive_gex[positive_gex['Strike'] > futures_ltp]

    nearest_support = supports_below.iloc[0] if not supports_below.empty else None
    nearest_resistance = resistances_above.iloc[0] if not resistances_above.empty else None

    gex_bias = flow_metrics['gex_near_total']
    dex_bias = flow_metrics['dex_near_total']
    combined_signal = flow_metrics['combined_signal']

    atm_strike = atm_info['atm_strike']
    atm_straddle_premium = atm_info['atm_straddle_premium']

    support_strike = float(nearest_support['Strike']) if nearest_support is not None else futures_ltp - 100
    resistance_strike = float(nearest_resistance['Strike']) if nearest_resistance is not None else futures_ltp + 100

    if gex_bias > 50:
        strategies.append({
            'name': '🦅 Iron Condor',
            'category': 'SIDEWAYS TO BULLISH',
            'rationale': 'Strong positive GEX → Sideways movement expected, sell premium',
            'setup': f"Sell {int(futures_ltp)} CE + Buy {int(resistance_strike)} CE | Sell {int(futures_ltp)} PE + Buy {int(support_strike)} PE",
            'max_profit': 'Net Premium Received',
            'max_loss': 'Limited to strike width minus premium',
            'risk_level': '⚠️ MODERATE',
            'conditions': 'Hold if price stays between support and resistance'
        })

        if dex_bias > 0:
            strategies.append({
                'name': '📈 Bull Call Spread',
                'category': 'SIDEWAYS TO BULLISH',
                'rationale': 'Positive GEX + Bullish DEX → Mild upside with limited risk',
                'setup': f"Buy {int(futures_ltp)} CE + Sell {int(resistance_strike)} CE",
                'max_profit': 'Strike width - Premium',
                'max_loss': 'Premium Paid',
                'risk_level': '✅ LOW-MODERATE',
                'conditions': 'Bullish bias within resistance zone'
            })

        strategies.append({
            'name': '🔒 Short ATM Straddle',
            'category': 'SIDEWAYS TO BULLISH',
            'rationale': 'Strong positive GEX → Low volatility, collect premium',
            'setup': f"Sell {int(atm_strike)} CE + Sell {int(atm_strike)} PE (ATM Straddle: ₹{atm_straddle_premium:.2f})",
            'max_profit': f'₹{atm_straddle_premium:.2f} per lot',
            'max_loss': 'UNLIMITED (use stops or hedges)',
            'risk_level': '⚠️⚠️ HIGH (Requires experience)',
            'conditions': 'Price stays near ATM, low volatility persists'
        })

    elif gex_bias < -50:
        strategies.append({
            'name': '🎭 Long ATM Straddle',
            'category': 'HIGH VOLATILITY',
            'rationale': 'Negative GEX → High volatility expected, buy options',
            'setup': f"Buy {int(atm_strike)} CE + Buy {int(atm_strike)} PE (Cost: ₹{atm_straddle_premium:.2f})",
            'max_profit': 'Unlimited (both directions)',
            'max_loss': f'Premium Paid (₹{atm_straddle_premium:.2f})',
            'risk_level': '⚠️ HIGH (Needs big move)',
            'conditions': f'Price must move beyond ₹{atm_straddle_premium:.2f} to profit'
        })

        if dex_bias < -20:
            strategies.append({
                'name': '📉 Long Put',
                'category': 'HIGH VOLATILITY',
                'rationale': 'Negative GEX + Bearish DEX → Downside breakout likely',
                'setup': f"Buy {int(futures_ltp)} PE (ATM) or {int(futures_ltp - 100)} PE (OTM)",
                'max_profit': 'Substantial (down to zero)',
                'max_loss': 'Premium Paid',
                'risk_level': '⚠️ HIGH (Limited to premium)',
                'conditions': 'Breakdown below support expected'
            })

            strategies.append({
                'name': '🐻 Bear Put Spread',
                'category': 'HIGH VOLATILITY',
                'rationale': 'Reduce cost while maintaining downside exposure',
                'setup': f"Buy {int(futures_ltp)} PE + Sell {int(futures_ltp - 200)} PE",
                'max_profit': 'Strike width - Premium',
                'max_loss': 'Premium Paid',
                'risk_level': '✅ MODERATE',
                'conditions': 'Defined risk with bearish bias'
            })
        elif dex_bias > 20:
            strategies.append({
                'name': '🚀 Long Call (Counter-trend)',
                'category': 'HIGH VOLATILITY',
                'rationale': 'Negative GEX (volatility) + Bullish DEX → Upside volatility',
                'setup': f"Buy {int(futures_ltp)} CE (ATM) or {int(futures_ltp + 100)} CE (OTM)",
                'max_profit': 'Unlimited',
                'max_loss': 'Premium Paid',
                'risk_level': '⚠️ HIGH (Limited to premium)',
                'conditions': 'Volatile upside breakout expected'
            })

    else:
        if abs(dex_bias) > 20:
            if dex_bias > 0:
                strategies.append({
                    'name': '📈 Bull Call Spread',
                    'category': 'CAUTIOUS',
                    'rationale': 'Neutral GEX but bullish DEX → Defined risk bullish play',
                    'setup': f"Buy {int(futures_ltp)} CE + Sell {int(futures_ltp + 100)} CE",
                    'max_profit': 'Strike width - Premium',
                    'max_loss': 'Premium Paid',
                    'risk_level': '✅ MODERATE',
                    'conditions': 'Mild upside move expected'
                })
            else:
                strategies.append({
                    'name': '📉 Bear Put Spread',
                    'category': 'CAUTIOUS',
                    'rationale': 'Neutral GEX but bearish DEX → Defined risk bearish play',
                    'setup': f"Buy {int(futures_ltp)} PE + Sell {int(futures_ltp - 100)} PE",
                    'max_profit': 'Strike width - Premium',
                    'max_loss': 'Premium Paid',
                    'risk_level': '✅ MODERATE',
                    'conditions': 'Mild downside move expected'
                })
        else:
            strategies.append({
                'name': '⏸️ WAIT FOR CLARITY',
                'category': 'NO TRADE',
                'rationale': 'Mixed signals from both GEX and DEX → No clear edge',
                'setup': 'Stay in cash or small positions only',
                'max_profit': 'N/A',
                'max_loss': 'Opportunity cost',
                'risk_level': '✅ ZERO RISK',
                'conditions': 'Wait for stronger directional signals'
            })

    return strategies, {
        'gex_bias': gex_bias,
        'dex_bias': dex_bias,
        'combined_signal': combined_signal,
        'atm_strike': atm_strike,
        'atm_straddle_premium': atm_straddle_premium,
        'support_strike': support_strike,
        'resistance_strike': resistance_strike
    }


# ============================================================================
# PRICE HISTORY CHART
# ============================================================================

def create_price_history_chart():
    """Create a mini chart showing price movement through captured snapshots"""
    
    if len(st.session_state.snapshot_times) < 2:
        return None
    
    times = []
    prices = []
    gex_values = []
    
    for t in st.session_state.snapshot_times:
        if t in st.session_state.data_snapshots:
            snap = st.session_state.data_snapshots[t]
            times.append(t)
            prices.append(snap['futures_ltp'])
            gex_values.append(snap['flow_metrics']['gex_near_total'])
    
    if len(times) < 2:
        return None
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('Futures Price', 'GEX Flow'),
                        row_heights=[0.6, 0.4],
                        vertical_spacing=0.1)
    
    # Price line
    fig.add_trace(go.Scatter(x=times, y=prices, mode='lines+markers',
                             name='Futures Price', line=dict(color='#6c5ce7', width=2),
                             marker=dict(size=6)),
                  row=1, col=1)
    
    # GEX bars
    colors = ['green' if x > 0 else 'red' for x in gex_values]
    fig.add_trace(go.Bar(x=times, y=gex_values, name='GEX Flow',
                         marker_color=colors),
                  row=2, col=1)
    
    # Mark selected time if in historical mode
    if not st.session_state.is_live_mode and st.session_state.selected_time_index is not None:
        selected_time = st.session_state.snapshot_times[st.session_state.selected_time_index]
        if selected_time in st.session_state.data_snapshots:
            selected_price = st.session_state.data_snapshots[selected_time]['futures_ltp']
            fig.add_vline(x=selected_time, line_dash="dash", line_color="orange", line_width=2)
            fig.add_annotation(x=selected_time, y=selected_price,
                             text="📍 Selected", showarrow=True, arrowhead=2,
                             row=1, col=1)
    
    fig.update_layout(
        height=300,
        showlegend=False,
        template='plotly_dark',
        margin=dict(l=50, r=50, t=50, b=30),
        paper_bgcolor='rgba(26, 26, 46, 1)',
        plot_bgcolor='rgba(26, 26, 46, 1)'
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    
    return fig


# ============================================================================
# MAIN STREAMLIT APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">📊 GEX + DEX Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Gamma & Delta Exposure Analysis with Time Machine Backtest | By NYZTrade</div>', unsafe_allow_html=True)

    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    symbol = st.sidebar.selectbox(
        "Select Index",
        ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"],
        index=0
    )
    
    strikes_range = st.sidebar.slider(
        "Strikes Range (from ATM)",
        min_value=5,
        max_value=25,
        value=12,
        help="Number of strikes to analyze on each side of ATM"
    )
    
    expiry_index = st.sidebar.number_input(
        "Expiry Index",
        min_value=0,
        max_value=10,
        value=0,
        help="0 = Current weekly, 1 = Next weekly, etc."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📸 Data Capture")
    
    # Manual capture button
    if st.sidebar.button("📸 Capture Snapshot Now", type="primary"):
        st.session_state.force_capture = True
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=False)
    refresh_interval = st.sidebar.slider(
        "Refresh Interval (seconds)",
        min_value=30,
        max_value=300,
        value=60,
        disabled=not auto_refresh
    )
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Sidebar stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Session Stats")
    st.sidebar.metric("Snapshots Captured", len(st.session_state.snapshot_times))
    if st.session_state.snapshot_times:
        st.sidebar.text(f"First: {st.session_state.snapshot_times[0].strftime('%I:%M %p')}")
        st.sidebar.text(f"Last: {st.session_state.snapshot_times[-1].strftime('%I:%M %p')}")

    # ========================================================================
    # TIME SLIDER SECTION
    # ========================================================================
    historical_data = render_time_slider()
    
    # Show price history chart if we have data
    if len(st.session_state.snapshot_times) >= 2:
        st.markdown("#### 📈 Intraday Price & GEX History")
        history_chart = create_price_history_chart()
        if history_chart:
            st.plotly_chart(history_chart, use_container_width=True)

    # ========================================================================
    # DATA FETCHING / LOADING
    # ========================================================================
    
    if historical_data and not st.session_state.is_live_mode:
        # Use historical data
        df = historical_data['df']
        futures_ltp = historical_data['futures_ltp']
        market_info = historical_data['market_info']
        atm_info = historical_data['atm_info']
        flow_metrics = historical_data['flow_metrics']
        is_historical = True
        historical_time = st.session_state.snapshot_times[st.session_state.selected_time_index]
    else:
        # Fetch live data
        is_historical = False
        historical_time = None
        
        calculator = EnhancedGEXDEXCalculator()
        
        with st.spinner("Connecting to NSE..."):
            success, message = calculator.initialize_session()
            if not success:
                st.error(f"❌ {message}")
                st.stop()

        with st.spinner(f"Fetching {symbol} data..."):
            df, futures_ltp, market_info, atm_info, error = calculator.fetch_and_calculate_gex_dex(
                symbol, strikes_range, expiry_index
            )
        
        if error:
            st.error(f"❌ {error}")
            st.stop()
        
        if df is None or atm_info is None:
            st.error("❌ Failed to fetch data. Please try again.")
            st.stop()

        # Calculate flow metrics
        flow_metrics = calculate_dual_gex_dex_flow(df, futures_ltp)
        
        # Auto-capture snapshot if enabled
        if st.session_state.auto_capture or getattr(st.session_state, 'force_capture', False):
            captured = capture_snapshot(df, futures_ltp, market_info, atm_info, flow_metrics)
            if captured:
                st.toast("📸 Snapshot captured!", icon="✅")
            st.session_state.force_capture = False

    # ========================================================================
    # MARKET INFO SECTION
    # ========================================================================
    st.markdown("---")
    
    # Mode indicator
    if is_historical:
        st.warning(f"📜 **HISTORICAL MODE** - Viewing data from {historical_time.strftime('%I:%M:%S %p')}")
    elif market_info.get('fetch_method') == 'Demo Data':
        st.warning("""
        ⚠️ **DEMO DATA MODE** - NSE India blocks cloud server requests. 
        This demo shows how the dashboard works with simulated data.
        
        **For live data:** Run the app locally using `streamlit run app.py`
        """)
    else:
        st.success("🔴 **LIVE MODE** - Showing real-time data")
    
    st.subheader("💰 Market Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Spot Price",
            f"₹{market_info['spot_price']:,.2f}",
            help="Underlying spot price"
        )
    
    with col2:
        st.metric(
            "Futures LTP",
            f"₹{futures_ltp:,.2f}",
            delta=f"{market_info['basis']:+.2f} ({market_info['basis_pct']:+.3f}%)",
            help=f"Fetched via {market_info['fetch_method']}"
        )
    
    with col3:
        st.metric(
            "ATM Strike",
            f"{atm_info['atm_strike']:,.0f}",
            help="At-the-money strike"
        )
    
    with col4:
        st.metric(
            "ATM Straddle",
            f"₹{atm_info['atm_straddle_premium']:.2f}",
            help="ATM Call + Put premium"
        )
    
    with col5:
        st.metric(
            "Days to Expiry",
            f"{market_info['days_to_expiry']} days",
            help=f"Selected: {market_info['selected_expiry']}"
        )

    st.info(f"📅 **Selected Expiry:** {market_info['selected_expiry']} | ⏰ **Data Time:** {market_info['timestamp']} | 🔧 **Method:** {market_info['fetch_method']}")

    # ========================================================================
    # GEX + DEX FLOW ANALYSIS
    # ========================================================================
    st.markdown("---")
    st.subheader("📊 GEX + DEX Flow Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 GEX (Gamma Exposure)")
        gex_color = "🟢" if flow_metrics['gex_near_total'] > 0 else "🔴"
        st.markdown(f"**Near-term Flow:** {gex_color} {flow_metrics['gex_near_total']:.4f} B")
        st.markdown(f"**Bias:** {flow_metrics['gex_near_bias']}")
        st.markdown(f"**Total GEX:** {flow_metrics['gex_total_all']:.4f} B")
        
    with col2:
        st.markdown("### 📈 DEX (Delta Exposure)")
        dex_color = "🟢" if flow_metrics['dex_near_total'] > 0 else "🔴"
        st.markdown(f"**Near-term Flow:** {dex_color} {flow_metrics['dex_near_total']:.4f} B")
        st.markdown(f"**Bias:** {flow_metrics['dex_near_bias']}")
        st.markdown(f"**Total DEX:** {flow_metrics['dex_total_all']:.4f} B")
        
    with col3:
        st.markdown("### ⚡ Combined Signal")
        combined_color = "🟢" if flow_metrics['combined_signal'] > 0 else "🔴"
        st.markdown(f"**Combined Flow:** {combined_color} {flow_metrics['combined_signal']:.4f} B")
        st.markdown(f"**Overall Bias:** {flow_metrics['combined_bias']}")
        
        if flow_metrics['gex_near_bias'] != flow_metrics['dex_near_bias']:
            st.warning("⚠️ GEX and DEX showing divergence!")

    # ========================================================================
    # DETAILED METRICS TABLE
    # ========================================================================
    st.markdown("---")
    st.subheader("📋 Detailed Flow Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### GEX Metrics")
        gex_data = {
            "Metric": ["Positive GEX (Near)", "Negative GEX (Near)", "Net GEX (Near)", "Total Positive GEX", "Total Negative GEX", "Total Net GEX"],
            "Value (B)": [
                f"{flow_metrics['gex_near_positive']:.4f}",
                f"{flow_metrics['gex_near_negative']:.4f}",
                f"{flow_metrics['gex_near_total']:.4f}",
                f"{flow_metrics['gex_total_positive']:.4f}",
                f"{flow_metrics['gex_total_negative']:.4f}",
                f"{flow_metrics['gex_total_all']:.4f}"
            ]
        }
        st.dataframe(pd.DataFrame(gex_data), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("#### DEX Metrics")
        dex_data = {
            "Metric": ["Positive DEX (Near)", "Negative DEX (Near)", "Net DEX (Near)", "Total Positive DEX", "Total Negative DEX", "Total Net DEX"],
            "Value (B)": [
                f"{flow_metrics['dex_near_positive']:.4f}",
                f"{flow_metrics['dex_near_negative']:.4f}",
                f"{flow_metrics['dex_near_total']:.4f}",
                f"{flow_metrics['dex_total_positive']:.4f}",
                f"{flow_metrics['dex_total_negative']:.4f}",
                f"{flow_metrics['dex_total_all']:.4f}"
            ]
        }
        st.dataframe(pd.DataFrame(dex_data), hide_index=True, use_container_width=True)

    # ATM Straddle Info
    st.markdown("#### 💰 ATM Straddle Analysis")
    straddle_cols = st.columns(5)
    
    with straddle_cols[0]:
        st.metric("ATM Strike", f"{atm_info['atm_strike']:,.0f}")
    with straddle_cols[1]:
        st.metric("Call Premium", f"₹{atm_info['atm_call_premium']:.2f}")
    with straddle_cols[2]:
        st.metric("Put Premium", f"₹{atm_info['atm_put_premium']:.2f}")
    with straddle_cols[3]:
        st.metric("Straddle Premium", f"₹{atm_info['atm_straddle_premium']:.2f}")
    with straddle_cols[4]:
        upper_be = atm_info['atm_strike'] + atm_info['atm_straddle_premium']
        lower_be = atm_info['atm_strike'] - atm_info['atm_straddle_premium']
        st.metric("Breakevens", f"{lower_be:.0f} - {upper_be:.0f}")

    # ========================================================================
    # STRIKES ANALYSIS TABLE
    # ========================================================================
    st.markdown("---")
    st.subheader("📋 Strikes Near Futures LTP")
    
    df_unique = df.drop_duplicates(subset=['Strike']).copy()
    positive_gex_df = df_unique[df_unique['Net_GEX_B'] > 0].copy()
    positive_gex_df['Distance'] = abs(positive_gex_df['Strike'] - futures_ltp)
    positive_gex_strikes = positive_gex_df.nsmallest(5, 'Distance')

    negative_gex_df = df_unique[df_unique['Net_GEX_B'] < 0].copy()
    negative_gex_df['Distance'] = abs(negative_gex_df['Strike'] - futures_ltp)
    negative_gex_strikes = negative_gex_df.nsmallest(5, 'Distance')

    relevant_strikes = pd.concat([positive_gex_strikes, negative_gex_strikes]).sort_values('Strike')
    
    display_df = relevant_strikes[['Strike', 'Net_GEX_B', 'Net_DEX_B', 'Total_Volume', 'Call_OI', 'Put_OI']].copy()
    display_df['Position'] = display_df['Strike'].apply(lambda x: "🔼 ABOVE" if x > futures_ltp else ("⚡ ATM" if abs(x - futures_ltp) < 10 else "🔽 BELOW"))
    display_df['Net_GEX_B'] = display_df['Net_GEX_B'].apply(lambda x: f"🟢 +{x:.4f}B" if x > 0.001 else (f"🔴 {x:.4f}B" if x < -0.001 else f"⚪ {x:.4f}B"))
    display_df['Net_DEX_B'] = display_df['Net_DEX_B'].apply(lambda x: f"🟢 +{x:.4f}B" if x > 0.001 else (f"🔴 {x:.4f}B" if x < -0.001 else f"⚪ {x:.4f}B"))
    display_df['Strike'] = display_df['Strike'].apply(lambda x: f"{x:,.0f}")
    display_df['Total_Volume'] = display_df['Total_Volume'].apply(lambda x: f"{x:,.0f}")
    display_df['Call_OI'] = display_df['Call_OI'].apply(lambda x: f"{x:,.0f}")
    display_df['Put_OI'] = display_df['Put_OI'].apply(lambda x: f"{x:,.0f}")
    
    display_df = display_df[['Position', 'Strike', 'Net_GEX_B', 'Net_DEX_B', 'Total_Volume', 'Call_OI', 'Put_OI']]
    display_df.columns = ['Position', 'Strike', 'Net GEX', 'Net DEX', 'Volume', 'Call OI', 'Put OI']
    
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    # ========================================================================
    # CHARTS SECTION
    # ========================================================================
    st.markdown("---")
    st.subheader("📈 Interactive Charts")
    
    fig = create_enhanced_dashboard(df, futures_ltp, symbol, flow_metrics, market_info['fetch_method'], atm_info, is_historical, historical_time)
    st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # TRADING STRATEGIES SECTION
    # ========================================================================
    st.markdown("---")
    st.subheader("💼 Recommended Trading Strategies")
    
    strategies, setup_info = generate_trading_strategies(df, futures_ltp, flow_metrics, atm_info)
    
    st.markdown("#### 📊 Market Setup Analysis")
    setup_cols = st.columns(4)
    
    with setup_cols[0]:
        gex_color = "green" if setup_info['gex_bias'] > 0 else "red"
        st.markdown(f"**GEX Flow:** <span style='color:{gex_color}'>{setup_info['gex_bias']:.2f}</span>", unsafe_allow_html=True)
    with setup_cols[1]:
        dex_color = "green" if setup_info['dex_bias'] > 0 else "red"
        st.markdown(f"**DEX Flow:** <span style='color:{dex_color}'>{setup_info['dex_bias']:.2f}</span>", unsafe_allow_html=True)
    with setup_cols[2]:
        st.markdown(f"**Support:** {setup_info['support_strike']:,.0f}")
    with setup_cols[3]:
        st.markdown(f"**Resistance:** {setup_info['resistance_strike']:,.0f}")

    for idx, strategy in enumerate(strategies, 1):
        with st.expander(f"**Strategy #{idx}: {strategy['name']}** ({strategy['category']})", expanded=idx==1):
            st.markdown(f"**Rationale:** {strategy['rationale']}")
            st.markdown(f"**Setup:** `{strategy['setup']}`")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Max Profit:** {strategy['max_profit']}")
            with col2:
                st.markdown(f"**Max Loss:** {strategy['max_loss']}")
            with col3:
                st.markdown(f"**Risk Level:** {strategy['risk_level']}")
            
            st.info(f"**Conditions:** {strategy['conditions']}")

    # ========================================================================
    # GEX INTERPRETATION GUIDE
    # ========================================================================
    st.markdown("---")
    st.subheader("📖 GEX Flow Interpretation Guide")
    
    with st.expander("Learn how to interpret GEX & DEX signals", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### ✅ Positive GEX Flow (Sideways to Bullish)
            - 5 closest strikes with **positive Net GEX** near spot
            - Market makers delta hedge by **BUYING** underlying on dips
            - Acts as price **SUPPORT** → sideways to bullish movement
            - **STRATEGY:** Sell premium (Iron Condor, Credit Spreads, Short Straddle)
            """)
            
        with col2:
            st.markdown("""
            ### ❌ Negative GEX Flow (Bearish & High Volatility)
            - 5 closest strikes with **negative Net GEX** near spot
            - Market makers delta hedge by **SELLING** underlying on rallies
            - Acts as price **RESISTANCE** → bearish and volatile movement
            - **STRATEGY:** Buy volatility (Long Straddle, Long Options)
            """)
        
        st.markdown("""
        ### ⚖️ Neutral GEX
        - Balanced positive and negative GEX
        - No strong hedging bias from market makers
        - Follow **DEX (Delta) bias** for directional plays
        """)

    # ========================================================================
    # RISK MANAGEMENT
    # ========================================================================
    st.markdown("---")
    st.subheader("⚠️ Risk Management Rules")
    
    with st.expander("Important risk management guidelines", expanded=False):
        st.markdown(f"""
        ### 🛡️ Position Sizing
        - Never risk more than **2% of capital** per trade
        - For spreads: Risk defined by strike width minus premium
        - For long options: Max loss = Premium paid
        - For short straddles: **USE STOP LOSSES** or protective wings

        ### 🎯 Entry Timing
        - Wait for price to approach key GEX support/resistance levels
        - Enter when combined GEX+DEX bias aligns with your strategy
        - Avoid trading during first 15 mins and last 30 mins

        ### 🚪 Exit Rules
        - Take profit at **50-70%** of max profit for spreads
        - Use trailing stops for long options (20-30% of unrealized profit)
        - Exit immediately if GEX/DEX bias changes significantly
        - For short straddle: Exit if price moves > ₹{atm_info['atm_straddle_premium']*0.5:.2f} from ATM

        ### ⏰ Time Decay
        - Selling strategies: Theta works in your favor
        - Buying strategies: Monitor theta - don't hold too close to expiry
        - Weekly options: Higher gamma risk, faster decay

        ### 📊 Monitoring
        - Check GEX+DEX every 1-3 hours during market
        - Watch for changes in flow metrics (OI changes)
        - If combined bias flips, reassess positions immediately
        """)

    # ========================================================================
    # RAW DATA SECTION
    # ========================================================================
    st.markdown("---")
    st.subheader("📁 Raw Data")
    
    with st.expander("View complete option chain data"):
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False)
        timestamp_str = historical_time.strftime('%Y%m%d_%H%M%S') if is_historical else datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol}_GEX_DEX_Analysis_{timestamp_str}.csv"
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )

    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>📊 GEX + DEX Analysis Dashboard with Time Machine</strong> | Created by <strong>NYZTrade</strong></p>
        <p>⚠️ <em>Disclaimer: This tool is for educational purposes only. Always consult with a financial advisor before trading.</em></p>
        <p>📺 Subscribe to <a href='https://youtube.com/@NYZTrade' target='_blank'>NYZTrade YouTube Channel</a> for more content!</p>
    </div>
    """, unsafe_allow_html=True)

    # Auto-refresh logic
    if auto_refresh and st.session_state.is_live_mode:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
