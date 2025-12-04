# ============================================================================
# ADVANCED GEX + DEX ANALYSIS - STREAMLIT DASHBOARD
# WITH TIME MACHINE FOR BACKTESTING
# Created by NYZTrade - Options Analytics
# ============================================================================
# 
# FOR LIVE DATA: Run locally with `streamlit run app.py`
# NSE blocks cloud server IPs, so live data only works on local machines.
#
# ============================================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import warnings
import time
import json

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="GEX + DEX Analysis | NYZTrade",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #6c5ce7;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: bold;
    }
    .live-badge {
        background: linear-gradient(90deg, #00b894, #00cec9);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .demo-badge {
        background: linear-gradient(90deg, #fdcb6e, #e17055);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .hist-badge {
        background: linear-gradient(90deg, #a29bfe, #6c5ce7);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'data_snapshots': {},
        'snapshot_times': [],
        'selected_time_index': None,
        'is_live_mode': True,
        'last_capture_time': None,
        'auto_capture': True,
        'capture_interval': 3,
        'force_capture': False,
        'nse_session': None,
        'session_initialized': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ============================================================================
# BLACK-SCHOLES CALCULATOR
# ============================================================================

class BlackScholesCalculator:
    """Calculate option Greeks using Black-Scholes model"""
    
    @staticmethod
    def calculate_d1(S, K, T, r, sigma):
        """Calculate d1 parameter for Black-Scholes"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            return d1
        except:
            return 0

    @staticmethod
    def calculate_d2(S, K, T, r, sigma):
        """Calculate d2 parameter for Black-Scholes"""
        if T <= 0 or sigma <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            d2 = d1 - sigma * np.sqrt(T)
            return d2
        except:
            return 0

    @staticmethod
    def calculate_gamma(S, K, T, r, sigma):
        """Calculate option gamma (same for calls and puts)"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            n_prime_d1 = norm.pdf(d1)
            gamma = n_prime_d1 / (S * sigma * np.sqrt(T))
            return gamma
        except:
            return 0

    @staticmethod
    def calculate_call_delta(S, K, T, r, sigma):
        """Calculate call option delta"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            return norm.cdf(d1)
        except:
            return 0

    @staticmethod
    def calculate_put_delta(S, K, T, r, sigma):
        """Calculate put option delta"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            return norm.cdf(d1) - 1
        except:
            return 0

    @staticmethod
    def calculate_vega(S, K, T, r, sigma):
        """Calculate option vega"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            return vega
        except:
            return 0

    @staticmethod
    def calculate_theta_call(S, K, T, r, sigma):
        """Calculate call option theta"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            d2 = BlackScholesCalculator.calculate_d2(S, K, T, r, sigma)
            term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
            theta = (term1 - term2) / 365
            return theta
        except:
            return 0


# ============================================================================
# NSE DATA FETCHER - LIVE DATA
# ============================================================================

class NSEDataFetcher:
    """Fetch live option chain data from NSE India"""
    
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.option_chain_url = "https://www.nseindia.com/api/option-chain-indices"
        self.quote_url = "https://www.nseindia.com/api/quote-derivative"
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.cookies_set = False
        self.risk_free_rate = 0.07
        self.bs_calc = BlackScholesCalculator()

    def initialize_session(self):
        """Initialize session with NSE website to get cookies"""
        try:
            # First request to get cookies
            response = self.session.get(
                self.base_url,
                timeout=10,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                # Update headers for API calls
                self.session.headers.update({
                    'Accept': 'application/json, text/plain, */*',
                    'Referer': 'https://www.nseindia.com/option-chain',
                    'X-Requested-With': 'XMLHttpRequest',
                })
                self.cookies_set = True
                time.sleep(0.5)  # Small delay
                return True, "Session initialized successfully"
            else:
                return False, f"Failed to initialize: Status {response.status_code}"
                
        except requests.exceptions.Timeout:
            return False, "Connection timeout - NSE server not responding"
        except requests.exceptions.ConnectionError:
            return False, "Connection error - Check internet connection"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def fetch_option_chain(self, symbol="NIFTY"):
        """Fetch option chain data from NSE"""
        if not self.cookies_set:
            success, msg = self.initialize_session()
            if not success:
                return None, msg
        
        try:
            url = f"{self.option_chain_url}?symbol={symbol}"
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 401:
                # Session expired, reinitialize
                self.cookies_set = False
                success, msg = self.initialize_session()
                if success:
                    response = self.session.get(url, timeout=10)
                else:
                    return None, msg
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'records' in data:
                        return data, None
                    else:
                        return None, "Invalid response format"
                except json.JSONDecodeError:
                    return None, "Failed to parse JSON response"
            else:
                return None, f"HTTP Error: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return None, "Request timeout"
        except Exception as e:
            return None, f"Error: {str(e)}"

    def fetch_futures_price(self, symbol, spot_price, expiry_date):
        """
        Fetch futures price using multiple methods:
        1. Direct from Groww.in
        2. Calculate from Put-Call Parity
        3. Cost of Carry model
        """
        # Method 1: Try Groww.in
        futures_ltp = self._fetch_from_groww(symbol)
        if futures_ltp and self._validate_price(symbol, futures_ltp):
            return futures_ltp, "Groww.in"
        
        # Method 2: Put-Call Parity (from ATM options)
        futures_ltp = self._calculate_from_pcp(symbol, spot_price, expiry_date)
        if futures_ltp and self._validate_price(symbol, futures_ltp):
            return futures_ltp, "Put-Call Parity"
        
        # Method 3: Cost of Carry
        days_to_expiry = self._get_days_to_expiry(expiry_date)
        futures_ltp = spot_price * np.exp(self.risk_free_rate * days_to_expiry / 365)
        return futures_ltp, "Cost of Carry"

    def _fetch_from_groww(self, symbol):
        """Fetch futures price from Groww.in"""
        try:
            symbol_map = {
                'NIFTY': 'nifty',
                'BANKNIFTY': 'banknifty',
                'FINNIFTY': 'finnifty',
                'MIDCPNIFTY': 'midcpnifty'
            }
            
            groww_symbol = symbol_map.get(symbol, 'nifty')
            url = f"https://groww.in/v1/api/stocks_fo_data/v1/derivatives/futures/{groww_symbol}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            }
            
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if 'ltp' in data:
                    return float(data['ltp'])
            
            # Try alternate URL
            url2 = f"https://groww.in/futures/{groww_symbol}"
            response = requests.get(url2, headers={'User-Agent': headers['User-Agent']}, timeout=5)
            
            if response.status_code == 200:
                patterns = [
                    r'"ltp":\s*([0-9.]+)',
                    r'"lastPrice":\s*([0-9.]+)',
                    r'"close":\s*([0-9.]+)',
                ]
                for pattern in patterns:
                    matches = re.findall(pattern, response.text)
                    if matches:
                        for match in matches:
                            price = float(match)
                            if self._validate_price(symbol, price):
                                return price
            
            return None
        except:
            return None

    def _calculate_from_pcp(self, symbol, spot_price, expiry_date):
        """Calculate synthetic futures from Put-Call Parity"""
        try:
            data, error = self.fetch_option_chain(symbol)
            if error or not data:
                return None
            
            records = data['records']
            atm_strike = None
            min_diff = float('inf')
            
            for item in records.get('data', []):
                if item.get('expiryDate') != expiry_date:
                    continue
                strike = item.get('strikePrice', 0)
                diff = abs(strike - spot_price)
                if diff < min_diff:
                    min_diff = diff
                    atm_strike = strike
                    ce = item.get('CE', {})
                    pe = item.get('PE', {})
            
            if atm_strike and ce and pe:
                call_price = ce.get('lastPrice', 0)
                put_price = pe.get('lastPrice', 0)
                
                if call_price > 0 and put_price > 0:
                    days = self._get_days_to_expiry(expiry_date)
                    r = self.risk_free_rate
                    T = days / 365
                    
                    # F = K + (C - P) * e^(rT)
                    futures = atm_strike + (call_price - put_price) * np.exp(r * T)
                    return futures
            
            return None
        except:
            return None

    def _validate_price(self, symbol, price):
        """Validate if price is within expected range"""
        ranges = {
            'NIFTY': (15000, 35000),
            'BANKNIFTY': (35000, 75000),
            'FINNIFTY': (15000, 35000),
            'MIDCPNIFTY': (5000, 25000)
        }
        min_p, max_p = ranges.get(symbol, (5000, 100000))
        return min_p < price < max_p

    def _get_days_to_expiry(self, expiry_str):
        """Calculate days to expiry"""
        try:
            expiry = datetime.strptime(expiry_str, "%d-%b-%Y")
            days = (expiry - datetime.now()).days
            return max(days, 1)
        except:
            return 7

    def get_contract_specs(self, symbol):
        """Get contract specifications"""
        specs = {
            'NIFTY': {'lot_size': 25, 'strike_interval': 50},
            'BANKNIFTY': {'lot_size': 15, 'strike_interval': 100},
            'FINNIFTY': {'lot_size': 40, 'strike_interval': 50},
            'MIDCPNIFTY': {'lot_size': 75, 'strike_interval': 25}
        }
        return specs.get(symbol, specs['NIFTY'])


# ============================================================================
# GEX DEX CALCULATOR
# ============================================================================

class GEXDEXCalculator:
    """Calculate Gamma Exposure (GEX) and Delta Exposure (DEX)"""
    
    def __init__(self):
        self.fetcher = NSEDataFetcher()
        self.bs_calc = BlackScholesCalculator()
        self.risk_free_rate = 0.07
        self.use_demo_data = False

    def initialize(self):
        """Initialize the calculator"""
        success, msg = self.fetcher.initialize_session()
        if not success:
            self.use_demo_data = True
        return success, msg

    def calculate_time_to_expiry(self, expiry_str):
        """Calculate time to expiry in years"""
        try:
            expiry = datetime.strptime(expiry_str, "%d-%b-%Y")
            days = (expiry - datetime.now()).days
            T = max(days / 365, 1/365)  # Minimum 1 day
            return T, max(days, 1)
        except:
            return 7/365, 7

    def fetch_and_calculate(self, symbol="NIFTY", strikes_range=10, expiry_index=0):
        """Main function to fetch data and calculate GEX/DEX"""
        
        if self.use_demo_data:
            return self._generate_demo_data(symbol, strikes_range)
        
        # Fetch option chain
        data, error = self.fetcher.fetch_option_chain(symbol)
        
        if error or not data:
            self.use_demo_data = True
            return self._generate_demo_data(symbol, strikes_range)
        
        try:
            records = data['records']
            spot_price = records.get('underlyingValue', 0)
            timestamp = records.get('timestamp', datetime.now().strftime('%d-%b-%Y %H:%M:%S'))
            expiry_dates = records.get('expiryDates', [])
            
            if not expiry_dates or spot_price == 0:
                return self._generate_demo_data(symbol, strikes_range)
            
            # Select expiry
            selected_expiry = expiry_dates[min(expiry_index, len(expiry_dates) - 1)]
            T, days_to_expiry = self.calculate_time_to_expiry(selected_expiry)
            
            # Get futures price
            futures_ltp, fetch_method = self.fetcher.fetch_futures_price(
                symbol, spot_price, selected_expiry
            )
            
            # Get contract specs
            specs = self.fetcher.get_contract_specs(symbol)
            lot_size = specs['lot_size']
            strike_interval = specs['strike_interval']
            
            # Process option chain data
            all_strikes = []
            processed = set()
            atm_strike = None
            min_diff = float('inf')
            atm_call_premium = 0
            atm_put_premium = 0
            
            for item in records.get('data', []):
                if item.get('expiryDate') != selected_expiry:
                    continue
                
                strike = item.get('strikePrice', 0)
                if strike == 0 or strike in processed:
                    continue
                
                processed.add(strike)
                
                # Filter by strikes range
                distance = abs(strike - futures_ltp) / strike_interval
                if distance > strikes_range:
                    continue
                
                ce = item.get('CE', {})
                pe = item.get('PE', {})
                
                # Extract data
                call_oi = ce.get('openInterest', 0) or 0
                put_oi = pe.get('openInterest', 0) or 0
                call_oi_change = ce.get('changeinOpenInterest', 0) or 0
                put_oi_change = pe.get('changeinOpenInterest', 0) or 0
                call_volume = ce.get('totalTradedVolume', 0) or 0
                put_volume = pe.get('totalTradedVolume', 0) or 0
                call_iv = ce.get('impliedVolatility', 0) or 15
                put_iv = pe.get('impliedVolatility', 0) or 15
                call_ltp = ce.get('lastPrice', 0) or 0
                put_ltp = pe.get('lastPrice', 0) or 0
                call_bid = ce.get('bidPrice', 0) or 0
                call_ask = ce.get('askPrice', 0) or 0
                put_bid = pe.get('bidPrice', 0) or 0
                put_ask = pe.get('askPrice', 0) or 0
                
                # Track ATM strike
                diff = abs(strike - futures_ltp)
                if diff < min_diff:
                    min_diff = diff
                    atm_strike = strike
                    atm_call_premium = call_ltp
                    atm_put_premium = put_ltp
                
                # Calculate Greeks
                call_iv_dec = max(call_iv / 100, 0.05)
                put_iv_dec = max(put_iv / 100, 0.05)
                
                call_gamma = self.bs_calc.calculate_gamma(futures_ltp, strike, T, self.risk_free_rate, call_iv_dec)
                put_gamma = self.bs_calc.calculate_gamma(futures_ltp, strike, T, self.risk_free_rate, put_iv_dec)
                call_delta = self.bs_calc.calculate_call_delta(futures_ltp, strike, T, self.risk_free_rate, call_iv_dec)
                put_delta = self.bs_calc.calculate_put_delta(futures_ltp, strike, T, self.risk_free_rate, put_iv_dec)
                call_vega = self.bs_calc.calculate_vega(futures_ltp, strike, T, self.risk_free_rate, call_iv_dec)
                put_vega = self.bs_calc.calculate_vega(futures_ltp, strike, T, self.risk_free_rate, put_iv_dec)
                
                # Calculate GEX (in Billions)
                # GEX = OI × Gamma × Spot² × Contract Size / 1B
                gex_mult = futures_ltp * futures_ltp * lot_size / 1_000_000_000
                call_gex = call_oi * call_gamma * gex_mult
                put_gex = -put_oi * put_gamma * gex_mult  # Negative for puts
                
                # Calculate DEX (in Billions)
                # DEX = OI × Delta × Spot × Contract Size / 1B
                dex_mult = futures_ltp * lot_size / 1_000_000_000
                call_dex = call_oi * call_delta * dex_mult
                put_dex = put_oi * put_delta * dex_mult
                
                # Flow GEX (based on OI change)
                call_flow_gex = call_oi_change * call_gamma * gex_mult
                put_flow_gex = -put_oi_change * put_gamma * gex_mult
                
                # Flow DEX (based on OI change)
                call_flow_dex = call_oi_change * call_delta * dex_mult
                put_flow_dex = put_oi_change * put_delta * dex_mult
                
                # Vega Exposure
                call_vex = call_oi * call_vega * lot_size / 1_000_000
                put_vex = put_oi * put_vega * lot_size / 1_000_000
                
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
                    'Call_Bid': call_bid,
                    'Call_Ask': call_ask,
                    'Put_Bid': put_bid,
                    'Put_Ask': put_ask,
                    'Call_Gamma': call_gamma,
                    'Put_Gamma': put_gamma,
                    'Call_Delta': call_delta,
                    'Put_Delta': put_delta,
                    'Call_Vega': call_vega,
                    'Put_Vega': put_vega,
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
                    'Net_Flow_DEX': call_flow_dex + put_flow_dex,
                    'Call_VEX': call_vex,
                    'Put_VEX': put_vex,
                    'Net_VEX': call_vex + put_vex
                })
            
            if not all_strikes:
                return self._generate_demo_data(symbol, strikes_range)
            
            # Create DataFrame
            df = pd.DataFrame(all_strikes).sort_values('Strike').reset_index(drop=True)
            
            # Add _B suffix columns for compatibility
            for col in ['Call_GEX', 'Put_GEX', 'Net_GEX', 'Call_DEX', 'Put_DEX', 'Net_DEX',
                        'Call_Flow_GEX', 'Put_Flow_GEX', 'Net_Flow_GEX',
                        'Call_Flow_DEX', 'Put_Flow_DEX', 'Net_Flow_DEX']:
                df[f'{col}_B'] = df[col]
            
            df['Total_Volume'] = df['Call_Volume'] + df['Put_Volume']
            df['Total_OI'] = df['Call_OI'] + df['Put_OI']
            df['PCR'] = np.where(df['Call_OI'] > 0, df['Put_OI'] / df['Call_OI'], 0)
            
            # Hedging Pressure
            max_gex = df['Net_GEX_B'].abs().max()
            df['Hedging_Pressure'] = (df['Net_GEX_B'] / max_gex * 100) if max_gex > 0 else 0
            
            # ATM info
            atm_info = {
                'atm_strike': atm_strike or df.iloc[len(df)//2]['Strike'],
                'atm_call_premium': atm_call_premium,
                'atm_put_premium': atm_put_premium,
                'atm_straddle_premium': atm_call_premium + atm_put_premium
            }
            
            # Market info
            basis = futures_ltp - spot_price
            basis_pct = (basis / spot_price * 100) if spot_price > 0 else 0
            
            market_info = {
                'spot_price': spot_price,
                'futures_ltp': futures_ltp,
                'basis': basis,
                'basis_pct': basis_pct,
                'fetch_method': fetch_method,
                'timestamp': timestamp,
                'expiry_dates': expiry_dates,
                'selected_expiry': selected_expiry,
                'days_to_expiry': days_to_expiry,
                'lot_size': lot_size
            }
            
            return df, futures_ltp, market_info, atm_info, None
            
        except Exception as e:
            return self._generate_demo_data(symbol, strikes_range)

    def _generate_demo_data(self, symbol="NIFTY", strikes_range=10):
        """Generate demo data when live data is not available"""
        np.random.seed(int(datetime.now().timestamp()) % 10000)
        
        # Demo spot prices
        spot_prices = {
            'NIFTY': 24250 + np.random.randn() * 50,
            'BANKNIFTY': 51850 + np.random.randn() * 100,
            'FINNIFTY': 23150 + np.random.randn() * 50,
            'MIDCPNIFTY': 12450 + np.random.randn() * 25
        }
        
        spot_price = spot_prices.get(symbol, 24250)
        specs = self.fetcher.get_contract_specs(symbol)
        lot_size = specs['lot_size']
        strike_interval = specs['strike_interval']
        
        futures_ltp = spot_price * 1.0008
        atm_strike = round(spot_price / strike_interval) * strike_interval
        T = 7 / 365
        
        all_strikes = []
        
        for i in range(-strikes_range, strikes_range + 1):
            strike = atm_strike + (i * strike_interval)
            dist = abs(i)
            
            # Realistic OI distribution
            base_oi = 400000 + np.random.randint(-100000, 100000)
            if i < 0:
                call_oi = int(base_oi * (0.4 + 0.2 * np.random.random()) * max(0.2, 1 - dist * 0.08))
                put_oi = int(base_oi * (1.1 + 0.3 * np.random.random()) * max(0.3, 1 - dist * 0.05))
            else:
                call_oi = int(base_oi * (1.1 + 0.3 * np.random.random()) * max(0.3, 1 - dist * 0.05))
                put_oi = int(base_oi * (0.4 + 0.2 * np.random.random()) * max(0.2, 1 - dist * 0.08))
            
            call_oi_change = int((np.random.random() - 0.5) * call_oi * 0.15)
            put_oi_change = int((np.random.random() - 0.5) * put_oi * 0.15)
            call_volume = int(call_oi * (0.05 + 0.1 * np.random.random()))
            put_volume = int(put_oi * (0.05 + 0.1 * np.random.random()))
            
            # IV with smile
            base_iv = 13 + dist * 0.35 + np.random.random() * 1.5
            call_iv = base_iv + (0.8 if i > 0 else -0.3)
            put_iv = base_iv + (0.8 if i < 0 else -0.3)
            
            # LTP
            if strike < spot_price:
                call_ltp = max(5, spot_price - strike + np.random.random() * 20)
                put_ltp = max(1, np.random.random() * 30 * max(0.1, 1 - dist * 0.12))
            else:
                call_ltp = max(1, np.random.random() * 30 * max(0.1, 1 - dist * 0.12))
                put_ltp = max(5, strike - spot_price + np.random.random() * 20)
            
            # Greeks
            call_iv_dec = call_iv / 100
            put_iv_dec = put_iv / 100
            
            call_gamma = self.bs_calc.calculate_gamma(futures_ltp, strike, T, self.risk_free_rate, call_iv_dec)
            put_gamma = self.bs_calc.calculate_gamma(futures_ltp, strike, T, self.risk_free_rate, put_iv_dec)
            call_delta = self.bs_calc.calculate_call_delta(futures_ltp, strike, T, self.risk_free_rate, call_iv_dec)
            put_delta = self.bs_calc.calculate_put_delta(futures_ltp, strike, T, self.risk_free_rate, put_iv_dec)
            
            # GEX/DEX
            gex_mult = futures_ltp * futures_ltp * lot_size / 1_000_000_000
            dex_mult = futures_ltp * lot_size / 1_000_000_000
            
            call_gex = call_oi * call_gamma * gex_mult
            put_gex = -put_oi * put_gamma * gex_mult
            call_dex = call_oi * call_delta * dex_mult
            put_dex = put_oi * put_delta * dex_mult
            
            call_flow_gex = call_oi_change * call_gamma * gex_mult
            put_flow_gex = -put_oi_change * put_gamma * gex_mult
            call_flow_dex = call_oi_change * call_delta * dex_mult
            put_flow_dex = put_oi_change * put_delta * dex_mult
            
            all_strikes.append({
                'Strike': strike,
                'Call_OI': call_oi, 'Put_OI': put_oi,
                'Call_OI_Change': call_oi_change, 'Put_OI_Change': put_oi_change,
                'Call_Volume': call_volume, 'Put_Volume': put_volume,
                'Call_IV': round(call_iv, 2), 'Put_IV': round(put_iv, 2),
                'Call_LTP': round(call_ltp, 2), 'Put_LTP': round(put_ltp, 2),
                'Call_Gamma': call_gamma, 'Put_Gamma': put_gamma,
                'Call_Delta': call_delta, 'Put_Delta': put_delta,
                'Call_GEX': call_gex, 'Put_GEX': put_gex, 'Net_GEX': call_gex + put_gex,
                'Call_DEX': call_dex, 'Put_DEX': put_dex, 'Net_DEX': call_dex + put_dex,
                'Call_Flow_GEX': call_flow_gex, 'Put_Flow_GEX': put_flow_gex, 'Net_Flow_GEX': call_flow_gex + put_flow_gex,
                'Call_Flow_DEX': call_flow_dex, 'Put_Flow_DEX': put_flow_dex, 'Net_Flow_DEX': call_flow_dex + put_flow_dex
            })
        
        df = pd.DataFrame(all_strikes).sort_values('Strike').reset_index(drop=True)
        
        for col in ['Call_GEX', 'Put_GEX', 'Net_GEX', 'Call_DEX', 'Put_DEX', 'Net_DEX',
                    'Call_Flow_GEX', 'Put_Flow_GEX', 'Net_Flow_GEX',
                    'Call_Flow_DEX', 'Put_Flow_DEX', 'Net_Flow_DEX']:
            df[f'{col}_B'] = df[col]
        
        df['Total_Volume'] = df['Call_Volume'] + df['Put_Volume']
        df['Total_OI'] = df['Call_OI'] + df['Put_OI']
        
        max_gex = df['Net_GEX_B'].abs().max()
        df['Hedging_Pressure'] = (df['Net_GEX_B'] / max_gex * 100) if max_gex > 0 else 0
        
        atm_row = df[df['Strike'] == atm_strike]
        if len(atm_row) > 0:
            atm_row = atm_row.iloc[0]
        else:
            atm_row = df.iloc[len(df)//2]
        
        atm_info = {
            'atm_strike': atm_strike,
            'atm_call_premium': atm_row['Call_LTP'],
            'atm_put_premium': atm_row['Put_LTP'],
            'atm_straddle_premium': atm_row['Call_LTP'] + atm_row['Put_LTP']
        }
        
        # Next Thursday for expiry
        today = datetime.now()
        days_to_thu = (3 - today.weekday()) % 7
        if days_to_thu == 0:
            days_to_thu = 7
        next_thu = today + timedelta(days=days_to_thu)
        
        market_info = {
            'spot_price': round(spot_price, 2),
            'futures_ltp': round(futures_ltp, 2),
            'basis': round(futures_ltp - spot_price, 2),
            'basis_pct': round((futures_ltp - spot_price) / spot_price * 100, 3),
            'fetch_method': 'Demo Data',
            'timestamp': datetime.now().strftime('%d-%b-%Y %H:%M:%S'),
            'expiry_dates': [next_thu.strftime("%d-%b-%Y")],
            'selected_expiry': next_thu.strftime("%d-%b-%Y"),
            'days_to_expiry': days_to_thu,
            'lot_size': lot_size
        }
        
        return df, futures_ltp, market_info, atm_info, None


# ============================================================================
# FLOW ANALYSIS
# ============================================================================

def calculate_flow_metrics(df, futures_ltp):
    """Calculate comprehensive GEX and DEX flow metrics"""
    df_unique = df.drop_duplicates(subset=['Strike']).sort_values('Strike').reset_index(drop=True)
    
    # GEX Flow - 5 positive + 5 negative closest to spot
    pos_gex = df_unique[df_unique['Net_GEX_B'] > 0].copy()
    if len(pos_gex) > 0:
        pos_gex['Dist'] = abs(pos_gex['Strike'] - futures_ltp)
        pos_gex = pos_gex.nsmallest(5, 'Dist')
    
    neg_gex = df_unique[df_unique['Net_GEX_B'] < 0].copy()
    if len(neg_gex) > 0:
        neg_gex['Dist'] = abs(neg_gex['Strike'] - futures_ltp)
        neg_gex = neg_gex.nsmallest(5, 'Dist')
    
    gex_near_pos = float(pos_gex['Net_GEX_B'].sum()) if len(pos_gex) > 0 else 0
    gex_near_neg = float(neg_gex['Net_GEX_B'].sum()) if len(neg_gex) > 0 else 0
    gex_near_total = gex_near_pos + gex_near_neg
    
    gex_total_pos = float(df_unique[df_unique['Net_GEX_B'] > 0]['Net_GEX_B'].sum())
    gex_total_neg = float(df_unique[df_unique['Net_GEX_B'] < 0]['Net_GEX_B'].sum())
    gex_total_all = gex_total_pos + gex_total_neg
    
    # DEX Flow
    above = df_unique[df_unique['Strike'] > futures_ltp].head(5)
    below = df_unique[df_unique['Strike'] < futures_ltp].tail(5)
    
    dex_near_pos = float(above['Net_DEX_B'].sum()) if len(above) > 0 else 0
    dex_near_neg = float(below['Net_DEX_B'].sum()) if len(below) > 0 else 0
    dex_near_total = dex_near_pos + dex_near_neg
    
    dex_total_pos = float(df_unique[df_unique['Net_DEX_B'] > 0]['Net_DEX_B'].sum())
    dex_total_neg = float(df_unique[df_unique['Net_DEX_B'] < 0]['Net_DEX_B'].sum())
    dex_total_all = dex_total_pos + dex_total_neg
    
    # Key Levels
    max_call_oi_strike = df_unique.loc[df_unique['Call_OI'].idxmax(), 'Strike'] if len(df_unique) > 0 else 0
    max_put_oi_strike = df_unique.loc[df_unique['Put_OI'].idxmax(), 'Strike'] if len(df_unique) > 0 else 0
    max_gex_strike = df_unique.loc[df_unique['Net_GEX_B'].abs().idxmax(), 'Strike'] if len(df_unique) > 0 else 0
    
    # PCR
    total_call_oi = df_unique['Call_OI'].sum()
    total_put_oi = df_unique['Put_OI'].sum()
    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1
    
    # Bias determination
    def get_bias(val, threshold=50):
        if val > threshold:
            return "🟢 STRONG BULLISH", "#00d4aa"
        elif val > 0:
            return "🟢 BULLISH", "#55efc4"
        elif val < -threshold:
            return "🔴 STRONG BEARISH", "#ff6b6b"
        elif val < 0:
            return "🔴 BEARISH", "#fab1a0"
        return "⚪ NEUTRAL", "#b2bec3"
    
    gex_bias, gex_color = get_bias(gex_near_total)
    dex_bias, dex_color = get_bias(dex_near_total)
    combined = (gex_near_total + dex_near_total) / 2
    combined_bias, combined_color = get_bias(combined)
    
    return {
        'gex_near_positive': gex_near_pos,
        'gex_near_negative': gex_near_neg,
        'gex_near_total': gex_near_total,
        'gex_total_positive': gex_total_pos,
        'gex_total_negative': gex_total_neg,
        'gex_total_all': gex_total_all,
        'gex_near_bias': gex_bias,
        'gex_near_color': gex_color,
        'dex_near_positive': dex_near_pos,
        'dex_near_negative': dex_near_neg,
        'dex_near_total': dex_near_total,
        'dex_total_positive': dex_total_pos,
        'dex_total_negative': dex_total_neg,
        'dex_total_all': dex_total_all,
        'dex_near_bias': dex_bias,
        'dex_near_color': dex_color,
        'combined_signal': combined,
        'combined_bias': combined_bias,
        'combined_color': combined_color,
        'max_call_oi_strike': max_call_oi_strike,
        'max_put_oi_strike': max_put_oi_strike,
        'max_gex_strike': max_gex_strike,
        'pcr': pcr,
        'total_call_oi': total_call_oi,
        'total_put_oi': total_put_oi
    }


# ============================================================================
# SNAPSHOT FUNCTIONS
# ============================================================================

def capture_snapshot(df, futures_ltp, market_info, atm_info, flow_metrics):
    """Capture current data as snapshot"""
    now = datetime.now().replace(microsecond=0)
    
    if st.session_state.last_capture_time:
        elapsed = (now - st.session_state.last_capture_time).total_seconds() / 60
        if elapsed < st.session_state.capture_interval:
            return False
    
    st.session_state.data_snapshots[now] = {
        'df': df.copy(),
        'futures_ltp': futures_ltp,
        'market_info': market_info.copy(),
        'atm_info': atm_info.copy(),
        'flow_metrics': flow_metrics.copy()
    }
    
    if now not in st.session_state.snapshot_times:
        st.session_state.snapshot_times.append(now)
        st.session_state.snapshot_times.sort()
    
    st.session_state.last_capture_time = now
    
    # Limit snapshots
    while len(st.session_state.snapshot_times) > 500:
        oldest = st.session_state.snapshot_times.pop(0)
        st.session_state.data_snapshots.pop(oldest, None)
    
    return True


# ============================================================================
# TIME MACHINE UI
# ============================================================================

def render_time_machine():
    """Render time machine UI"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown("### ⏰ Time Machine - Backtest")
    
    with col2:
        if st.session_state.is_live_mode:
            st.markdown('<span class="live-badge">🟢 LIVE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="hist-badge">📜 HISTORY</span>', unsafe_allow_html=True)
    
    with col3:
        if not st.session_state.is_live_mode:
            if st.button("🔴 Go Live", use_container_width=True):
                st.session_state.is_live_mode = True
                st.session_state.selected_time_index = None
                st.rerun()
    
    if not st.session_state.snapshot_times:
        st.info("📝 No snapshots yet. Data will be captured automatically every few minutes.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.auto_capture = st.checkbox("🔄 Auto-capture", value=st.session_state.auto_capture)
        with col2:
            st.session_state.capture_interval = st.selectbox(
                "Interval", [1, 2, 3, 5, 10],
                index=[1, 2, 3, 5, 10].index(st.session_state.capture_interval) if st.session_state.capture_interval in [1, 2, 3, 5, 10] else 2,
                format_func=lambda x: f"{x} min"
            )
        return None
    
    first = st.session_state.snapshot_times[0]
    last = st.session_state.snapshot_times[-1]
    st.caption(f"📊 {len(st.session_state.snapshot_times)} snapshots | {first.strftime('%I:%M %p')} → {last.strftime('%I:%M %p')}")
    
    if len(st.session_state.snapshot_times) > 1:
        labels = [t.strftime('%I:%M %p') for t in st.session_state.snapshot_times]
        
        current_idx = st.session_state.selected_time_index
        if current_idx is None:
            current_idx = len(st.session_state.snapshot_times) - 1
        
        selected = st.select_slider(
            "Select time point",
            options=list(range(len(st.session_state.snapshot_times))),
            value=current_idx,
            format_func=lambda x: labels[x]
        )
        
        if selected != len(st.session_state.snapshot_times) - 1:
            st.session_state.is_live_mode = False
            st.session_state.selected_time_index = selected
        
        st.markdown("**Quick Jump:**")
        cols = st.columns(7)
        presets = [("5m", 5), ("15m", 15), ("30m", 30), ("1h", 60), ("2h", 120), ("3h", 180), ("Start", 9999)]
        
        for i, (label, mins) in enumerate(presets):
            with cols[i]:
                if st.button(label, key=f"jump_{mins}", use_container_width=True):
                    if mins == 9999:
                        target_idx = 0
                    else:
                        target = datetime.now() - timedelta(minutes=mins)
                        target_idx = min(
                            range(len(st.session_state.snapshot_times)),
                            key=lambda x: abs((st.session_state.snapshot_times[x] - target).total_seconds())
                        )
                    st.session_state.selected_time_index = target_idx
                    st.session_state.is_live_mode = False
                    st.rerun()
    
    with st.expander("⚙️ Capture Settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.auto_capture = st.checkbox("Auto-capture", value=st.session_state.auto_capture)
        with col2:
            st.session_state.capture_interval = st.selectbox(
                "Interval", [1, 2, 3, 5, 10],
                index=[1, 2, 3, 5, 10].index(st.session_state.capture_interval) if st.session_state.capture_interval in [1, 2, 3, 5, 10] else 2,
                format_func=lambda x: f"{x} min",
                key="interval_expander"
            )
        with col3:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.data_snapshots = {}
                st.session_state.snapshot_times = []
                st.session_state.selected_time_index = None
                st.session_state.is_live_mode = True
                st.rerun()
    
    if not st.session_state.is_live_mode and st.session_state.selected_time_index is not None:
        sel_time = st.session_state.snapshot_times[st.session_state.selected_time_index]
        return st.session_state.data_snapshots.get(sel_time)
    
    return None


# ============================================================================
# CHARTS
# ============================================================================

def create_main_charts(df, futures_ltp, symbol, flow_metrics, atm_info, is_historical=False, hist_time=None):
    """Create main dashboard charts"""
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '📊 Net GEX by Strike', '📈 Net DEX by Strike',
            '🔄 GEX Flow (OI Change)', '📉 DEX Flow (OI Change)',
            '🎯 Hedging Pressure', '⚡ Combined Signal'
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.08
    )
    
    strikes = df['Strike'].values
    
    # Chart 1: Net GEX
    colors1 = ['#00d4aa' if x > 0 else '#ff6b6b' for x in df['Net_GEX_B']]
    fig.add_trace(
        go.Bar(y=strikes, x=df['Net_GEX_B'], orientation='h', marker_color=colors1, 
               name='Net GEX', hovertemplate='Strike: %{y}<br>GEX: %{x:.4f}B<extra></extra>'),
        row=1, col=1
    )
    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="#6c5ce7", line_width=2, row=1, col=1,
                  annotation_text=f"Futures: {futures_ltp:,.0f}")
    
    # Chart 2: Net DEX
    colors2 = ['#00d4aa' if x > 0 else '#ff6b6b' for x in df['Net_DEX_B']]
    fig.add_trace(
        go.Bar(y=strikes, x=df['Net_DEX_B'], orientation='h', marker_color=colors2,
               name='Net DEX', hovertemplate='Strike: %{y}<br>DEX: %{x:.4f}B<extra></extra>'),
        row=1, col=2
    )
    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="#6c5ce7", line_width=2, row=1, col=2)
    
    # Chart 3: GEX Flow
    colors3 = ['#00d4aa' if x > 0 else '#ff6b6b' for x in df['Net_Flow_GEX_B']]
    fig.add_trace(
        go.Bar(y=strikes, x=df['Net_Flow_GEX_B'], orientation='h', marker_color=colors3,
               name='GEX Flow', hovertemplate='Strike: %{y}<br>Flow: %{x:.4f}B<extra></extra>'),
        row=2, col=1
    )
    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="#6c5ce7", line_width=2, row=2, col=1)
    
    # Chart 4: DEX Flow
    colors4 = ['#00d4aa' if x > 0 else '#ff6b6b' for x in df['Net_Flow_DEX_B']]
    fig.add_trace(
        go.Bar(y=strikes, x=df['Net_Flow_DEX_B'], orientation='h', marker_color=colors4,
               name='DEX Flow', hovertemplate='Strike: %{y}<br>Flow: %{x:.4f}B<extra></extra>'),
        row=2, col=2
    )
    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="#6c5ce7", line_width=2, row=2, col=2)
    
    # Chart 5: Hedging Pressure
    fig.add_trace(
        go.Bar(y=strikes, x=df['Hedging_Pressure'], orientation='h',
               marker=dict(color=df['Hedging_Pressure'], colorscale='RdYlGn', cmin=-100, cmax=100),
               name='Pressure', hovertemplate='Strike: %{y}<br>Pressure: %{x:.1f}%<extra></extra>'),
        row=3, col=1
    )
    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="#6c5ce7", line_width=2, row=3, col=1)
    
    # Chart 6: Combined
    df['Combined'] = (df['Net_GEX_B'] + df['Net_DEX_B']) / 2
    colors6 = ['#00d4aa' if x > 0 else '#ff6b6b' for x in df['Combined']]
    fig.add_trace(
        go.Bar(y=strikes, x=df['Combined'], orientation='h', marker_color=colors6,
               name='Combined', hovertemplate='Strike: %{y}<br>Signal: %{x:.4f}B<extra></extra>'),
        row=3, col=2
    )
    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="#6c5ce7", line_width=2, row=3, col=2)
    
    # Add key level annotations
    if flow_metrics['max_call_oi_strike'] > 0:
        fig.add_hline(y=flow_metrics['max_call_oi_strike'], line_dash="dot", line_color="green", 
                      line_width=1, row=1, col=1, annotation_text="Max Call OI")
    if flow_metrics['max_put_oi_strike'] > 0:
        fig.add_hline(y=flow_metrics['max_put_oi_strike'], line_dash="dot", line_color="red",
                      line_width=1, row=1, col=1, annotation_text="Max Put OI")
    
    mode_str = f"📜 HISTORICAL - {hist_time.strftime('%I:%M %p')}" if is_historical else "🔴 LIVE"
    
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol} GEX + DEX Analysis</b> ({mode_str})<br><sup>Futures: ₹{futures_ltp:,.2f} | {flow_metrics['combined_bias']} | PCR: {flow_metrics['pcr']:.2f}</sup>",
            x=0.5,
            font=dict(size=18)
        ),
        height=1000,
        showlegend=False,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_oi_chart(df, futures_ltp, symbol):
    """Create Open Interest chart"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Call vs Put OI', 'OI Change'))
    
    strikes = df['Strike'].values
    
    # OI
    fig.add_trace(
        go.Bar(y=strikes, x=df['Call_OI'], orientation='h', marker_color='#00d4aa', name='Call OI'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(y=strikes, x=-df['Put_OI'], orientation='h', marker_color='#ff6b6b', name='Put OI'),
        row=1, col=1
    )
    
    # OI Change
    fig.add_trace(
        go.Bar(y=strikes, x=df['Call_OI_Change'], orientation='h', marker_color='#55efc4', name='Call Δ'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(y=strikes, x=-df['Put_OI_Change'], orientation='h', marker_color='#fab1a0', name='Put Δ'),
        row=1, col=2
    )
    
    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="#6c5ce7", line_width=2, row=1, col=1)
    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="#6c5ce7", line_width=2, row=1, col=2)
    
    fig.update_layout(
        title=f"<b>{symbol} Open Interest Analysis</b>",
        height=500,
        showlegend=True,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        barmode='overlay'
    )
    
    return fig


def create_straddle_chart(atm_info, futures_ltp):
    """Create ATM straddle payoff chart"""
    atm = atm_info['atm_strike']
    call_prem = atm_info['atm_call_premium']
    put_prem = atm_info['atm_put_premium']
    straddle_prem = atm_info['atm_straddle_premium']
    
    strikes = np.linspace(atm * 0.92, atm * 1.08, 100)
    
    call_payoff = np.maximum(strikes - atm, 0) - call_prem
    put_payoff = np.maximum(atm - strikes, 0) - put_prem
    straddle_payoff = call_payoff + put_payoff
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=strikes, y=straddle_payoff, mode='lines', name='Straddle',
                             line=dict(color='#6c5ce7', width=3)))
    fig.add_trace(go.Scatter(x=strikes, y=call_payoff, mode='lines', name='Call',
                             line=dict(color='#00d4aa', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=strikes, y=put_payoff, mode='lines', name='Put',
                             line=dict(color='#ff6b6b', width=2, dash='dot')))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=atm, line_color="#6c5ce7", annotation_text=f"ATM: {atm:,.0f}")
    fig.add_vline(x=futures_ltp, line_color="yellow", line_dash="dash")
    
    upper_be = atm + straddle_prem
    lower_be = atm - straddle_prem
    fig.add_vline(x=upper_be, line_color="orange", line_dash="dot", annotation_text=f"BE: {upper_be:,.0f}")
    fig.add_vline(x=lower_be, line_color="orange", line_dash="dot", annotation_text=f"BE: {lower_be:,.0f}")
    
    fig.update_layout(
        title=f"<b>ATM Straddle Payoff</b><br><sup>Strike: {atm:,.0f} | Premium: ₹{straddle_prem:.2f}</sup>",
        xaxis_title="Spot Price at Expiry",
        yaxis_title="Profit/Loss (₹)",
        height=400,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_history_chart():
    """Create intraday history chart"""
    if len(st.session_state.snapshot_times) < 2:
        return None
    
    times, prices, gex_vals = [], [], []
    
    for t in st.session_state.snapshot_times:
        if t in st.session_state.data_snapshots:
            times.append(t)
            prices.append(st.session_state.data_snapshots[t]['futures_ltp'])
            gex_vals.append(st.session_state.data_snapshots[t]['flow_metrics']['gex_near_total'])
    
    if len(times) < 2:
        return None
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                        vertical_spacing=0.08, subplot_titles=('Futures Price', 'GEX Flow'))
    
    fig.add_trace(go.Scatter(x=times, y=prices, mode='lines+markers',
                             line=dict(color='#6c5ce7', width=2), marker=dict(size=4)), row=1, col=1)
    
    colors = ['#00d4aa' if x > 0 else '#ff6b6b' for x in gex_vals]
    fig.add_trace(go.Bar(x=times, y=gex_vals, marker_color=colors), row=2, col=1)
    
    if not st.session_state.is_live_mode and st.session_state.selected_time_index is not None:
        sel_time = st.session_state.snapshot_times[st.session_state.selected_time_index]
        if sel_time in st.session_state.data_snapshots:
            fig.add_vline(x=sel_time, line_dash="dash", line_color="orange", line_width=2)
    
    fig.update_layout(height=280, showlegend=False, template='plotly_dark',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=50, r=30, t=40, b=30))
    
    return fig


# ============================================================================
# STRATEGIES
# ============================================================================

def generate_strategies(flow_metrics, atm_info, market_info):
    """Generate trading strategy suggestions"""
    strategies = []
    gex = flow_metrics['gex_near_total']
    dex = flow_metrics['dex_near_total']
    pcr = flow_metrics['pcr']
    straddle = atm_info['atm_straddle_premium']
    atm = atm_info['atm_strike']
    dte = market_info['days_to_expiry']
    
    expected_move = straddle / atm * 100
    
    if gex > 50:
        strategies.append({
            'name': '🦅 Iron Condor',
            'type': 'NEUTRAL',
            'desc': f'Strong +GEX indicates low volatility. Sell {atm-200}/{atm-100} Put spread and {atm+100}/{atm+200} Call spread.',
            'risk': 'Limited risk, limited reward'
        })
        strategies.append({
            'name': '🔒 Short Straddle',
            'type': 'NEUTRAL',
            'desc': f'Collect ₹{straddle:.2f} premium at {atm} strike. Max profit if expires at ATM.',
            'risk': 'Unlimited risk on large moves'
        })
        if dex > 20:
            strategies.append({
                'name': '📈 Bull Call Spread',
                'type': 'BULLISH',
                'desc': f'+GEX with bullish DEX. Buy {atm} Call, Sell {atm+100} Call.',
                'risk': 'Limited risk, limited reward'
            })
    elif gex < -50:
        strategies.append({
            'name': '🎭 Long Straddle',
            'type': 'VOLATILITY',
            'desc': f'Negative GEX = high vol expected. Cost: ₹{straddle:.2f}, BE: ±{expected_move:.1f}%',
            'risk': 'Max loss = premium paid'
        })
        if dex < -30:
            strategies.append({
                'name': '📉 Long Put',
                'type': 'BEARISH',
                'desc': f'Negative GEX + Bearish DEX. Buy {atm} or {atm-50} Put.',
                'risk': 'Max loss = premium paid'
            })
            strategies.append({
                'name': '🐻 Bear Put Spread',
                'type': 'BEARISH',
                'desc': f'Buy {atm} Put, Sell {atm-100} Put for defined risk.',
                'risk': 'Limited risk, limited reward'
            })
        elif dex > 30:
            strategies.append({
                'name': '🚀 Long Call',
                'type': 'BULLISH',
                'desc': f'Negative GEX + Bullish DEX = possible breakout. Buy {atm} or {atm+50} Call.',
                'risk': 'Max loss = premium paid'
            })
    else:
        if pcr > 1.2:
            strategies.append({
                'name': '📈 Contrarian Bullish',
                'type': 'BULLISH',
                'desc': f'High PCR ({pcr:.2f}) often signals bottom. Consider Bull Call Spread.',
                'risk': 'Wait for confirmation'
            })
        elif pcr < 0.8:
            strategies.append({
                'name': '📉 Contrarian Bearish',
                'type': 'BEARISH',
                'desc': f'Low PCR ({pcr:.2f}) may signal top. Consider Bear Put Spread.',
                'risk': 'Wait for confirmation'
            })
        else:
            strategies.append({
                'name': '⏸️ Wait & Watch',
                'type': 'NEUTRAL',
                'desc': 'No clear edge. Wait for GEX/DEX signals to strengthen.',
                'risk': 'N/A'
            })
    
    return strategies


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">📊 GEX + DEX Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Gamma & Delta Exposure Analysis with Time Machine | By NYZTrade</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    symbol = st.sidebar.selectbox("Select Index", ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"], index=0)
    strikes_range = st.sidebar.slider("Strikes Range", 5, 25, 12, help="Strikes on each side of ATM")
    expiry_index = st.sidebar.number_input("Expiry Index", 0, 10, 0, help="0=Current, 1=Next")
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("📸 Capture Now", type="primary", use_container_width=True):
        st.session_state.force_capture = True
    
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (60s)", value=False)
    
    if st.session_state.snapshot_times:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📊 Session Stats**")
        st.sidebar.caption(f"Snapshots: {len(st.session_state.snapshot_times)}")
        st.sidebar.caption(f"First: {st.session_state.snapshot_times[0].strftime('%I:%M %p')}")
        st.sidebar.caption(f"Last: {st.session_state.snapshot_times[-1].strftime('%I:%M %p')}")
    
    # Time Machine
    historical_data = render_time_machine()
    
    # History Chart
    if len(st.session_state.snapshot_times) >= 2:
        hist_chart = create_history_chart()
        if hist_chart:
            st.plotly_chart(hist_chart, use_container_width=True)
    
    # Load Data
    if historical_data and not st.session_state.is_live_mode:
        df = historical_data['df']
        futures_ltp = historical_data['futures_ltp']
        market_info = historical_data['market_info']
        atm_info = historical_data['atm_info']
        flow_metrics = historical_data['flow_metrics']
        is_historical = True
        hist_time = st.session_state.snapshot_times[st.session_state.selected_time_index]
    else:
        is_historical = False
        hist_time = None
        
        calculator = GEXDEXCalculator()
        success, msg = calculator.initialize()
        
        with st.spinner(f"Loading {symbol} data..."):
            df, futures_ltp, market_info, atm_info, error = calculator.fetch_and_calculate(symbol, strikes_range, expiry_index)
        
        if df is None:
            st.error(f"Failed to load data: {error}")
            st.stop()
        
        flow_metrics = calculate_flow_metrics(df, futures_ltp)
        
        if st.session_state.auto_capture or st.session_state.force_capture:
            if capture_snapshot(df, futures_ltp, market_info, atm_info, flow_metrics):
                st.toast("📸 Snapshot captured!", icon="✅")
            st.session_state.force_capture = False
    
    # Mode Banner
    st.markdown("---")
    if is_historical:
        st.warning(f"📜 **HISTORICAL MODE** - Viewing data from {hist_time.strftime('%I:%M:%S %p')}")
    elif market_info.get('fetch_method') == 'Demo Data':
        st.warning("""
        ⚠️ **DEMO MODE** - NSE blocks cloud server requests.  
        **For live data:** Run locally with `streamlit run app.py`
        """)
    else:
        st.success(f"🔴 **LIVE MODE** - Real-time NSE data via {market_info.get('fetch_method', 'NSE')}")
    
    # Market Overview
    st.subheader("💰 Market Overview")
    
    cols = st.columns(6)
    cols[0].metric("Spot", f"₹{market_info['spot_price']:,.2f}")
    cols[1].metric("Futures", f"₹{futures_ltp:,.2f}", f"{market_info['basis']:+.2f}")
    cols[2].metric("ATM", f"{atm_info['atm_strike']:,.0f}")
    cols[3].metric("Straddle", f"₹{atm_info['atm_straddle_premium']:.2f}")
    cols[4].metric("DTE", f"{market_info['days_to_expiry']}d")
    cols[5].metric("PCR", f"{flow_metrics['pcr']:.2f}")
    
    st.info(f"📅 **Expiry:** {market_info['selected_expiry']} | ⏰ **Time:** {market_info['timestamp']} | 🔧 **Method:** {market_info['fetch_method']}")
    
    # Flow Analysis
    st.markdown("---")
    st.subheader("📊 GEX + DEX Flow Analysis")
    
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("### 🎯 GEX (Gamma)")
        st.metric("Near-term", f"{flow_metrics['gex_near_total']:.4f} B")
        st.markdown(f"**{flow_metrics['gex_near_bias']}**")
        st.caption(f"+{flow_metrics['gex_near_positive']:.4f} / {flow_metrics['gex_near_negative']:.4f}")
        st.caption(f"Total: {flow_metrics['gex_total_all']:.4f} B")
    
    with cols[1]:
        st.markdown("### 📈 DEX (Delta)")
        st.metric("Near-term", f"{flow_metrics['dex_near_total']:.4f} B")
        st.markdown(f"**{flow_metrics['dex_near_bias']}**")
        st.caption(f"+{flow_metrics['dex_near_positive']:.4f} / {flow_metrics['dex_near_negative']:.4f}")
        st.caption(f"Total: {flow_metrics['dex_total_all']:.4f} B")
    
    with cols[2]:
        st.markdown("### ⚡ Combined")
        st.metric("Signal", f"{flow_metrics['combined_signal']:.4f} B")
        st.markdown(f"**{flow_metrics['combined_bias']}**")
        st.caption(f"Max Call OI: {flow_metrics['max_call_oi_strike']:,.0f}")
        st.caption(f"Max Put OI: {flow_metrics['max_put_oi_strike']:,.0f}")
    
    # Main Charts
    st.markdown("---")
    main_fig = create_main_charts(df, futures_ltp, symbol, flow_metrics, atm_info, is_historical, hist_time)
    st.plotly_chart(main_fig, use_container_width=True)
    
    # OI Chart
    st.markdown("---")
    st.subheader("📊 Open Interest Analysis")
    oi_fig = create_oi_chart(df, futures_ltp, symbol)
    st.plotly_chart(oi_fig, use_container_width=True)
    
    # Straddle Chart
    st.markdown("---")
    st.subheader("💎 ATM Straddle Analysis")
    straddle_fig = create_straddle_chart(atm_info, futures_ltp)
    st.plotly_chart(straddle_fig, use_container_width=True)
    
    cols = st.columns(4)
    cols[0].metric("Call Premium", f"₹{atm_info['atm_call_premium']:.2f}")
    cols[1].metric("Put Premium", f"₹{atm_info['atm_put_premium']:.2f}")
    cols[2].metric("Straddle Cost", f"₹{atm_info['atm_straddle_premium']:.2f}")
    expected_move = atm_info['atm_straddle_premium'] / atm_info['atm_strike'] * 100
    cols[3].metric("Expected Move", f"±{expected_move:.2f}%")
    
    # Strategies
    st.markdown("---")
    st.subheader("💼 Trading Strategies")
    
    strategies = generate_strategies(flow_metrics, atm_info, market_info)
    
    for strat in strategies:
        with st.expander(f"**{strat['name']}** ({strat['type']})"):
            st.markdown(f"**Description:** {strat['desc']}")
            st.markdown(f"**Risk:** {strat['risk']}")
    
    # Guide
    with st.expander("📚 GEX Interpretation Guide"):
        st.markdown("""
        | GEX Value | Market Behavior | Strategy |
        |-----------|-----------------|----------|
        | **Strong +GEX (>50)** | Low volatility, range-bound | Iron Condor, Short Straddle |
        | **Mild +GEX (0-50)** | Stable with mild support | Credit Spreads |
        | **Mild -GEX (-50-0)** | Increasing volatility | Directional trades |
        | **Strong -GEX (<-50)** | High volatility | Long Straddle, Directional |
        
        **Key Levels:**
        - **Max Call OI:** Potential resistance
        - **Max Put OI:** Potential support
        - **GEX Flip:** Where dealers switch from buying to selling
        """)
    
    # Raw Data
    with st.expander("📁 Raw Data"):
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False),
            f"{symbol}_GEX_DEX_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<center>📊 <b>GEX + DEX Dashboard</b> | <b>NYZTrade</b> | "
        "<a href='https://youtube.com/@NYZTrade'>YouTube</a></center>",
        unsafe_allow_html=True
    )
    
    if auto_refresh and st.session_state.is_live_mode:
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
