# ============================================================================
# ADVANCED GEX + DEX ANALYSIS - STREAMLIT DASHBOARD
# WITH TIME MACHINE FOR BACKTESTING
# Created by NYZTrade - Options Analytics
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

# Custom CSS for styling
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
    .stAlert {
        border-radius: 10px;
    }
    div[data-testid="stExpander"] {
        border-radius: 10px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    defaults = {
        'data_snapshots': {},
        'snapshot_times': [],
        'selected_time_index': None,
        'is_live_mode': True,
        'last_capture_time': None,
        'auto_capture': True,
        'capture_interval': 3,
        'force_capture': False
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
        """Calculate d1 parameter"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
        except:
            return 0

    @staticmethod
    def calculate_gamma(S, K, T, r, sigma):
        """Calculate option gamma"""
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


# ============================================================================
# GEX DEX CALCULATOR CLASS
# ============================================================================

class GEXDEXCalculator:
    """Calculate Gamma Exposure (GEX) and Delta Exposure (DEX)"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.nseindia.com/',
            'Connection': 'keep-alive',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.base_url = "https://www.nseindia.com"
        self.option_chain_url = "https://www.nseindia.com/api/option-chain-indices"
        self.risk_free_rate = 0.07
        self.bs_calc = BlackScholesCalculator()
        self.use_demo_data = False

    def initialize_session(self):
        """Initialize session with NSE website"""
        try:
            response = self.session.get(self.base_url, timeout=10)
            if response.status_code == 200:
                time.sleep(0.5)
                return True, "Connected to NSE"
        except Exception as e:
            pass
        self.use_demo_data = True
        return True, "Using demo data (NSE not accessible)"

    def get_contract_specs(self, symbol):
        """Get contract specifications for different indices"""
        specs = {
            'NIFTY': {'size': 25, 'interval': 50},
            'BANKNIFTY': {'size': 15, 'interval': 100},
            'FINNIFTY': {'size': 40, 'interval': 50},
            'MIDCPNIFTY': {'size': 75, 'interval': 25}
        }
        return specs.get(symbol, specs['NIFTY'])

    def get_demo_spot_price(self, symbol):
        """Get demo spot prices for different indices"""
        prices = {
            'NIFTY': 24250.50,
            'BANKNIFTY': 51850.75,
            'FINNIFTY': 23150.25,
            'MIDCPNIFTY': 12450.50
        }
        return prices.get(symbol, 24250.50)

    def generate_demo_data(self, symbol="NIFTY", strikes_range=10):
        """Generate realistic demo data when NSE is not accessible"""
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        
        spot_price = self.get_demo_spot_price(symbol)
        specs = self.get_contract_specs(symbol)
        contract_size = specs['size']
        strike_interval = specs['interval']
        
        futures_ltp = spot_price * 1.0008
        atm_strike = round(spot_price / strike_interval) * strike_interval
        time_to_expiry = 7 / 365
        
        all_strikes = []
        
        for i in range(-strikes_range, strikes_range + 1):
            strike = atm_strike + (i * strike_interval)
            distance = abs(i)
            
            # Generate realistic OI distribution
            base_oi = 500000
            if i < 0:  # ITM calls / OTM puts
                call_oi = int(base_oi * (0.4 + 0.3 * np.random.random()) * max(0.2, 1 - distance * 0.08))
                put_oi = int(base_oi * (1.2 + 0.4 * np.random.random()) * max(0.3, 1 - distance * 0.05))
            else:  # OTM calls / ITM puts
                call_oi = int(base_oi * (1.2 + 0.4 * np.random.random()) * max(0.3, 1 - distance * 0.05))
                put_oi = int(base_oi * (0.4 + 0.3 * np.random.random()) * max(0.2, 1 - distance * 0.08))
            
            # OI changes
            call_oi_change = int((np.random.random() - 0.5) * call_oi * 0.12)
            put_oi_change = int((np.random.random() - 0.5) * put_oi * 0.12)
            
            # Volume
            call_volume = int(call_oi * (0.08 + 0.15 * np.random.random()))
            put_volume = int(put_oi * (0.08 + 0.15 * np.random.random()))
            
            # IV with smile
            base_iv = 14 + distance * 0.4 + np.random.random() * 2
            call_iv = base_iv + (1 if i > 0 else -0.5)
            put_iv = base_iv + (1 if i < 0 else -0.5)
            
            # LTP calculation
            if strike < spot_price:
                call_ltp = max(5, spot_price - strike + np.random.random() * 30)
                put_ltp = max(2, np.random.random() * 40 * max(0.1, 1 - distance * 0.15))
            else:
                call_ltp = max(2, np.random.random() * 40 * max(0.1, 1 - distance * 0.15))
                put_ltp = max(5, strike - spot_price + np.random.random() * 30)
            
            # Greeks calculation
            call_iv_dec = call_iv / 100
            put_iv_dec = put_iv / 100
            
            call_gamma = self.bs_calc.calculate_gamma(futures_ltp, strike, time_to_expiry, self.risk_free_rate, call_iv_dec)
            put_gamma = self.bs_calc.calculate_gamma(futures_ltp, strike, time_to_expiry, self.risk_free_rate, put_iv_dec)
            call_delta = self.bs_calc.calculate_call_delta(futures_ltp, strike, time_to_expiry, self.risk_free_rate, call_iv_dec)
            put_delta = self.bs_calc.calculate_put_delta(futures_ltp, strike, time_to_expiry, self.risk_free_rate, put_iv_dec)
            
            # GEX and DEX calculations
            multiplier = futures_ltp * futures_ltp * contract_size / 1_000_000_000
            dex_multiplier = futures_ltp * contract_size / 1_000_000_000
            
            call_gex = call_oi * call_gamma * multiplier
            put_gex = -put_oi * put_gamma * multiplier
            call_dex = call_oi * call_delta * dex_multiplier
            put_dex = put_oi * put_delta * dex_multiplier
            
            call_flow_gex = call_oi_change * call_gamma * multiplier
            put_flow_gex = -put_oi_change * put_gamma * multiplier
            call_flow_dex = call_oi_change * call_delta * dex_multiplier
            put_flow_dex = put_oi_change * put_delta * dex_multiplier
            
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
                'Call_LTP': round(call_ltp, 2),
                'Put_LTP': round(put_ltp, 2),
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
        
        df = pd.DataFrame(all_strikes).sort_values('Strike').reset_index(drop=True)
        
        # Add _B columns for compatibility
        for col in ['Call_GEX', 'Put_GEX', 'Net_GEX', 'Call_DEX', 'Put_DEX', 'Net_DEX',
                    'Call_Flow_GEX', 'Put_Flow_GEX', 'Net_Flow_GEX', 
                    'Call_Flow_DEX', 'Put_Flow_DEX', 'Net_Flow_DEX']:
            df[f'{col}_B'] = df[col]
        
        df['Total_Volume'] = df['Call_Volume'] + df['Put_Volume']
        
        max_gex = df['Net_GEX_B'].abs().max()
        df['Hedging_Pressure'] = (df['Net_GEX_B'] / max_gex * 100) if max_gex > 0 else 0
        
        # ATM info
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
        
        # Calculate next expiry
        today = datetime.now()
        days_to_thursday = (3 - today.weekday()) % 7
        if days_to_thursday == 0:
            days_to_thursday = 7
        next_expiry = today + timedelta(days=days_to_thursday)
        
        market_info = {
            'spot_price': spot_price,
            'futures_ltp': futures_ltp,
            'basis': futures_ltp - spot_price,
            'basis_pct': (futures_ltp - spot_price) / spot_price * 100,
            'fetch_method': 'Demo Data',
            'timestamp': datetime.now().strftime('%d-%b-%Y %H:%M:%S'),
            'expiry_dates': [next_expiry.strftime("%d-%b-%Y")],
            'selected_expiry': next_expiry.strftime("%d-%b-%Y"),
            'days_to_expiry': days_to_thursday
        }
        
        return df, futures_ltp, market_info, atm_info, None

    def calculate_time_to_expiry(self, expiry_str):
        """Calculate time to expiry in years"""
        try:
            expiry = datetime.strptime(expiry_str, "%d-%b-%Y")
            days = (expiry - datetime.now()).days
            return max(days / 365, 0.001), max(days, 1)
        except:
            return 7/365, 7

    def fetch_and_calculate(self, symbol="NIFTY", strikes_range=10, expiry_index=0):
        """Main function to fetch data and calculate GEX/DEX"""
        
        if self.use_demo_data:
            return self.generate_demo_data(symbol, strikes_range)
        
        try:
            url = f"{self.option_chain_url}?symbol={symbol}"
            response = self.session.get(url, timeout=15)
            
            if response.status_code != 200:
                return self.generate_demo_data(symbol, strikes_range)
            
            try:
                data = response.json()
            except:
                return self.generate_demo_data(symbol, strikes_range)
            
            if 'records' not in data:
                return self.generate_demo_data(symbol, strikes_range)
            
            records = data['records']
            spot_price = records.get('underlyingValue', 0)
            timestamp = records.get('timestamp', datetime.now().strftime('%d-%b-%Y %H:%M:%S'))
            expiry_dates = records.get('expiryDates', [])
            
            if not expiry_dates or spot_price == 0:
                return self.generate_demo_data(symbol, strikes_range)
            
            selected_expiry = expiry_dates[min(expiry_index, len(expiry_dates) - 1)]
            time_to_expiry, days_to_expiry = self.calculate_time_to_expiry(selected_expiry)
            
            specs = self.get_contract_specs(symbol)
            contract_size = specs['size']
            strike_interval = specs['interval']
            
            futures_ltp = spot_price * np.exp(self.risk_free_rate * time_to_expiry)
            
            all_strikes = []
            processed = set()
            atm_strike = None
            min_diff = float('inf')
            atm_call = 0
            atm_put = 0
            
            for item in records.get('data', []):
                if item.get('expiryDate') != selected_expiry:
                    continue
                
                strike = item.get('strikePrice', 0)
                if strike == 0 or strike in processed:
                    continue
                
                processed.add(strike)
                
                if abs(strike - futures_ltp) / strike_interval > strikes_range:
                    continue
                
                ce = item.get('CE', {})
                pe = item.get('PE', {})
                
                call_oi = ce.get('openInterest', 0)
                put_oi = pe.get('openInterest', 0)
                call_oi_change = ce.get('changeinOpenInterest', 0)
                put_oi_change = pe.get('changeinOpenInterest', 0)
                call_volume = ce.get('totalTradedVolume', 0)
                put_volume = pe.get('totalTradedVolume', 0)
                call_iv = ce.get('impliedVolatility', 0) or 15
                put_iv = pe.get('impliedVolatility', 0) or 15
                call_ltp = ce.get('lastPrice', 0)
                put_ltp = pe.get('lastPrice', 0)
                
                # Track ATM
                diff = abs(strike - futures_ltp)
                if diff < min_diff:
                    min_diff = diff
                    atm_strike = strike
                    atm_call = call_ltp
                    atm_put = put_ltp
                
                # Calculate Greeks
                call_iv_dec = call_iv / 100 if call_iv > 0 else 0.15
                put_iv_dec = put_iv / 100 if put_iv > 0 else 0.15
                
                call_gamma = self.bs_calc.calculate_gamma(futures_ltp, strike, time_to_expiry, self.risk_free_rate, call_iv_dec)
                put_gamma = self.bs_calc.calculate_gamma(futures_ltp, strike, time_to_expiry, self.risk_free_rate, put_iv_dec)
                call_delta = self.bs_calc.calculate_call_delta(futures_ltp, strike, time_to_expiry, self.risk_free_rate, call_iv_dec)
                put_delta = self.bs_calc.calculate_put_delta(futures_ltp, strike, time_to_expiry, self.risk_free_rate, put_iv_dec)
                
                # GEX/DEX calculations
                mult = futures_ltp * futures_ltp * contract_size / 1_000_000_000
                dex_mult = futures_ltp * contract_size / 1_000_000_000
                
                call_gex = call_oi * call_gamma * mult
                put_gex = -put_oi * put_gamma * mult
                call_dex = call_oi * call_delta * dex_mult
                put_dex = put_oi * put_delta * dex_mult
                
                call_flow_gex = call_oi_change * call_gamma * mult
                put_flow_gex = -put_oi_change * put_gamma * mult
                call_flow_dex = call_oi_change * call_delta * dex_mult
                put_flow_dex = put_oi_change * put_delta * dex_mult
                
                all_strikes.append({
                    'Strike': strike,
                    'Call_OI': call_oi, 'Put_OI': put_oi,
                    'Call_OI_Change': call_oi_change, 'Put_OI_Change': put_oi_change,
                    'Call_Volume': call_volume, 'Put_Volume': put_volume,
                    'Call_IV': call_iv, 'Put_IV': put_iv,
                    'Call_LTP': call_ltp, 'Put_LTP': put_ltp,
                    'Call_Gamma': call_gamma, 'Put_Gamma': put_gamma,
                    'Call_Delta': call_delta, 'Put_Delta': put_delta,
                    'Call_GEX': call_gex, 'Put_GEX': put_gex, 'Net_GEX': call_gex + put_gex,
                    'Call_DEX': call_dex, 'Put_DEX': put_dex, 'Net_DEX': call_dex + put_dex,
                    'Call_Flow_GEX': call_flow_gex, 'Put_Flow_GEX': put_flow_gex, 'Net_Flow_GEX': call_flow_gex + put_flow_gex,
                    'Call_Flow_DEX': call_flow_dex, 'Put_Flow_DEX': put_flow_dex, 'Net_Flow_DEX': call_flow_dex + put_flow_dex
                })
            
            if not all_strikes:
                return self.generate_demo_data(symbol, strikes_range)
            
            df = pd.DataFrame(all_strikes).sort_values('Strike').reset_index(drop=True)
            
            for col in ['Call_GEX', 'Put_GEX', 'Net_GEX', 'Call_DEX', 'Put_DEX', 'Net_DEX',
                        'Call_Flow_GEX', 'Put_Flow_GEX', 'Net_Flow_GEX',
                        'Call_Flow_DEX', 'Put_Flow_DEX', 'Net_Flow_DEX']:
                df[f'{col}_B'] = df[col]
            
            df['Total_Volume'] = df['Call_Volume'] + df['Put_Volume']
            max_gex = df['Net_GEX_B'].abs().max()
            df['Hedging_Pressure'] = (df['Net_GEX_B'] / max_gex * 100) if max_gex > 0 else 0
            
            atm_info = {
                'atm_strike': atm_strike or df.iloc[len(df)//2]['Strike'],
                'atm_call_premium': atm_call,
                'atm_put_premium': atm_put,
                'atm_straddle_premium': atm_call + atm_put
            }
            
            market_info = {
                'spot_price': spot_price,
                'futures_ltp': futures_ltp,
                'basis': futures_ltp - spot_price,
                'basis_pct': (futures_ltp - spot_price) / spot_price * 100 if spot_price > 0 else 0,
                'fetch_method': 'NSE Live',
                'timestamp': timestamp,
                'expiry_dates': expiry_dates,
                'selected_expiry': selected_expiry,
                'days_to_expiry': days_to_expiry
            }
            
            return df, futures_ltp, market_info, atm_info, None
            
        except Exception as e:
            return self.generate_demo_data(symbol, strikes_range)


# ============================================================================
# FLOW ANALYSIS
# ============================================================================

def calculate_flow_metrics(df, futures_ltp):
    """Calculate GEX and DEX flow metrics"""
    df_unique = df.drop_duplicates(subset=['Strike']).sort_values('Strike').reset_index(drop=True)
    
    # GEX flow - 5 positive + 5 negative closest to spot
    pos_gex = df_unique[df_unique['Net_GEX_B'] > 0].copy()
    pos_gex['Dist'] = abs(pos_gex['Strike'] - futures_ltp)
    pos_gex = pos_gex.nsmallest(5, 'Dist')
    
    neg_gex = df_unique[df_unique['Net_GEX_B'] < 0].copy()
    neg_gex['Dist'] = abs(neg_gex['Strike'] - futures_ltp)
    neg_gex = neg_gex.nsmallest(5, 'Dist')
    
    gex_near_pos = float(pos_gex['Net_GEX_B'].sum()) if len(pos_gex) > 0 else 0
    gex_near_neg = float(neg_gex['Net_GEX_B'].sum()) if len(neg_gex) > 0 else 0
    gex_near_total = gex_near_pos + gex_near_neg
    
    gex_total = float(df_unique['Net_GEX_B'].sum())
    
    # DEX flow - above and below futures
    above = df_unique[df_unique['Strike'] > futures_ltp].head(5)
    below = df_unique[df_unique['Strike'] < futures_ltp].tail(5)
    
    dex_near_pos = float(above['Net_DEX_B'].sum()) if len(above) > 0 else 0
    dex_near_neg = float(below['Net_DEX_B'].sum()) if len(below) > 0 else 0
    dex_near_total = dex_near_pos + dex_near_neg
    
    dex_total = float(df_unique['Net_DEX_B'].sum())
    
    # Bias determination
    def get_bias(val, threshold=50):
        if val > threshold:
            return "🟢 STRONG BULLISH", "green"
        elif val > 0:
            return "🟢 BULLISH", "lightgreen"
        elif val < -threshold:
            return "🔴 STRONG BEARISH", "red"
        elif val < 0:
            return "🔴 BEARISH", "salmon"
        return "⚪ NEUTRAL", "gray"
    
    gex_bias, gex_color = get_bias(gex_near_total)
    dex_bias, dex_color = get_bias(dex_near_total)
    combined = (gex_near_total + dex_near_total) / 2
    combined_bias, combined_color = get_bias(combined)
    
    return {
        'gex_near_positive': gex_near_pos,
        'gex_near_negative': gex_near_neg,
        'gex_near_total': gex_near_total,
        'gex_total_all': gex_total,
        'gex_near_bias': gex_bias,
        'gex_near_color': gex_color,
        'dex_near_positive': dex_near_pos,
        'dex_near_negative': dex_near_neg,
        'dex_near_total': dex_near_total,
        'dex_total_all': dex_total,
        'dex_near_bias': dex_bias,
        'dex_near_color': dex_color,
        'combined_signal': combined,
        'combined_bias': combined_bias,
        'combined_color': combined_color
    }


# ============================================================================
# SNAPSHOT FUNCTIONS
# ============================================================================

def capture_snapshot(df, futures_ltp, market_info, atm_info, flow_metrics):
    """Capture current data as a snapshot for time machine"""
    now = datetime.now().replace(microsecond=0)
    
    # Check interval
    if st.session_state.last_capture_time:
        elapsed = (now - st.session_state.last_capture_time).total_seconds() / 60
        if elapsed < st.session_state.capture_interval:
            return False
    
    # Store snapshot
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
    
    # Limit to 500 snapshots
    while len(st.session_state.snapshot_times) > 500:
        oldest = st.session_state.snapshot_times.pop(0)
        st.session_state.data_snapshots.pop(oldest, None)
    
    return True


# ============================================================================
# TIME MACHINE UI
# ============================================================================

def render_time_machine():
    """Render the time machine / time slider UI"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown("### ⏰ Time Machine")
    
    with col2:
        if st.session_state.is_live_mode:
            st.success("🟢 LIVE")
        else:
            st.warning("📜 HISTORY")
    
    with col3:
        if not st.session_state.is_live_mode:
            if st.button("🔴 Go Live", use_container_width=True):
                st.session_state.is_live_mode = True
                st.session_state.selected_time_index = None
                st.rerun()
    
    # No snapshots yet
    if not st.session_state.snapshot_times:
        st.info("📝 No historical data yet. Data will be captured automatically.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.auto_capture = st.checkbox(
                "🔄 Auto-capture enabled",
                value=st.session_state.auto_capture
            )
        with col2:
            st.session_state.capture_interval = st.selectbox(
                "Capture interval",
                options=[1, 2, 3, 5, 10],
                index=[1, 2, 3, 5, 10].index(st.session_state.capture_interval) if st.session_state.capture_interval in [1, 2, 3, 5, 10] else 2,
                format_func=lambda x: f"{x} min"
            )
        return None
    
    # Show time range
    first = st.session_state.snapshot_times[0]
    last = st.session_state.snapshot_times[-1]
    st.caption(f"📊 {len(st.session_state.snapshot_times)} snapshots | {first.strftime('%I:%M %p')} → {last.strftime('%I:%M %p')}")
    
    # Time slider
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
        
        # Quick jump buttons
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
    
    # Settings expander
    with st.expander("⚙️ Capture Settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.auto_capture = st.checkbox(
                "Auto-capture",
                value=st.session_state.auto_capture
            )
        with col2:
            st.session_state.capture_interval = st.selectbox(
                "Interval",
                options=[1, 2, 3, 5, 10],
                index=[1, 2, 3, 5, 10].index(st.session_state.capture_interval) if st.session_state.capture_interval in [1, 2, 3, 5, 10] else 2,
                format_func=lambda x: f"{x} min",
                key="interval_select"
            )
        with col3:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.data_snapshots = {}
                st.session_state.snapshot_times = []
                st.session_state.selected_time_index = None
                st.session_state.is_live_mode = True
                st.rerun()
    
    # Return historical data if in historical mode
    if not st.session_state.is_live_mode and st.session_state.selected_time_index is not None:
        sel_time = st.session_state.snapshot_times[st.session_state.selected_time_index]
        return st.session_state.data_snapshots.get(sel_time)
    
    return None


# ============================================================================
# CHARTS
# ============================================================================

def create_main_charts(df, futures_ltp, symbol, flow_metrics, atm_info, is_historical=False, hist_time=None):
    """Create the main dashboard charts"""
    
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
        go.Bar(y=strikes, x=df['Net_GEX_B'], orientation='h', marker_color=colors1, name='Net GEX'),
        row=1, col=1
    )
    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="#6c5ce7", line_width=2, row=1, col=1)
    
    # Chart 2: Net DEX
    colors2 = ['#00d4aa' if x > 0 else '#ff6b6b' for x in df['Net_DEX_B']]
    fig.add_trace(
        go.Bar(y=strikes, x=df['Net_DEX_B'], orientation='h', marker_color=colors2, name='Net DEX'),
        row=1, col=2
    )
    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="#6c5ce7", line_width=2, row=1, col=2)
    
    # Chart 3: GEX Flow
    colors3 = ['#00d4aa' if x > 0 else '#ff6b6b' for x in df['Net_Flow_GEX_B']]
    fig.add_trace(
        go.Bar(y=strikes, x=df['Net_Flow_GEX_B'], orientation='h', marker_color=colors3, name='GEX Flow'),
        row=2, col=1
    )
    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="#6c5ce7", line_width=2, row=2, col=1)
    
    # Chart 4: DEX Flow
    colors4 = ['#00d4aa' if x > 0 else '#ff6b6b' for x in df['Net_Flow_DEX_B']]
    fig.add_trace(
        go.Bar(y=strikes, x=df['Net_Flow_DEX_B'], orientation='h', marker_color=colors4, name='DEX Flow'),
        row=2, col=2
    )
    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="#6c5ce7", line_width=2, row=2, col=2)
    
    # Chart 5: Hedging Pressure
    fig.add_trace(
        go.Bar(
            y=strikes, x=df['Hedging_Pressure'], orientation='h',
            marker=dict(color=df['Hedging_Pressure'], colorscale='RdYlGn', cmin=-100, cmax=100),
            name='Pressure'
        ),
        row=3, col=1
    )
    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="#6c5ce7", line_width=2, row=3, col=1)
    
    # Chart 6: Combined Signal
    df['Combined'] = (df['Net_GEX_B'] + df['Net_DEX_B']) / 2
    colors6 = ['#00d4aa' if x > 0 else '#ff6b6b' for x in df['Combined']]
    fig.add_trace(
        go.Bar(y=strikes, x=df['Combined'], orientation='h', marker_color=colors6, name='Combined'),
        row=3, col=2
    )
    fig.add_hline(y=futures_ltp, line_dash="dash", line_color="#6c5ce7", line_width=2, row=3, col=2)
    
    # Layout
    mode_str = f"📜 HISTORICAL - {hist_time.strftime('%I:%M %p')}" if is_historical else "🔴 LIVE"
    
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol} GEX + DEX Analysis</b> ({mode_str})<br><sup>Futures: ₹{futures_ltp:,.2f} | {flow_metrics['combined_bias']}</sup>",
            x=0.5,
            font=dict(size=20)
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
    
    fig.add_trace(go.Scatter(
        x=strikes, y=straddle_payoff,
        mode='lines', name='Straddle',
        line=dict(color='#6c5ce7', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=strikes, y=call_payoff,
        mode='lines', name='Call',
        line=dict(color='#00d4aa', width=2, dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=strikes, y=put_payoff,
        mode='lines', name='Put',
        line=dict(color='#ff6b6b', width=2, dash='dot')
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=atm, line_color="#6c5ce7", annotation_text=f"ATM: {atm:,.0f}")
    fig.add_vline(x=futures_ltp, line_color="yellow", line_dash="dash", annotation_text="Futures")
    
    # Breakeven lines
    upper_be = atm + straddle_prem
    lower_be = atm - straddle_prem
    fig.add_vline(x=upper_be, line_color="orange", line_dash="dot", annotation_text=f"BE: {upper_be:,.0f}")
    fig.add_vline(x=lower_be, line_color="orange", line_dash="dot", annotation_text=f"BE: {lower_be:,.0f}")
    
    fig.update_layout(
        title=f"<b>ATM Straddle Payoff</b><br><sup>Strike: {atm:,.0f} | Premium: ₹{straddle_prem:.2f} | Breakevens: {lower_be:,.0f} - {upper_be:,.0f}</sup>",
        xaxis_title="Spot Price at Expiry",
        yaxis_title="Profit/Loss (₹)",
        height=400,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    return fig


def create_history_chart():
    """Create intraday price history chart"""
    if len(st.session_state.snapshot_times) < 2:
        return None
    
    times = []
    prices = []
    gex_vals = []
    
    for t in st.session_state.snapshot_times:
        if t in st.session_state.data_snapshots:
            times.append(t)
            prices.append(st.session_state.data_snapshots[t]['futures_ltp'])
            gex_vals.append(st.session_state.data_snapshots[t]['flow_metrics']['gex_near_total'])
    
    if len(times) < 2:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
        subplot_titles=('Futures Price', 'GEX Flow')
    )
    
    fig.add_trace(
        go.Scatter(x=times, y=prices, mode='lines+markers',
                   line=dict(color='#6c5ce7', width=2),
                   marker=dict(size=4)),
        row=1, col=1
    )
    
    colors = ['#00d4aa' if x > 0 else '#ff6b6b' for x in gex_vals]
    fig.add_trace(
        go.Bar(x=times, y=gex_vals, marker_color=colors),
        row=2, col=1
    )
    
    # Mark selected time if in historical mode
    if not st.session_state.is_live_mode and st.session_state.selected_time_index is not None:
        sel_time = st.session_state.snapshot_times[st.session_state.selected_time_index]
        if sel_time in st.session_state.data_snapshots:
            sel_price = st.session_state.data_snapshots[sel_time]['futures_ltp']
            fig.add_vline(x=sel_time, line_dash="dash", line_color="orange", line_width=2)
            fig.add_annotation(x=sel_time, y=sel_price, text="📍", showarrow=False, row=1, col=1)
    
    fig.update_layout(
        height=280,
        showlegend=False,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=30, t=40, b=30)
    )
    
    return fig


# ============================================================================
# STRATEGIES
# ============================================================================

def generate_strategies(flow_metrics, atm_info):
    """Generate trading strategy suggestions based on GEX/DEX"""
    
    strategies = []
    gex = flow_metrics['gex_near_total']
    dex = flow_metrics['dex_near_total']
    straddle = atm_info['atm_straddle_premium']
    
    if gex > 50:
        strategies.append({
            'name': '🦅 Iron Condor',
            'type': 'NEUTRAL',
            'desc': f'Strong +GEX suggests low volatility. Sell OTM options on both sides.'
        })
        strategies.append({
            'name': '🔒 Short Straddle',
            'type': 'NEUTRAL',
            'desc': f'Collect ₹{straddle:.2f} premium. Market likely range-bound.'
        })
    elif gex < -50:
        strategies.append({
            'name': '🎭 Long Straddle',
            'type': 'VOLATILITY',
            'desc': f'Negative GEX = high volatility expected. Cost: ₹{straddle:.2f}'
        })
        if dex < -20:
            strategies.append({
                'name': '📉 Long Put',
                'type': 'BEARISH',
                'desc': 'Negative GEX + Bearish DEX = downside momentum'
            })
        elif dex > 20:
            strategies.append({
                'name': '🚀 Long Call',
                'type': 'BULLISH',
                'desc': 'Negative GEX + Bullish DEX = upside breakout possible'
            })
    else:
        if dex > 30:
            strategies.append({
                'name': '📈 Bull Call Spread',
                'type': 'BULLISH',
                'desc': 'Neutral GEX with bullish DEX = controlled upside bet'
            })
        elif dex < -30:
            strategies.append({
                'name': '📉 Bear Put Spread',
                'type': 'BEARISH',
                'desc': 'Neutral GEX with bearish DEX = controlled downside bet'
            })
        else:
            strategies.append({
                'name': '⏸️ Wait for Clarity',
                'type': 'NEUTRAL',
                'desc': 'No clear edge - wait for stronger signals'
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
    
    symbol = st.sidebar.selectbox(
        "Select Index",
        ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"],
        index=0
    )
    
    strikes_range = st.sidebar.slider(
        "Strikes Range",
        min_value=5, max_value=25, value=12,
        help="Number of strikes on each side of ATM"
    )
    
    expiry_index = st.sidebar.number_input(
        "Expiry Index",
        min_value=0, max_value=10, value=0,
        help="0=Current, 1=Next, etc."
    )
    
    st.sidebar.markdown("---")
    
    # Manual capture button
    if st.sidebar.button("📸 Capture Snapshot", type="primary", use_container_width=True):
        st.session_state.force_capture = True
    
    # Auto refresh
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (60s)", value=False)
    
    # Sidebar stats
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
        # Historical mode
        df = historical_data['df']
        futures_ltp = historical_data['futures_ltp']
        market_info = historical_data['market_info']
        atm_info = historical_data['atm_info']
        flow_metrics = historical_data['flow_metrics']
        is_historical = True
        hist_time = st.session_state.snapshot_times[st.session_state.selected_time_index]
    else:
        # Live mode
        is_historical = False
        hist_time = None
        
        calculator = GEXDEXCalculator()
        calculator.initialize_session()
        
        with st.spinner(f"Loading {symbol} data..."):
            df, futures_ltp, market_info, atm_info, error = calculator.fetch_and_calculate(
                symbol, strikes_range, expiry_index
            )
        
        if df is None or error:
            st.error(f"Failed to load data: {error}")
            st.stop()
        
        flow_metrics = calculate_flow_metrics(df, futures_ltp)
        
        # Auto capture
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
        This shows simulated data to demonstrate dashboard functionality.  
        **For live data:** Run locally with `streamlit run app.py`
        """)
    else:
        st.success("🔴 **LIVE MODE** - Real-time NSE data")
    
    # Market Overview
    st.subheader("💰 Market Overview")
    
    cols = st.columns(5)
    
    with cols[0]:
        st.metric("Spot Price", f"₹{market_info['spot_price']:,.2f}")
    
    with cols[1]:
        st.metric(
            "Futures",
            f"₹{futures_ltp:,.2f}",
            f"{market_info['basis']:+.2f} ({market_info['basis_pct']:+.2f}%)"
        )
    
    with cols[2]:
        st.metric("ATM Strike", f"{atm_info['atm_strike']:,.0f}")
    
    with cols[3]:
        st.metric("ATM Straddle", f"₹{atm_info['atm_straddle_premium']:.2f}")
    
    with cols[4]:
        st.metric("Days to Expiry", f"{market_info['days_to_expiry']} days")
    
    st.info(f"📅 Expiry: {market_info['selected_expiry']} | ⏰ Data: {market_info['timestamp']} | 🔧 {market_info['fetch_method']}")
    
    # Flow Analysis
    st.markdown("---")
    st.subheader("📊 GEX + DEX Flow Analysis")
    
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("### 🎯 GEX (Gamma)")
        st.metric("Near-term Flow", f"{flow_metrics['gex_near_total']:.4f} B")
        st.markdown(f"**Bias:** {flow_metrics['gex_near_bias']}")
        st.caption(f"Total: {flow_metrics['gex_total_all']:.4f} B")
    
    with cols[1]:
        st.markdown("### 📈 DEX (Delta)")
        st.metric("Near-term Flow", f"{flow_metrics['dex_near_total']:.4f} B")
        st.markdown(f"**Bias:** {flow_metrics['dex_near_bias']}")
        st.caption(f"Total: {flow_metrics['dex_total_all']:.4f} B")
    
    with cols[2]:
        st.markdown("### ⚡ Combined Signal")
        st.metric("Signal Strength", f"{flow_metrics['combined_signal']:.4f} B")
        st.markdown(f"**Overall:** {flow_metrics['combined_bias']}")
    
    # Main Charts
    st.markdown("---")
    
    main_fig = create_main_charts(df, futures_ltp, symbol, flow_metrics, atm_info, is_historical, hist_time)
    st.plotly_chart(main_fig, use_container_width=True)
    
    # Straddle Chart
    st.markdown("---")
    st.subheader("💎 ATM Straddle Analysis")
    
    straddle_fig = create_straddle_chart(atm_info, futures_ltp)
    st.plotly_chart(straddle_fig, use_container_width=True)
    
    cols = st.columns(4)
    with cols[0]:
        st.metric("Call Premium", f"₹{atm_info['atm_call_premium']:.2f}")
    with cols[1]:
        st.metric("Put Premium", f"₹{atm_info['atm_put_premium']:.2f}")
    with cols[2]:
        st.metric("Straddle Cost", f"₹{atm_info['atm_straddle_premium']:.2f}")
    with cols[3]:
        be_range = atm_info['atm_straddle_premium'] / atm_info['atm_strike'] * 100
        st.metric("Expected Move", f"±{be_range:.2f}%")
    
    # Strategies
    st.markdown("---")
    st.subheader("💼 Suggested Strategies")
    
    strategies = generate_strategies(flow_metrics, atm_info)
    
    for strat in strategies:
        with st.expander(f"**{strat['name']}** ({strat['type']})"):
            st.markdown(strat['desc'])
    
    # GEX Interpretation Guide
    with st.expander("📚 GEX Interpretation Guide"):
        st.markdown("""
        **Understanding GEX (Gamma Exposure):**
        
        | GEX Value | Market Behavior | MM Action | Strategy |
        |-----------|-----------------|-----------|----------|
        | **Strong Positive (>50)** | Low volatility, range-bound | Buy dips, sell rallies | Iron Condor, Short Straddle |
        | **Positive (0 to 50)** | Mild support | Stabilizing | Credit Spreads |
        | **Negative (-50 to 0)** | Increasing volatility | Amplifying moves | Directional trades |
        | **Strong Negative (<-50)** | High volatility | Selling into rallies | Long Straddle, Directional |
        
        **Key Levels:**
        - **GEX Flip Point:** Where GEX changes from positive to negative
        - **Max Pain Strike:** Highest combined OI (often attracts price)
        - **Key Support/Resistance:** Strikes with highest positive GEX
        """)
    
    # Raw Data
    with st.expander("📁 Raw Data"):
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"{symbol}_GEX_DEX_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<center>📊 <b>GEX + DEX Dashboard</b> | Created by <b>NYZTrade</b> | "
        "<a href='https://youtube.com/@NYZTrade' target='_blank'>YouTube</a></center>",
        unsafe_allow_html=True
    )
    
    # Auto refresh
    if auto_refresh and st.session_state.is_live_mode:
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
