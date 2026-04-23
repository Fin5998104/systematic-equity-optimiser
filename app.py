"""
PORTFOLIO OPTIMISER WEB APP
=============================
Flask web app that:
  1. Displays historical performance from primary.json
  2. Generates live portfolios using current Yahoo Finance prices
  3. Tracks user portfolio state across rebalances

Run from PowerShell:
    pip install flask yfinance scipy scikit-learn pandas numpy
    python app.py

Then open: http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import yfinance as yf
import json
import os
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================================================
# CONFIGURATION (locked from final sweep)
# ============================================================
TRAIN_DAYS       = 147       # 7mo
TEST_PERIOD      = 63        # quarterly
ADAPT_STRENGTH   = 3.5
RATE_STRENGTH    = 0.75
DISP_FLOOR       = 0.3
DISP_CAP         = 1.0
DISP_LOOKBACK    = 63
RATE_LOOKBACK    = 6
MAX_WEIGHT       = 1.0
MIN_STOCKS       = 20
SPREAD_THRESHOLD = 1.2

# Data paths
DASHBOARD_DIR = os.path.expanduser('~/.norgate_dashboard')
PRIMARY_PATH = os.path.join(DASHBOARD_DIR, 'primary.json')
CHARTS_PATH = os.path.join(DASHBOARD_DIR, 'charts.json')
USER_STATE_DIR = os.path.expanduser('~/.portfolio_optimiser')
os.makedirs(USER_STATE_DIR, exist_ok=True)

# ============================================================
# LOAD HISTORICAL DATA
# ============================================================
historical_data = {}
if os.path.exists(PRIMARY_PATH):
    with open(PRIMARY_PATH, 'r') as f:
        historical_data = json.load(f)
    print(f"Loaded historical data from {PRIMARY_PATH}")
else:
    print(f"Warning: {PRIMARY_PATH} not found - dashboard will be empty")

charts_data = {}
if os.path.exists(CHARTS_PATH):
    with open(CHARTS_PATH, 'r') as f:
        charts_data = json.load(f)
    print(f"Loaded chart data from {CHARTS_PATH}")
else:
    print(f"Warning: {CHARTS_PATH} not found - charts will be hidden "
          "(run build_charts.py to generate)")

# ============================================================
# UNIVERSE - S&P 500 + NASDAQ 100 current constituents
# ============================================================
_universe_cache = {'tickers': None, 'names': None, 'fetched': None}

def get_universe():
    """Get current S&P 500 + NASDAQ 100 tickers and company names."""
    import requests
    from io import StringIO

    # Cache for an hour to avoid hammering Wikipedia
    if (_universe_cache['fetched'] and
        (datetime.now() - _universe_cache['fetched']).seconds < 3600):
        return _universe_cache['tickers'], _universe_cache['names']

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36'
    }

    name_map = {}
    try:
        # S&P 500
        r = requests.get(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
            headers=headers, timeout=10)
        r.raise_for_status()
        sp_table = pd.read_html(StringIO(r.text))[0]
        sp500 = sp_table['Symbol'].tolist()
        sp_names = sp_table['Security'].tolist() if 'Security' in sp_table.columns else sp500
        for t, n in zip(sp500, sp_names):
            name_map[str(t).replace('.', '-')] = str(n)

        # NASDAQ 100
        r = requests.get(
            'https://en.wikipedia.org/wiki/Nasdaq-100',
            headers=headers, timeout=10)
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))

        nasdaq = []
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if 'ticker' in cols or 'symbol' in cols:
                ticker_col = 'Ticker' if 'ticker' in cols else 'Symbol'
                # Find a name column
                name_col = None
                for c in t.columns:
                    cl = str(c).lower()
                    if cl in ('company', 'security', 'name'):
                        name_col = c
                        break
                nasdaq = t[ticker_col].tolist()
                if name_col:
                    nas_names = t[name_col].tolist()
                    for tk, nm in zip(nasdaq, nas_names):
                        key = str(tk).replace('.', '-')
                        if key not in name_map:
                            name_map[key] = str(nm)
                break

        universe = sorted(set(sp500 + nasdaq))
        universe = [str(t).replace('.', '-') for t in universe
                    if pd.notna(t)]

        _universe_cache['tickers'] = universe
        _universe_cache['names'] = name_map
        _universe_cache['fetched'] = datetime.now()

        return universe, name_map
    except Exception as e:
        print(f"Universe fetch failed: {e}")
        return [], {}

# ============================================================
# OPTIMISATION
# ============================================================
def optimise_adaptive(train_data, k):
    clean = np.nan_to_num(train_data, nan=0.0)
    mean_r = clean.mean(axis=0)
    n = clean.shape[1]
    if n < 5:
        return np.full(n, 1.0/n)
    lw = LedoitWolf()
    lw.fit(clean)
    cov_matrix = lw.covariance_
    def fn(w):
        p = w @ mean_r
        v = np.sqrt(w @ cov_matrix @ w)
        if v <= 0:
            return 0
        if p >= 0:
            return -(p ** k) / v
        else:
            return (abs(p) ** k) / v
    result = minimize(fn, np.full(n, 1.0/n), method='SLSQP',
        bounds=tuple((0, MAX_WEIGHT) for _ in range(n)),
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        options={'maxiter': 500})
    return result.x

def optimise_sharpe(train_data):
    return optimise_adaptive(train_data, 1.0)

# ============================================================
# LIVE PORTFOLIO GENERATION
# ============================================================
def get_gbp_usd():
    """Fetch current GBP/USD rate."""
    try:
        ticker = yf.Ticker('GBPUSD=X')
        hist = ticker.history(period='5d')
        if len(hist) > 0:
            return float(hist['Close'].iloc[-1])
    except Exception as e:
        print(f"FX fetch failed: {e}")
    return 1.27  # fallback

def generate_portfolio(portfolio_value_gbp, min_position_value=None,
                       fractional_shares=False):
    """Download current prices, run optimisation, return target portfolio.

    portfolio_value_gbp is in GBP; converted to USD for optimisation.
    fractional_shares: if True, allow fractional share quantities.
    """
    # Get FX rate and convert
    fx_rate = get_gbp_usd()
    portfolio_value = portfolio_value_gbp * fx_rate  # USD value

    # Auto-calculate minimum position if not specified (in USD)
    if min_position_value is None:
        if portfolio_value < 1000:
            min_position_value = portfolio_value * 0.05
        elif portfolio_value < 5000:
            min_position_value = portfolio_value * 0.025
        elif portfolio_value < 25000:
            min_position_value = portfolio_value * 0.01
        else:
            min_position_value = max(100, portfolio_value * 0.005)
    print("Fetching universe...")
    universe, name_map = get_universe()
    if not universe:
        return {'error': 'Failed to fetch universe'}

    print(f"Universe size: {len(universe)}")
    print(f"Downloading {TRAIN_DAYS + 30} days of price data...")

    # Download with buffer for weekends/holidays
    end = datetime.now()
    start = end - timedelta(days=int(TRAIN_DAYS * 1.6))

    try:
        data = yf.download(universe, start=start, end=end,
                           progress=False, auto_adjust=True)
        if data.empty:
            return {'error': 'No price data returned'}
        prices = data['Close'] if 'Close' in data.columns.get_level_values(0) else data
    except Exception as e:
        return {'error': f'Price download failed: {str(e)}'}

    # Compute returns
    rets = prices.pct_change().dropna(how='all')

    # Filter to stocks with adequate coverage
    coverage = rets.notna().mean()
    valid_tickers = coverage[coverage >= 0.95].index.tolist()
    rets_valid = rets[valid_tickers].iloc[-TRAIN_DAYS:]

    if len(valid_tickers) < MIN_STOCKS:
        return {'error': f'Only {len(valid_tickers)} valid stocks'}

    print(f"Valid stocks: {len(valid_tickers)}")

    # Compute conditioning signals
    train_data = rets_valid.values

    # Dispersion (last 63 days)
    disp_window = train_data[-DISP_LOOKBACK:]
    disp_clean = np.nan_to_num(disp_window, nan=0.0)
    stock_rets = (1 + disp_clean).prod(axis=0) - 1
    current_disp = float(stock_rets.std())

    # Use target_disp from historical config
    target_disp = historical_data.get('config', {}).get('target_disp', 0.0996)
    disp_ratio = current_disp / target_disp if target_disp > 0 else 1.0
    disp_scale = max(DISP_FLOOR, min(DISP_CAP, disp_ratio))

    # Rate change (FRED)
    rate_chg = 0.0
    spread_elev = False
    try:
        import pandas_datareader as pdr
        ff = pdr.DataReader('DFF', 'fred',
                            (end - timedelta(days=400)).strftime('%Y-%m-%d'),
                            end.strftime('%Y-%m-%d'))['DFF']
        cs = pdr.DataReader('BAMLH0A0HYM2', 'fred',
                            (end - timedelta(days=30)).strftime('%Y-%m-%d'),
                            end.strftime('%Y-%m-%d'))['BAMLH0A0HYM2']
        ff_now = ff.dropna().iloc[-1]
        ff_past = ff.loc[:end - pd.DateOffset(months=RATE_LOOKBACK)].dropna().iloc[-1]
        rate_chg = float(ff_now - ff_past)
        current_spread = float(cs.dropna().iloc[-1])
        spread_median = historical_data.get('config', {}).get('spread_median', 4.54)
        spread_elev = current_spread > spread_median * SPREAD_THRESHOLD
    except Exception as e:
        print(f"FRED fetch failed: {e}")
        current_spread = 4.0

    # Compute k
    disp_k = ADAPT_STRENGTH * (1.0 - disp_scale)
    rate_signal = max(0, rate_chg) / 2.0
    rate_k = RATE_STRENGTH * rate_signal
    k = 1.0 + disp_k + rate_k
    k = min(k, 3.0)

    # Optimise
    print(f"Optimising with k={k:.2f}...")
    if k > 1.001:
        weights = optimise_adaptive(train_data, k)
    else:
        weights = optimise_sharpe(train_data)

    # Credit gate blend
    blend = 0.0
    if rate_chg < 0 and spread_elev:
        blend = min(1.0, abs(rate_chg) / 2.0)
        eq_weights = np.full(len(valid_tickers), 1.0 / len(valid_tickers))
        weights = (1 - blend) * weights + blend * eq_weights

    # Get current prices for share calculation
    current_prices = prices[valid_tickers].iloc[-1]

    # First pass: identify positions above the minimum
    min_weight = min_position_value / portfolio_value
    kept_indices = []
    for i, ticker in enumerate(valid_tickers):
        w = float(weights[i])
        if w < min_weight:
            continue
        price = float(current_prices[ticker])
        if np.isnan(price) or price <= 0:
            continue
        kept_indices.append(i)

    # Renormalise the kept weights to sum to 1.0
    if kept_indices:
        kept_weight_sum = sum(float(weights[i]) for i in kept_indices)
        renorm_factor = 1.0 / kept_weight_sum if kept_weight_sum > 0 else 1.0
    else:
        renorm_factor = 1.0

    # Build position list
    positions = []
    for i in kept_indices:
        ticker = valid_tickers[i]
        w_renorm = float(weights[i]) * renorm_factor
        price = float(current_prices[ticker])
        target_value_usd = w_renorm * portfolio_value

        if fractional_shares:
            # Exact fractional share count
            shares = round(target_value_usd / price, 4)
            if shares < 0.0001:
                continue
        else:
            # Round to nearest whole share
            shares = int(round(target_value_usd / price))
            if shares < 1:
                shares = 1

        actual_value_usd = shares * price
        positions.append({
            'ticker': ticker,
            'name': name_map.get(ticker, ticker),
            'weight': w_renorm,
            'price': price,
            'shares': shares,
            'value': actual_value_usd,
            'value_gbp': actual_value_usd / fx_rate,
        })

    # Iteratively absorb residual cash (only for whole-share mode)
    invested_usd = sum(p['value'] for p in positions)
    cash_usd = portfolio_value - invested_usd

    if not fractional_shares:
        max_iterations = 20
        iteration = 0
        while cash_usd > 0 and iteration < max_iterations:
            iteration += 1
            best_idx = -1
            best_gap = 0
            for j, p in enumerate(positions):
                target_value = p['weight'] * portfolio_value
                gap = target_value - p['value']
                if (gap > p['price'] and gap > best_gap
                        and p['price'] <= cash_usd):
                    best_gap = gap
                    best_idx = j

            if best_idx == -1:
                break

            p = positions[best_idx]
            p['shares'] += 1
            p['value'] = p['shares'] * p['price']
            p['value_gbp'] = p['value'] / fx_rate
            invested_usd = sum(pp['value'] for pp in positions)
            cash_usd = portfolio_value - invested_usd

    # Sort by weight descending
    positions.sort(key=lambda x: -x['weight'])

    # Final residual
    invested_usd = sum(p['value'] for p in positions)
    cash_usd = portfolio_value - invested_usd
    invested_gbp = invested_usd / fx_rate
    cash_gbp = portfolio_value_gbp - invested_gbp

    # Determine regime label
    if blend > 0.1:
        regime = "DEFENSIVE (blended toward equal weight)"
    elif k > 1.2:
        regime = "CONCENTRATED (adaptive k active)"
    elif k > 1.05:
        regime = "TILTED (mild concentration)"
    else:
        regime = "NEUTRAL (standard optimisation)"

    # Next rebalance date
    next_rebalance = (end + timedelta(days=63)).strftime('%Y-%m-%d')

    return {
        'positions': positions,
        'n_positions': len(positions),
        'portfolio_value_gbp': portfolio_value_gbp,
        'portfolio_value_usd': portfolio_value,
        'fx_rate': fx_rate,
        'invested_gbp': invested_gbp,
        'invested_usd': invested_usd,
        'cash_gbp': cash_gbp,
        'cash_usd': cash_usd,
        'k': k,
        'blend': blend,
        'disp_ratio': disp_ratio,
        'rate_chg': rate_chg,
        'spread': current_spread,
        'regime': regime,
        'rebalance_date': end.strftime('%Y-%m-%d'),
        'next_rebalance': next_rebalance,
        'universe_size': len(valid_tickers),
    }

# ============================================================
# USER STATE MANAGEMENT
# ============================================================
def save_user_state(user_id, state):
    path = os.path.join(USER_STATE_DIR, f'{user_id}.json')
    with open(path, 'w') as f:
        json.dump(state, f, default=str, indent=2)

def load_user_state(user_id):
    path = os.path.join(USER_STATE_DIR, f'{user_id}.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

# ============================================================
# HTML TEMPLATE (single-file)
# ============================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Portfolio Optimiser</title>
<style>
  :root {
    --bg: #0f1419;
    --bg-card: #1a1f2e;
    --bg-card-hover: #232938;
    --border: #2a3142;
    --text: #e8eaed;
    --text-dim: #9aa0a6;
    --accent: #5b8def;
    --positive: #4ade80;
    --negative: #f87171;
    --warning: #fbbf24;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, system-ui, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    padding: 24px;
  }
  .container { max-width: 1200px; margin: 0 auto; }

  header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 32px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }
  h1 { font-size: 24px; font-weight: 600; }
  .subtitle { color: var(--text-dim); font-size: 13px; }

  .tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 24px;
    border-bottom: 1px solid var(--border);
  }
  .tab {
    padding: 12px 20px;
    cursor: pointer;
    color: var(--text-dim);
    border-bottom: 2px solid transparent;
    transition: all 0.15s;
    font-size: 14px;
  }
  .tab:hover { color: var(--text); }
  .tab.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }

  .panel { display: none; }
  .panel.active { display: block; }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    transition: background 0.15s;
  }
  .card-label {
    font-size: 12px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
  }
  .card-value {
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 4px;
  }
  .card-sublabel {
    font-size: 12px;
    color: var(--text-dim);
  }
  .positive { color: var(--positive); }
  .negative { color: var(--negative); }
  .neutral { color: var(--text); }

  .explainer-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
  }
  @media (max-width: 768px) {
    .explainer-grid { grid-template-columns: 1fr; }
  }
  .explainer {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
  }
  .explainer h3 {
    font-size: 14px;
    color: var(--accent);
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .explainer p {
    font-size: 14px;
    color: var(--text);
    line-height: 1.6;
  }
  .explainer p + p { margin-top: 10px; }

  table {
    width: 100%;
    border-collapse: collapse;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 24px;
  }
  th, td {
    padding: 12px 16px;
    text-align: right;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
  }
  th {
    background: var(--bg-card-hover);
    color: var(--text-dim);
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
    font-weight: 500;
  }
  th:first-child, td:first-child { text-align: left; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: var(--bg-card-hover); }

  .section-title {
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-dim);
    margin-bottom: 12px;
    margin-top: 24px;
  }

  .toggle-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 16px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 16px;
  }
  .toggle-row input[type="checkbox"] {
    width: 18px;
    height: 18px;
    margin-top: 2px;
    cursor: pointer;
    accent-color: var(--accent);
  }
  .toggle-row label {
    flex: 1;
    cursor: pointer;
  }
  .toggle-title {
    color: var(--text);
    font-size: 14px;
    font-weight: 500;
    display: block;
    margin-bottom: 4px;
  }
  .toggle-help {
    color: var(--text-dim);
    font-size: 12px;
    line-height: 1.5;
  }

  .form-group {
    margin-bottom: 16px;
  }
  .form-group label {
    display: block;
    font-size: 12px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
  }
  .form-group input, .form-group select {
    width: 100%;
    padding: 12px 16px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 16px;
    font-family: inherit;
  }
  .form-group input:focus, .form-group select:focus {
    outline: none;
    border-color: var(--accent);
  }
  button {
    background: var(--accent);
    color: white;
    border: none;
    padding: 14px 32px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    transition: opacity 0.15s;
  }
  button:hover { opacity: 0.9; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }

  .loading {
    text-align: center;
    padding: 40px;
    color: var(--text-dim);
    font-size: 14px;
  }
  .error {
    background: rgba(248, 113, 113, 0.1);
    border: 1px solid var(--negative);
    color: var(--negative);
    padding: 16px;
    border-radius: 6px;
    margin-bottom: 16px;
    font-size: 14px;
  }
  .info-banner {
    background: rgba(91, 141, 239, 0.1);
    border: 1px solid var(--accent);
    color: var(--accent);
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 16px;
    font-size: 13px;
  }
  .regime-banner {
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 16px;
    font-size: 13px;
    font-weight: 500;
    text-align: center;
  }
  .regime-NEUTRAL { background: rgba(154, 160, 166, 0.1);
                    border: 1px solid var(--text-dim); color: var(--text); }
  .regime-TILTED  { background: rgba(91, 141, 239, 0.1);
                    border: 1px solid var(--accent); color: var(--accent); }
  .regime-CONCENTRATED { background: rgba(251, 191, 36, 0.1);
                         border: 1px solid var(--warning); color: var(--warning); }
  .regime-DEFENSIVE { background: rgba(74, 222, 128, 0.1);
                      border: 1px solid var(--positive); color: var(--positive); }

  /* Charts */
  .chart-section { margin-top: 32px; }
  .chart-wrapper {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
  }
  .chart-title {
    font-size: 15px;
    color: var(--text);
    margin: 0 0 4px 0;
    font-weight: 500;
  }
  .scale-tag {
    display: inline-block;
    padding: 2px 8px;
    margin-left: 8px;
    background: rgba(148, 163, 184, 0.15);
    color: var(--text-dim);
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.5px;
    border-radius: 4px;
    vertical-align: 2px;
    text-transform: uppercase;
  }
  .chart-sub {
    font-size: 12px;
    color: var(--text-dim);
    margin: 0 0 14px 0;
  }
  .chart-canvas-box {
    position: relative;
    height: 360px;
    width: 100%;
  }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body>
<div class="container">

  <header>
    <div>
      <h1>Portfolio Optimiser</h1>
      <div class="subtitle">Adaptive Sharpe optimisation with regime detection</div>
    </div>
    <div class="subtitle">
      {{ historical.summary.alpha_ew_mean * 100 | round(2) if historical.summary }}% alpha
      &middot; backtested 2005-2026
    </div>
  </header>

  <div class="tabs">
    <div class="tab active" onclick="showTab('dashboard')">Performance</div>
    <div class="tab" onclick="showTab('portfolio')">Generate Portfolio</div>
    <div class="tab" onclick="showTab('saved')">My Portfolio</div>
  </div>

  <!-- ============================================================ -->
  <!-- DASHBOARD TAB -->
  <!-- ============================================================ -->
  <div id="dashboard" class="panel active">

    <div class="explainer-grid">
      <div class="explainer">
        <h3>What this does</h3>
        <p>This tool picks a portfolio of stocks designed to grow your money
        more steadily than buying the whole market. It looks at the largest
        ~800 US companies, analyses how they have moved together over the
        past several months, and chooses the combination that has the best
        balance of growth and safety.</p>
        <p>The portfolio is updated every 3 months. You enter how much money
        you want to invest, and the tool tells you exactly which stocks to
        buy and how many shares.</p>
      </div>
      <div class="explainer">
        <h3>How the model works</h3>
        <p>The optimiser maximises the portfolio's risk-adjusted return
        (the Sharpe ratio) by finding stock weightings where expected
        return is highest relative to volatility. It uses Ledoit-Wolf
        covariance shrinkage to stabilise the estimate.</p>
        <p>It then adapts based on market conditions: when stocks are
        moving uniformly (low dispersion), it concentrates more aggressively;
        when credit spreads widen and rates fall (signalling crisis), it
        blends toward equal weight as a defensive measure.</p>
      </div>
    </div>

    <div class="section-title">Headline Performance (vs Equal-Weight Benchmark)</div>
    <div class="grid">
      <div class="card">
        <div class="card-label">Annual Alpha</div>
        <div class="card-value positive">
          +{{ (historical.summary.alpha_ew_mean * 100) | round(2) if historical.summary }}%
        </div>
        <div class="card-sublabel">
          &plusmn; {{ (historical.summary.alpha_ew_std * 100) | round(2) if historical.summary }}%
          across {{ historical.summary.n_offsets if historical.summary }} offsets
        </div>
      </div>
      <div class="card">
        <div class="card-label">Portfolio Sharpe</div>
        <div class="card-value positive">
          {{ portfolio_sharpe | round(2) }}
        </div>
        <div class="card-sublabel">
          vs SPY {{ spy_sharpe | round(2) }} &middot; vs EW {{ ew_sharpe | round(2) }}
        </div>
      </div>
      <div class="card">
        <div class="card-label">Worst-Case Alpha Floor</div>
        <div class="card-value positive">
          +{{ (historical.summary.floor_ew * 100) | round(2) if historical.summary }}%
        </div>
        <div class="card-sublabel">Bottom 2.5% of outcomes</div>
      </div>
      <div class="card">
        <div class="card-label">Annual Return</div>
        <div class="card-value positive">
          +{{ (historical.summary.portfolio_ann_mean * 100) | round(2) if historical.summary }}%
        </div>
        <div class="card-sublabel">
          vs SPY +{{ (historical.summary.spy_ann_mean * 100) | round(2) if historical.summary }}%
        </div>
      </div>
    </div>

    <div class="section-title">Performance vs S&P 500 (SPY)</div>
    <div class="grid">
      <div class="card">
        <div class="card-label">Annual Alpha vs SPY</div>
        <div class="card-value positive">
          +{{ (historical.summary.alpha_spy_mean * 100) | round(2) if historical.summary }}%
        </div>
        <div class="card-sublabel">
          &plusmn; {{ (historical.summary.alpha_spy_std * 100) | round(2) if historical.summary }}%
        </div>
      </div>
      <div class="card">
        <div class="card-label">Cumulative Alpha vs SPY</div>
        <div class="card-value positive">
          +{{ (historical.summary.alpha_spy_cum_mean * 100) | round(0) if historical.summary }}%
        </div>
        <div class="card-sublabel">Extra return since 2005 vs holding SPY</div>
      </div>
      <div class="card">
        <div class="card-label">Cumulative Return</div>
        <div class="card-value positive">
          +{{ (historical.summary.portfolio_cum_mean * 100) | round(0) if historical.summary }}%
        </div>
        <div class="card-sublabel">
          If you'd followed this since 2005 (vs SPY +{{ (historical.summary.spy_cum_mean * 100) | round(0) if historical.summary }}%)
        </div>
      </div>
      <div class="card">
        <div class="card-label">Floor vs SPY</div>
        <div class="card-value {% if historical.summary.floor_spy > 0 %}positive{% else %}negative{% endif %}">
          {{ '+' if historical.summary.floor_spy > 0 else '' }}{{ (historical.summary.floor_spy * 100) | round(2) if historical.summary }}%
        </div>
        <div class="card-sublabel">Worst-case alpha vs SPY</div>
      </div>
    </div>

    {% if historical.eras %}
    <div class="section-title">Performance by Market Era</div>
    <table>
      <thead>
        <tr>
          <th>Era</th>
          <th>Portfolio</th>
          <th>EqWt</th>
          <th>SPY</th>
          <th>&alpha; vs EW</th>
          <th>&alpha; vs SPY</th>
          <th>Sharpe</th>
        </tr>
      </thead>
      <tbody>
        {% for era_name, era in historical.eras.items() %}
        <tr>
          <td>{{ era_name }}</td>
          <td class="{% if era.port_ann > 0 %}positive{% else %}negative{% endif %}">
            {{ '+' if era.port_ann > 0 else '' }}{{ (era.port_ann * 100) | round(2) }}%
          </td>
          <td class="{% if era.ew_ann > 0 %}positive{% else %}negative{% endif %}">
            {{ '+' if era.ew_ann > 0 else '' }}{{ (era.ew_ann * 100) | round(2) }}%
          </td>
          <td class="{% if era.spy_ann > 0 %}positive{% else %}negative{% endif %}">
            {{ '+' if era.spy_ann > 0 else '' }}{{ (era.spy_ann * 100) | round(2) }}%
          </td>
          <td class="{% if era.alpha_ew > 0 %}positive{% else %}negative{% endif %}">
            {{ '+' if era.alpha_ew > 0 else '' }}{{ (era.alpha_ew * 100) | round(2) }}%
          </td>
          <td class="{% if era.alpha_spy > 0 %}positive{% else %}negative{% endif %}">
            {{ '+' if era.alpha_spy > 0 else '' }}{{ (era.alpha_spy * 100) | round(2) }}%
          </td>
          <td class="neutral">{{ era.port_sharpe | round(2) }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% endif %}

    {% if historical.years %}
    <div class="section-title">Year by Year</div>
    <table>
      <thead>
        <tr>
          <th>Year</th>
          <th>Portfolio</th>
          <th>EqWt</th>
          <th>SPY</th>
          <th>&alpha; vs EW</th>
          <th>&alpha; vs SPY</th>
        </tr>
      </thead>
      <tbody>
        {% for year, yr in historical.years.items() if not year.startswith('_') %}
        <tr>
          <td>{{ year }}</td>
          <td class="{% if yr.port > 0 %}positive{% else %}negative{% endif %}">
            {{ '+' if yr.port > 0 else '' }}{{ (yr.port * 100) | round(2) }}%
          </td>
          <td class="{% if yr.ew > 0 %}positive{% else %}negative{% endif %}">
            {{ '+' if yr.ew > 0 else '' }}{{ (yr.ew * 100) | round(2) }}%
          </td>
          <td class="{% if yr.spy > 0 %}positive{% else %}negative{% endif %}">
            {{ '+' if yr.spy > 0 else '' }}{{ (yr.spy * 100) | round(2) }}%
          </td>
          <td class="{% if yr.alpha_ew > 0 %}positive{% else %}negative{% endif %}">
            {{ '+' if yr.alpha_ew > 0 else '' }}{{ (yr.alpha_ew * 100) | round(2) }}%
          </td>
          <td class="{% if yr.alpha_spy > 0 %}positive{% else %}negative{% endif %}">
            {{ '+' if yr.alpha_spy > 0 else '' }}{{ (yr.alpha_spy * 100) | round(2) }}%
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% endif %}

    {% if charts and charts.headline %}
    <div class="chart-section">
      <div class="section-title">Cumulative Performance</div>

      <div class="chart-wrapper">
        <div class="chart-title">
          Wealth curve since 2005 &mdash; Portfolio vs SPY vs Equal-Weight
          <span class="scale-tag">Log scale</span>
        </div>
        <div class="chart-sub">
          &pound;1 compounded from the start of the backtest. Shown for the
          median offset across the 63-offset ensemble (a real, achievable
          path whose final wealth is the cross-offset median).
        </div>
        <div class="chart-canvas-box">
          <canvas id="chart-headline"></canvas>
        </div>
      </div>

      {% if charts.fan and charts.fan.offsets %}
      <div class="chart-wrapper">
        <div class="chart-title">
          Portfolio wealth curve &mdash; all 63 quarterly rebalance offsets
          <span class="scale-tag">Log scale</span>
        </div>
        <div class="chart-sub">
          Each light-green line is a backtest run with a different rebalance
          start day. SPY (blue) and Equal-Weight (amber) reference lines use
          the median offset. Tight bundling above both benchmarks evidences
          robustness to rebalance-timing effects: all 63 offsets remain
          positive vs SPY and vs equal-weight.
        </div>
        <div class="chart-canvas-box">
          <canvas id="chart-fan"></canvas>
        </div>
      </div>
      {% endif %}
    </div>
    {% endif %}
  </div>

  <!-- ============================================================ -->
  <!-- GENERATE PORTFOLIO TAB -->
  <!-- ============================================================ -->
  <div id="portfolio" class="panel">
    <div class="explainer">
      <h3>Generate your portfolio</h3>
      <p>Enter your portfolio value below. The tool will download current
      stock prices, run the optimisation, and tell you exactly which stocks
      to buy. This takes 2-3 minutes.</p>
    </div>

    <div style="margin-top: 24px; max-width: 500px;">
      <div class="form-group">
        <label>Portfolio Value (GBP)</label>
        <input type="number" id="portfolioValue" value="10000" min="500" step="100">
      </div>
      <div class="toggle-row">
        <input type="checkbox" id="fractionalShares">
        <label for="fractionalShares">
          <span class="toggle-title">Fractional shares</span>
          <span class="toggle-help">
            Enable if your broker supports buying partial shares (Trading 212,
            Freetrade, Robinhood). Disable for brokers that only allow whole
            shares (Hargreaves Lansdown, Interactive Brokers default).
            Fractional shares give exact target weights with zero cash residual.
          </span>
        </label>
      </div>
      <button onclick="generatePortfolio()">Generate Portfolio</button>
    </div>

    <div id="portfolioResults" style="margin-top: 32px;"></div>
  </div>

  <!-- ============================================================ -->
  <!-- SAVED PORTFOLIO TAB -->
  <!-- ============================================================ -->
  <div id="saved" class="panel">
    <div class="explainer">
      <h3>Your saved portfolio</h3>
      <p>If you have generated a portfolio and clicked "Save Portfolio",
      it will appear here with the rebalance date and current state. When
      it's time to rebalance, click the rebalance button to compute the
      exact buy/sell trades needed.</p>
    </div>
    <div id="savedContent" style="margin-top: 24px;">
      <div class="loading">No saved portfolio yet.</div>
    </div>
  </div>

</div>

<script>
function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById(name).classList.add('active');

  if (name === 'saved') loadSaved();
}

async function generatePortfolio() {
  const value = parseFloat(document.getElementById('portfolioValue').value);
  const fractional = document.getElementById('fractionalShares').checked;
  const results = document.getElementById('portfolioResults');

  results.innerHTML = '<div class="loading">Downloading prices and optimising... (2-3 minutes)</div>';

  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        portfolio_value_gbp: value,
        fractional_shares: fractional
      })
    });
    const data = await response.json();

    if (data.error) {
      results.innerHTML = `<div class="error">Error: ${data.error}</div>`;
      return;
    }

    renderPortfolio(data, results);
  } catch (e) {
    results.innerHTML = `<div class="error">Request failed: ${e.message}</div>`;
  }
}

function renderPortfolio(data, container) {
  const regimeClass = 'regime-' + data.regime.split(' ')[0];

  let html = `
    <div class="regime-banner ${regimeClass}">
      Current regime: ${data.regime}
    </div>

    <div class="info-banner">
      Portfolio: &pound;${data.portfolio_value_gbp.toFixed(2)} ($${data.portfolio_value_usd.toFixed(2)} at FX ${data.fx_rate.toFixed(4)})
    </div>

    <div class="grid">
      <div class="card">
        <div class="card-label">Positions</div>
        <div class="card-value neutral">${data.n_positions}</div>
        <div class="card-sublabel">From ${data.universe_size} eligible stocks</div>
      </div>
      <div class="card">
        <div class="card-label">Invested</div>
        <div class="card-value neutral">&pound;${data.invested_gbp.toFixed(2)}</div>
        <div class="card-sublabel">Cash residual: &pound;${data.cash_gbp.toFixed(2)}</div>
      </div>
      <div class="card">
        <div class="card-label">Concentration (k)</div>
        <div class="card-value neutral">${data.k.toFixed(2)}</div>
        <div class="card-sublabel">Blend: ${(data.blend * 100).toFixed(1)}%</div>
      </div>
      <div class="card">
        <div class="card-label">Next Rebalance</div>
        <div class="card-value neutral" style="font-size:18px">${data.next_rebalance}</div>
        <div class="card-sublabel">In ~63 trading days</div>
      </div>
    </div>

    <div class="section-title">Market Conditions</div>
    <div class="grid">
      <div class="card">
        <div class="card-label">Dispersion Ratio</div>
        <div class="card-value neutral">${data.disp_ratio.toFixed(2)}x</div>
        <div class="card-sublabel">${data.disp_ratio > 1 ? 'High (diverse)' : 'Low (uniform)'}</div>
      </div>
      <div class="card">
        <div class="card-label">6mo Rate Change</div>
        <div class="card-value ${data.rate_chg >= 0 ? 'positive' : 'negative'}">
          ${data.rate_chg >= 0 ? '+' : ''}${data.rate_chg.toFixed(2)}%
        </div>
        <div class="card-sublabel">Fed funds direction</div>
      </div>
      <div class="card">
        <div class="card-label">Credit Spread</div>
        <div class="card-value neutral">${data.spread.toFixed(2)}%</div>
        <div class="card-sublabel">HY vs Treasury</div>
      </div>
    </div>

    <div class="section-title">Target Positions</div>
    <table>
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Company</th>
          <th>Weight</th>
          <th>Price (USD)</th>
          <th>Shares</th>
          <th>Value (GBP)</th>
        </tr>
      </thead>
      <tbody>
  `;

  for (const p of data.positions) {
    const sharesDisplay = (p.shares % 1 === 0)
      ? p.shares
      : p.shares.toFixed(4);
    html += `
      <tr>
        <td><strong>${p.ticker}</strong></td>
        <td style="color: var(--text-dim); font-size: 12px;">${p.name || p.ticker}</td>
        <td>${(p.weight * 100).toFixed(2)}%</td>
        <td>$${p.price.toFixed(2)}</td>
        <td>${sharesDisplay}</td>
        <td>&pound;${p.value_gbp.toFixed(2)}</td>
      </tr>
    `;
  }

  html += `
      </tbody>
    </table>
    <button onclick='savePortfolio(${JSON.stringify(JSON.stringify(data))})'>
      Save Portfolio
    </button>
  `;

  container.innerHTML = html;
}

async function savePortfolio(dataStr) {
  const data = JSON.parse(dataStr);
  try {
    const response = await fetch('/api/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({user_id: 'default', portfolio: data})
    });
    const result = await response.json();
    if (result.success) {
      alert('Portfolio saved! Switch to "My Portfolio" tab to view.');
    }
  } catch (e) {
    alert('Save failed: ' + e.message);
  }
}

async function loadSaved() {
  const container = document.getElementById('savedContent');
  container.innerHTML = '<div class="loading">Loading...</div>';

  try {
    const response = await fetch('/api/load?user_id=default');
    const data = await response.json();

    if (!data || data.error) {
      container.innerHTML = '<div class="loading">No saved portfolio yet. Generate one first.</div>';
      return;
    }

    const totalValue = data.portfolio.positions.reduce(
      (sum, p) => sum + p.value_gbp, 0);

    let html = `
      <div class="info-banner">
        Saved on ${data.saved_at} &middot; Next rebalance scheduled: ${data.portfolio.next_rebalance}
      </div>

      <div class="grid">
        <div class="card">
          <div class="card-label">Saved Portfolio Value</div>
          <div class="card-value neutral">&pound;${totalValue.toFixed(2)}</div>
          <div class="card-sublabel">${data.portfolio.n_positions} positions</div>
        </div>
        <div class="card">
          <div class="card-label">Saved Date</div>
          <div class="card-value neutral" style="font-size:18px">${data.saved_at}</div>
          <div class="card-sublabel">When you committed</div>
        </div>
        <div class="card">
          <div class="card-label">Rebalance Date</div>
          <div class="card-value neutral" style="font-size:18px">${data.portfolio.next_rebalance}</div>
          <div class="card-sublabel">Scheduled rebalance</div>
        </div>
      </div>

      <div class="section-title">Saved Positions</div>
      <table>
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Company</th>
            <th>Weight</th>
            <th>Shares</th>
            <th>Value (GBP)</th>
          </tr>
        </thead>
        <tbody>
    `;

    for (const p of data.portfolio.positions) {
      const sharesDisplay = (p.shares % 1 === 0) ? p.shares : p.shares.toFixed(4);
      html += `
        <tr>
          <td><strong>${p.ticker}</strong></td>
          <td style="color: var(--text-dim); font-size: 12px;">${p.name || p.ticker}</td>
          <td>${(p.weight * 100).toFixed(2)}%</td>
          <td>${sharesDisplay}</td>
          <td>&pound;${p.value_gbp.toFixed(2)}</td>
        </tr>
      `;
    }

    html += `
        </tbody>
      </table>

      <div class="section-title">Rebalance</div>
      <div class="explainer" style="margin-bottom: 16px;">
        <p>To rebalance, enter your current portfolio value (or leave blank
        to estimate from current prices). The tool will compare your saved
        positions to a fresh optimisation and tell you exactly which trades
        to make.</p>
      </div>
      <div style="max-width: 500px;">
        <div class="form-group">
          <label>Current Portfolio Value (GBP, leave blank to estimate)</label>
          <input type="number" id="rebalanceValue" placeholder="Auto-estimate" min="0" step="100">
        </div>
        <div class="toggle-row">
          <input type="checkbox" id="rebalanceFractional">
          <label for="rebalanceFractional">
            <span class="toggle-title">Fractional shares</span>
            <span class="toggle-help">Match the setting you used when saving.</span>
          </label>
        </div>
        <button onclick="runRebalance()">Compute Rebalance Trades</button>
      </div>

      <div id="rebalanceResults" style="margin-top: 32px;"></div>
    `;

    container.innerHTML = html;
  } catch (e) {
    container.innerHTML = `<div class="error">Load failed: ${e.message}</div>`;
  }
}

async function runRebalance() {
  const valueInput = document.getElementById('rebalanceValue').value;
  const fractional = document.getElementById('rebalanceFractional').checked;
  const results = document.getElementById('rebalanceResults');

  results.innerHTML = '<div class="loading">Computing rebalance trades... (2-3 minutes)</div>';

  const body = {
    user_id: 'default',
    fractional_shares: fractional,
  };
  if (valueInput) body.new_value_gbp = parseFloat(valueInput);

  try {
    const response = await fetch('/api/rebalance', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const data = await response.json();

    if (data.error) {
      results.innerHTML = `<div class="error">${data.error}</div>`;
      return;
    }

    renderRebalance(data, results);
  } catch (e) {
    results.innerHTML = `<div class="error">Rebalance failed: ${e.message}</div>`;
  }
}

function renderRebalance(data, container) {
  const valueChange = data.new_value_gbp - data.old_value_gbp;
  const valueChangePct = (valueChange / data.old_value_gbp) * 100;
  const changeClass = valueChange >= 0 ? 'positive' : 'negative';
  const changeSign = valueChange >= 0 ? '+' : '';

  let html = `
    <div class="grid">
      <div class="card">
        <div class="card-label">Old Value</div>
        <div class="card-value neutral">&pound;${data.old_value_gbp.toFixed(2)}</div>
        <div class="card-sublabel">When saved</div>
      </div>
      <div class="card">
        <div class="card-label">New Value</div>
        <div class="card-value neutral">&pound;${data.new_value_gbp.toFixed(2)}</div>
        <div class="card-sublabel">Today</div>
      </div>
      <div class="card">
        <div class="card-label">Change</div>
        <div class="card-value ${changeClass}">
          ${changeSign}&pound;${valueChange.toFixed(2)}
        </div>
        <div class="card-sublabel">${changeSign}${valueChangePct.toFixed(2)}%</div>
      </div>
      <div class="card">
        <div class="card-label">Trades Required</div>
        <div class="card-value neutral">${data.n_trades}</div>
        <div class="card-sublabel">${data.n_sells} sells, ${data.n_buys} buys</div>
      </div>
    </div>

    <div class="section-title">Trade List</div>
    <div class="info-banner">
      Execute sells first to free up cash for buys.
      Total sells: &pound;${data.total_sell_gbp.toFixed(2)} &middot;
      Total buys: &pound;${data.total_buy_gbp.toFixed(2)}
    </div>
    <table>
      <thead>
        <tr>
          <th>Action</th>
          <th>Ticker</th>
          <th>Company</th>
          <th>Old</th>
          <th>New</th>
          <th>Delta</th>
          <th>Price (USD)</th>
          <th>Value (GBP)</th>
        </tr>
      </thead>
      <tbody>
  `;

  for (const t of data.trades) {
    const actionClass = t.action === 'BUY' ? 'positive' : 'negative';
    const isNew = t.is_new ? ' (new)' : '';
    const isClose = t.is_close ? ' (close)' : '';
    const oldDisplay = (t.old_shares % 1 === 0) ? t.old_shares : t.old_shares.toFixed(4);
    const newDisplay = (t.new_shares % 1 === 0) ? t.new_shares : t.new_shares.toFixed(4);
    const deltaDisplay = (t.delta_shares % 1 === 0) ? t.delta_shares : t.delta_shares.toFixed(4);

    html += `
      <tr>
        <td><strong class="${actionClass}">${t.action}</strong>${isNew}${isClose}</td>
        <td><strong>${t.ticker}</strong></td>
        <td style="color: var(--text-dim); font-size: 12px;">${t.name}</td>
        <td>${oldDisplay}</td>
        <td>${newDisplay}</td>
        <td>${deltaDisplay}</td>
        <td>$${t.price.toFixed(2)}</td>
        <td>&pound;${t.delta_value_gbp.toFixed(2)}</td>
      </tr>
    `;
  }

  html += `
      </tbody>
    </table>

    <button onclick='commitRebalance(${JSON.stringify(JSON.stringify(data.new_portfolio))})'>
      I've Executed These Trades - Update Saved Portfolio
    </button>
  `;

  container.innerHTML = html;
}

async function commitRebalance(portfolioStr) {
  const portfolio = JSON.parse(portfolioStr);
  try {
    const response = await fetch('/api/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({user_id: 'default', portfolio: portfolio})
    });
    const result = await response.json();
    if (result.success) {
      alert('Rebalance committed! Reloading saved portfolio...');
      loadSaved();
    }
  } catch (e) {
    alert('Commit failed: ' + e.message);
  }
}

// ============================================================
// CHART RENDERING
// ============================================================
const chartsData = {{ charts | tojson | safe }};

function tsFor(dateStr) { return new Date(dateStr).getTime(); }

function buildHeadlineChart() {
  const el = document.getElementById('chart-headline');
  if (!el || !chartsData || !chartsData.headline) return;
  const h = chartsData.headline;
  const xs = h.dates.map(tsFor);
  const mk = (label, ys, color) => ({
    label: label,
    data: xs.map((x, i) => ({ x: x, y: ys[i] })),
    borderColor: color,
    backgroundColor: color,
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.1,
  });
  new Chart(el, {
    type: 'line',
    data: {
      datasets: [
        mk('Portfolio', h.portfolio_cum, '#4ade80'),
        mk('SPY',       h.spy_cum,       '#60a5fa'),
        mk('Equal-Wt',  h.ew_cum,        '#f59e0b'),
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      animation: false,
      interaction: { mode: 'nearest', intersect: false },
      scales: {
        x: {
          type: 'linear',
          ticks: {
            color: '#94a3b8',
            callback: (v) => new Date(v).getFullYear(),
            maxTicksLimit: 12,
          },
          grid: { color: 'rgba(148,163,184,0.08)' },
        },
        y: {
          type: 'logarithmic',
          ticks: {
            color: '#94a3b8',
            callback: (v) => v.toFixed(v < 2 ? 1 : 0) + 'x',
          },
          grid: { color: 'rgba(148,163,184,0.08)' },
        },
      },
      plugins: {
        legend: { labels: { color: '#e2e8f0' } },
        tooltip: {
          callbacks: {
            title: (items) => {
              const d = new Date(items[0].parsed.x);
              return d.toISOString().slice(0, 10);
            },
            label: (item) => item.dataset.label + ': ' +
              item.parsed.y.toFixed(2) + 'x',
          },
        },
      },
    },
  });
}

function buildFanChart() {
  const el = document.getElementById('chart-fan');
  if (!el || !chartsData || !chartsData.fan || !chartsData.fan.offsets) return;
  const offsets = chartsData.fan.offsets;
  const ref = chartsData.headline;  // median offset - used for SPY/EW refs

  // Compute median portfolio cum return across offsets at each index.
  const minLen = Math.min(...offsets.map(o => o.portfolio_cum.length));
  const medianValues = [];
  const medianDates = [];
  for (let i = 0; i < minLen; i++) {
    const vs = offsets.map(o => o.portfolio_cum[i]).sort((a, b) => a - b);
    const mid = Math.floor(vs.length / 2);
    medianValues.push(vs.length % 2 ? vs[mid] : (vs[mid - 1] + vs[mid]) / 2);
    medianDates.push(offsets[0].dates[i]);
  }

  // 63 translucent portfolio lines
  const fanDatasets = offsets.map((o) => ({
    data: o.dates.map((d, j) => ({ x: tsFor(d), y: o.portfolio_cum[j] })),
    borderColor: 'rgba(74, 222, 128, 0.12)',
    borderWidth: 1,
    pointRadius: 0,
    tension: 0.1,
    order: 3,
  }));

  // SPY reference line (from median offset)
  if (ref && ref.spy_cum) {
    fanDatasets.push({
      label: 'SPY',
      data: ref.dates.map((d, j) => ({ x: tsFor(d), y: ref.spy_cum[j] })),
      borderColor: '#60a5fa',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.1,
      order: 2,
    });
  }

  // Equal-Weight reference line (from median offset)
  if (ref && ref.ew_cum) {
    fanDatasets.push({
      label: 'Equal-Wt',
      data: ref.dates.map((d, j) => ({ x: tsFor(d), y: ref.ew_cum[j] })),
      borderColor: '#f59e0b',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.1,
      order: 2,
    });
  }

  // Median portfolio line on top
  fanDatasets.push({
    label: 'Portfolio (median offset)',
    data: medianDates.map((d, j) => ({ x: tsFor(d), y: medianValues[j] })),
    borderColor: '#4ade80',
    borderWidth: 2.5,
    pointRadius: 0,
    tension: 0.1,
    order: 1,
  });

  new Chart(el, {
    type: 'line',
    data: { datasets: fanDatasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      animation: false,
      scales: {
        x: {
          type: 'linear',
          ticks: {
            color: '#94a3b8',
            callback: (v) => new Date(v).getFullYear(),
            maxTicksLimit: 12,
          },
          grid: { color: 'rgba(148,163,184,0.08)' },
        },
        y: {
          type: 'logarithmic',
          ticks: {
            color: '#94a3b8',
            callback: (v) => v.toFixed(v < 2 ? 1 : 0) + 'x',
          },
          grid: { color: 'rgba(148,163,184,0.08)' },
        },
      },
      plugins: {
        legend: {
          labels: {
            color: '#e2e8f0',
            filter: (item) => !!item.text,  // hide the 63 unlabelled lines
          },
        },
        tooltip: { enabled: false },
      },
    },
  });
}

// Run after DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    buildHeadlineChart();
    buildFanChart();
  });
} else {
  buildHeadlineChart();
  buildFanChart();
}
</script>
</body>
</html>
"""

# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    # Compute average Sharpe ratios across eras (weighted by period count)
    portfolio_sharpe = 0
    spy_sharpe = 0
    ew_sharpe = 0

    if historical_data and 'eras' in historical_data:
        eras = historical_data['eras']
        total_n = sum(e.get('n', 0) for e in eras.values())
        if total_n > 0:
            portfolio_sharpe = sum(
                e.get('port_sharpe', 0) * e.get('n', 0)
                for e in eras.values()) / total_n

            # SPY and EW Sharpes need to be computed from era data
            # Use a simple average if not in primary.json
            spy_sharpe = historical_data.get('summary', {}).get(
                'spy_sharpe', 0.86)
            ew_sharpe = historical_data.get('summary', {}).get(
                'ew_sharpe', 0.87)

    return render_template_string(
        HTML_TEMPLATE,
        historical=historical_data,
        charts=charts_data,
        portfolio_sharpe=portfolio_sharpe,
        spy_sharpe=spy_sharpe,
        ew_sharpe=ew_sharpe,
    )

@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    portfolio_value_gbp = float(data.get('portfolio_value_gbp', 10000))
    fractional = bool(data.get('fractional_shares', False))
    result = generate_portfolio(portfolio_value_gbp,
                                 fractional_shares=fractional)
    return jsonify(result)

@app.route('/api/save', methods=['POST'])
def api_save():
    data = request.json
    user_id = data.get('user_id', 'default')
    portfolio = data.get('portfolio', {})
    state = {
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'portfolio': portfolio,
    }
    save_user_state(user_id, state)
    return jsonify({'success': True})

@app.route('/api/load')
def api_load():
    user_id = request.args.get('user_id', 'default')
    state = load_user_state(user_id)
    return jsonify(state if state else {'error': 'No saved state'})

@app.route('/api/rebalance', methods=['POST'])
def api_rebalance():
    """Compare saved portfolio to fresh optimisation, return trade list."""
    data = request.json
    user_id = data.get('user_id', 'default')
    new_value_gbp = data.get('new_value_gbp')
    fractional = bool(data.get('fractional_shares', False))

    saved = load_user_state(user_id)
    if not saved or 'portfolio' not in saved:
        return jsonify({'error': 'No saved portfolio to rebalance'})

    old_portfolio = saved['portfolio']

    # If user didn't supply new value, estimate from current prices
    if new_value_gbp is None:
        try:
            tickers = [p['ticker'] for p in old_portfolio['positions']]
            current = yf.download(tickers, period='5d', progress=False,
                                  auto_adjust=True)
            if 'Close' in current.columns.get_level_values(0):
                latest = current['Close'].iloc[-1]
            else:
                latest = current.iloc[-1]

            fx_rate = get_gbp_usd()
            total_usd = 0
            for p in old_portfolio['positions']:
                tk = p['ticker']
                if tk in latest.index and not pd.isna(latest[tk]):
                    total_usd += float(latest[tk]) * p['shares']
            new_value_gbp = total_usd / fx_rate
        except Exception as e:
            return jsonify({'error': f'Could not estimate value: {str(e)}'})

    # Generate fresh portfolio at new value
    new_portfolio = generate_portfolio(float(new_value_gbp),
                                        fractional_shares=fractional)
    if 'error' in new_portfolio:
        return jsonify(new_portfolio)

    # Build map of old shares by ticker
    old_shares = {p['ticker']: p['shares'] for p in old_portfolio['positions']}
    old_names = {p['ticker']: p.get('name', p['ticker'])
                 for p in old_portfolio['positions']}
    new_shares = {p['ticker']: p['shares'] for p in new_portfolio['positions']}
    new_prices = {p['ticker']: p['price'] for p in new_portfolio['positions']}
    new_names = {p['ticker']: p.get('name', p['ticker'])
                 for p in new_portfolio['positions']}

    # Compute trade list
    all_tickers = sorted(set(old_shares.keys()) | set(new_shares.keys()))
    trades = []

    fx_rate = new_portfolio['fx_rate']

    for ticker in all_tickers:
        old_qty = old_shares.get(ticker, 0)
        new_qty = new_shares.get(ticker, 0)
        delta = new_qty - old_qty

        if abs(delta) < 0.0001:
            continue

        # Get price - prefer new portfolio's price, fall back to fetching
        price = new_prices.get(ticker)
        if price is None:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period='5d')
                price = float(hist['Close'].iloc[-1]) if len(hist) > 0 else 0
            except:
                price = 0

        delta_value_usd = delta * price
        delta_value_gbp = delta_value_usd / fx_rate

        action = 'BUY' if delta > 0 else 'SELL'
        name = new_names.get(ticker) or old_names.get(ticker, ticker)

        # Format share quantity
        qty_display = (abs(delta) if not isinstance(delta, float)
                       else round(abs(delta), 4))

        trades.append({
            'ticker': ticker,
            'name': name,
            'action': action,
            'old_shares': old_qty,
            'new_shares': new_qty,
            'delta_shares': abs(delta),
            'price': price,
            'delta_value_gbp': abs(delta_value_gbp),
            'is_new': ticker not in old_shares,
            'is_close': ticker not in new_shares,
        })

    # Sort: sells first, then buys, both by value descending
    sells = sorted([t for t in trades if t['action'] == 'SELL'],
                   key=lambda x: -x['delta_value_gbp'])
    buys = sorted([t for t in trades if t['action'] == 'BUY'],
                  key=lambda x: -x['delta_value_gbp'])
    trade_list = sells + buys

    total_sell_gbp = sum(t['delta_value_gbp'] for t in sells)
    total_buy_gbp = sum(t['delta_value_gbp'] for t in buys)

    return jsonify({
        'old_value_gbp': sum(p['value_gbp'] for p in old_portfolio['positions']),
        'new_value_gbp': new_value_gbp,
        'trades': trade_list,
        'n_trades': len(trade_list),
        'n_sells': len(sells),
        'n_buys': len(buys),
        'total_sell_gbp': total_sell_gbp,
        'total_buy_gbp': total_buy_gbp,
        'new_portfolio': new_portfolio,
        'old_saved_at': saved.get('saved_at'),
    })

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("="*60)
    print("PORTFOLIO OPTIMISER")
    print("="*60)
    print("Open http://localhost:5000 in your browser")
    print("="*60)
    app.run(host='127.0.0.1', port=5000, debug=False)
