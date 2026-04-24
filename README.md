# Systematic Equity Portfolio Optimiser

An independent quantitative research project: a systematic long-only US-equity portfolio optimiser, validated with a 63-offset out-of-sample backtest over 2005–2026.

The strategy uses cross-sectional return dispersion and interest-rate signals, conditioned on a macro regime gate, to adaptively tilt portfolio concentration. It has beaten both SPY and an equal-weight benchmark in every one of 63 tested quarterly rebalance start-days across a 21-year backtest covering four distinct market regimes.

---

## Headline results

Tested on the current S&P 500 + NASDAQ 100 universe using survivorship-bias-free [Norgate Data](https://norgatedata.com), with a 126-day training window and quarterly rebalancing. Values shown are the cross-offset mean across all 63 rebalance start-days; the "worst-case" row is the bottom-2.5% outcome.

| Metric                       | Portfolio  | SPY       | Equal-Weight |
|------------------------------|-----------:|----------:|-------------:|
| Annualised return            | +17.08%    | +12.58%   | +8.59%       |
| Sharpe ratio                 | **1.17**   | 0.86      | 0.87         |
| Cumulative return (2005–26)  | +2,165%    | +854%     | +401%        |

| Alpha                            | vs SPY     | vs Equal-Weight |
|----------------------------------|-----------:|----------------:|
| Annualised alpha                 | **+4.12%** | **+7.99%**      |
| Std across 63 offsets            | ±1.50%     | ±1.53%          |
| Worst-case alpha (bottom 2.5%)   | +1.19%     | +4.98%          |
| Offsets with positive alpha      | 63 / 63    | 63 / 63         |

## By market era

| Era        | Years     | Portfolio ann. | SPY ann. | Alpha vs SPY | Sharpe | Max DD  |
|------------|-----------|---------------:|---------:|-------------:|-------:|--------:|
| Crisis     | 2007–10   | +6.15%         | −1.21%   | +7.42%       | 0.32   | −42.2%  |
| Recovery   | 2010–15   | +21.83%        | +16.35%  | +4.90%       | 1.39   | −23.3%  |
| Bull       | 2015–20   | +14.54%        | +11.17%  | +3.11%       | 1.08   | −27.7%  |
| Recent     | 2020–26   | +23.24%        | +18.34%  | +4.33%       | 1.48   | −18.4%  |

Alpha versus SPY is positive in every era. Sharpe varies meaningfully by regime, reflecting the method's dependence on cross-sectional signal dispersion — in the crisis era, when most stocks sold off together, the signal was weaker but still positive.

---

## Methodology

**Universe.** Current S&P 500 + NASDAQ 100 constituents (~600 unique tickers). Survivorship bias is handled via Norgate's historical membership data — at each rebalance date, only stocks actually in the index on that date are eligible.

**Training window.** 126 trading days (~6 months) of daily returns used to estimate the return/covariance structure at each rebalance.

**Rebalancing.** Quarterly (63 trading days). The 63-offset sweep tests the strategy across every possible rebalance start-day within a quarter, producing 63 parallel backtests. This is the key robustness test: a single-path backtest reports one schedule, lucky or unlucky; 63 offsets reveal whether the strategy's edge depends on when you happened to start.

**Signal construction.** Three interacting mechanisms:

1. A **dispersion-responsive adapter** tilts portfolio concentration based on current cross-sectional return dispersion — more concentrated when dispersion is high (optimiser's signal is strong), more diversified when dispersion is low.
2. A **rate-sensitivity component** modulates concentration based on the 6-month change in short rates, reflecting the empirical relationship between monetary policy shifts and factor returns.
3. A **dual-MA regime gate** (50-day vs 200-day SPY) blends toward defensive equal-weight allocation during bear regimes.

**Optimiser.** Mean-variance optimisation (scipy), with concentration controlled by an adaptive parameter rather than hard weight caps.

**Validation.** 63-offset out-of-sample testing across the full 2005–2026 period, with no look-ahead at any stage. Alpha measured against both SPY (buy-and-hold) and an equal-weight benchmark over the same universe and dates.

---

## The app

The Flask application in this repository is the deployable front-end of the system:

- **Historical dashboard:** the results tables above plus year-by-year breakdown and two log-scale wealth-curve charts — a canonical median-offset curve and a 63-offset fan chart.
- **Live portfolio generation:** enter a GBP value, and the app fetches current prices, runs the optimisation, and outputs a trade list with share counts, weights, and cash residuals. Supports both fractional-share and integer-only brokers with iterative cash absorption.
- **Persistence across rebalances:** save a portfolio, view it on return, and commit a rebalance with a confirm-to-commit flow.

Run locally:

```bash
# Requires Python 3.10+ and Anaconda (or adjust launch.bat for system Python)
python app.py
# On Windows, simply double-click launch.bat
```

---

## Reproducing the backtest

The backtest engine itself — the optimiser, 63-offset sweep runner, signal calibration, and feature-engineering pipeline for an in-development LightGBM extension — is kept in a private repository. This repository contains the deployable front-end plus aggregated backtest outputs (`primary.json`, `charts.json`) sufficient to explore the results.

**If you are a prospective employer and would like access to the underlying research code for review, please reach out via LinkedIn.**

---

## Limitations and honest caveats

- **This is a backtest, not live trading.** Simulated returns. No strategy retains its backtest Sharpe in live deployment; transaction costs and market impact erode performance.
- **2005–2026 covers one major bear regime.** The 2008 crisis is in-sample, but 21 years is a small number for estimating tail behaviour. A prolonged bear market (1973–74, 2000–02) is not represented.
- **Survivorship bias is handled at the constituent level, not the index level.** The S&P 500 and NASDAQ 100 are themselves pre-selected for size and survival, so results do not generalise directly to the broader US equity universe.
- **Cross-offset alpha Sharpe is a stability statistic, not a temporal Information Ratio.** It measures how much alpha varies across different rebalance start-days — a robustness measure — and is distinct from the temporal IR a fund would report.
- **Single universe, single asset class.** Long-only US equities. No shorts, no fixed-income, no international, no commodities.

---

## Future work

- **Fundamental-data integration** (earnings, book value, cash flow) as first-class features.
- **Factor-return decomposition** to isolate how much of the alpha is distinct from known factors (value, momentum, low-vol, quality).
- **v2 architecture** incorporating Black–Litterman-style view integration, allowing calibrated confidence in fundamental vs technical signals.
- **International developed-market extension.**

---

## About

An independent quantitative research project by **Findlay Roberts**. BSc Economics, London School of Economics (2021). Built from scratch in Python — numpy, pandas, scipy, scikit-learn, LightGBM, Flask.

Contact via [LinkedIn](#) <!-- replace # with your LinkedIn profile URL before committing -->.
