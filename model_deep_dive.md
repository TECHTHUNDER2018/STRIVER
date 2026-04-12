# GVC Intelligence Engine — Complete Technical Deep Dive

> Everything from first principles to every internal formula.

---

## Part 1: What Is This System Doing at the Highest Level?

The system answers one question: **"Will this stock go UP, STABLE, or DOWN in the next 1 hour / 4 hours / 24 hours?"**

It does this by combining **5 completely different types of evidence** — market price data, social media text, statistical causality tests, company fundamentals, and macro market context — and feeding all of it into **three machine learning models** that were trained on **191,435 real historical price windows** across **38 stocks**.

The output is not a simple "UP" — it is:
- A **probability distribution**: P(UP)=42%, P(STABLE)=41%, P(DOWN)=17%
- A **direction** decided by thresholding those probabilities
- A **confidence score** [0–95] representing how trustworthy the signal is
- A **price range** (P10, P50, P90) using Monte Carlo simulation
- A **SHAP explanation** of which features drove the decision

---

## Part 2: Machine Learning — From Basics

### What is a Decision Tree?

A decision tree is a series of yes/no questions:
```
Is RSI > 70?
  YES → Is MACD histogram < 0?
           YES → P(DOWN) = 68%
           NO  → P(STABLE) = 55%
  NO  → Is OBV slope > 0?
           YES → P(UP) = 61%
```
Each split is chosen by the algorithm to maximize **information gain** — i.e., the split that best separates UP from STABLE from DOWN.

### What is XGBoost?

XGBoost (**eXtreme Gradient Boosted trees**) builds **600 decision trees one after another**, where each new tree specifically tries to correct the mistakes of all previous trees.

```
Tree 1:  Initial guess → mostly STABLE (safe default)
Tree 2:  Focus on rows Tree 1 got wrong
Tree 3:  Focus on rows Tree 1+2 got wrong
...
Tree 600: Fine corrections
Final:   Sum of all tree outputs → probabilities
```

Hyperparameters used in this system:
```python
n_estimators    = 600      # 600 trees
max_depth       = 5        # max 5 yes/no splits per tree
learning_rate   = 0.030    # small steps = better generalization
subsample       = 0.75     # each tree sees 75% of data (prevents overfit)
colsample_bytree= 0.70     # each tree sees 70% of features
min_child_weight= 4        # minimum samples in a leaf
gamma           = 0.08     # minimum improvement needed to create a split
reg_alpha       = 0.12     # L1 regularization (sparsity)
reg_lambda      = 1.0      # L2 regularization (weight reduction)
early_stopping  = 40       # stop if no improvement for 40 rounds
```

### What is RandomForest?

RandomForest builds **400 trees independently and in parallel** (not sequentially). Each tree is trained on a **random bootstrap sample** of the data, and each split picks from a **random subset of features** (`sqrt(36) ≈ 6`). The final answer is a majority vote / probability average.

RF is less powerful than XGBoost but more **stable and diverse** — it rarely makes the same mistakes as XGBoost, which is exactly why we combine them.

```python
n_estimators  = 400
max_depth     = 9
min_samples_leaf = 4
max_features  = "sqrt"     # ~6 features per split
class_weight  = "balanced" # handle class imbalance
```

### What is LightGBM?

LightGBM is another gradient boosting framework, but it grows trees **leaf-first** (not level-first like XGBoost). This means it can grow deeper trees more efficiently and handles large feature spaces better.

```python
n_estimators   = 600
num_leaves     = 47        # max leaves in any one tree
max_depth      = 7
learning_rate  = 0.025
subsample      = 0.75
colsample_bytree = 0.70
min_child_samples = 10
```

### Why Three Models Together?

Each model has different strengths and biases:
- XGBoost: great at finding sharp decision boundaries, good at rare events
- RandomForest: stable, resistant to outliers, high variance in feature selection
- LightGBM: fast, handles feature interactions well, leaf-wise precision

**Ensemble soft-vote:**
```
Final_P(UP) = 0.50 × XGB_P(UP) + 0.25 × RF_P(UP) + 0.25 × LGB_P(UP)
```
This is called **soft voting** — we average the probabilities, not the class labels. The weights 50:25:25 reflect XGBoost's generally higher accuracy on financial time series.

---

## Part 3: Training Data — How the Models Learned

### What data was used?

```
38 tickers × (1 year + 2 years) of hourly OHLCV data
Downloaded from yfinance at 1-hour intervals
1y period  → ~1,700 bars per ticker (1h, 4h, 24h labels)
2y period  → ~3,400 bars per ticker (24h labels only — longer horizon)
Total: 191,435 training samples
```

### Ticker Universe
```
US Large Cap: AAPL, NVDA, MSFT, TSLA, AMZN, GOOGL, META, AMD, PLTR, UBER, MU, RIVN
US Finance:   JPM, GS
US Healthcare: JNJ, PFE
US Energy:    XOM, BA
US Media:     NFLX, SNAP, COIN
US ETFs:      SPY, QQQ, IWM, TLT, GLD
NSE India:    RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS, SBIN.NS, ADANIENT.NS, WIPRO.NS
Global ETFs:  EWJ (Japan), EWZ (Brazil), FXI (China), EFA (MSCI EAFE)
```

Why so many different stocks? Because each market regime — US tech volatility, Indian market seasonality, EM political risk — teaches the model different patterns. A model trained only on AAPL would fail badly on RELIANCE.NS.

### How are the labels created?

For every row (every 1-hour bar), we look forward in time and compute:
```python
pct_1h  = (close[i+1]  - close[i]) / close[i]
pct_4h  = (close[i+4]  - close[i]) / close[i]
pct_24h = (close[i+24] - close[i]) / close[i]
```

Then label using **ATR-scaled dynamic thresholds** (not fixed percentages):
```python
# Base threshold for 24h:
base_up  = +0.012   # +1.2%
base_dn  = -0.012   # -1.2%

# Dynamic threshold = max(base, atr_pct × multiplier)
# ATR_multiplier for 24h = 1.20
median_atr_pct = ATR14 / close  (rolling 20-bar median)
dyn_threshold  = max(0.012, median_atr_pct × 1.20)
```

**Why ATR-scaled?** A 1.2% move for a stable stock like JNJ is significant. For TSLA, which moves 3-4% routinely, 1.2% is just noise. By scaling to each stock's actual volatility, the UP/DOWN labels carry the same *statistical significance* across all tickers.

```
UP    : pct_return > +dyn_threshold
DOWN  : pct_return < -dyn_threshold
STABLE: everything in between
```

Resulting class distribution:
```
24h: DOWN=54,606 | STABLE=70,436 | UP=66,393   (relatively balanced)
1h:  DOWN=16,088 | STABLE=158,702 | UP=16,645  (heavily STABLE — hourly moves are tiny)
```

### How is the data split?

**Strictly chronological 80/20 split — no shuffling:**
```
First 80% of bars → Training set
Last 20% of bars  → Test set
```
This is critical. Random shuffling would cause **data leakage** — the model would see future bars during training, making accuracy look artificially high.

### Class Imbalance Handling

At 1h/4h timeframes, 83%+ of bars are STABLE. Without correction, the model would just always predict STABLE and get 83% accuracy while being useless.

**Fix: sample weights**
```python
from sklearn.utils.class_weight import compute_sample_weight
sw = compute_sample_weight(class_weight="balanced", y=y_train)
```
This assigns higher training weight to the rare UP and DOWN classes, forcing the model to pay equal attention to all three.

---

## Part 4: The 36 Features — Exact Math

Every prediction uses these 36 numbers, computed identically at training time (from raw OHLCV bars) and at inference time (from the current 5d×1h price window + live posts).

All values are normalized to approximately **[-1, +1]** or **[0, 1]** so no single feature dominates.

### Features 1–3: Social Signal Proxies

**Feature 1: sentiment_score**
```
Training:  5-bar price momentum × 10, clipped to [-1, +1]
           mom5 = (close[i] - close[i-5]) / close[i-5]
           f1 = clip(mom5 × 10, -1, +1)

Inference: Weighted average of all post FinBERT scores
           Each post: score × authority_weight × time_decay × sarcasm_mult × fear_damp
```

**Feature 2: post_count**
```
Training:  Volume z-score proxy → (volume - vol20) / vol20
           Scaled: clip(vol_z × 6 + 10, 0, 20) / 20

Inference: len(signal_posts) / 100, clipped to [0, 1]
```

**Feature 3: agreement_ratio**
```
Training:  5-bar directional consistency
           dir[i] = sign(close[i] - close[i-1])
           agr5 = |rolling_mean(dir, 5)|  → 0.0 means random, 1.0 means all same direction

Inference: Fraction of posts pointing same direction
           signs = [+1 if score > 0 else -1 for each post]
           agreement = count(dominant_sign) / total_posts
```

### Features 4–9: Classic Technical Indicators

**Feature 4: RSI (Relative Strength Index)**
```
delta = close.diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
rsi   = 100 - (100 / (1 + gain/loss))

Normalized: (rsi - 50) / 50   → range [-1, +1]
            RSI=70 → f4=+0.40 (overbought)
            RSI=30 → f4=−0.40 (oversold)
```

**Feature 5: MACD Histogram**
```
ema12 = close.ewm(span=12).mean()
ema26 = close.ewm(span=26).mean()
macd_line   = ema12 - ema26
signal_line = macd_line.ewm(span=9).mean()
histogram   = macd_line - signal_line

Normalized: clip(histogram × 50, -1, +1)
```

**Feature 6: Bollinger %B**
```
BB_middle = close.rolling(20).mean()
BB_std    = close.rolling(20).std()
BB_upper  = BB_middle + 2 × BB_std
BB_lower  = BB_middle − 2 × BB_std
pct_b     = (close - BB_lower) / (BB_upper - BB_lower)  → [0, 1]

Normalized: (pct_b - 0.5) × 2  → [-1, +1]
            pct_b=1.0 → f6=+1.0 (above upper band = overbought)
            pct_b=0.0 → f6=−1.0 (below lower band = oversold)
```

**Feature 7: Stochastic %K**
```
lo14 = low.rolling(14).min()
hi14 = high.rolling(14).max()
stoch_k = 100 × (close - lo14) / (hi14 - lo14)

Normalized: (stoch_k - 50) / 50  → [-1, +1]
```

**Feature 8: Volume Ratio**
```
vol5  = volume.rolling(5).mean()
vol20 = volume.rolling(20).mean()
vol_ratio = vol5 / vol20

Normalized: clip(vol_ratio - 1.0, -1, +1)
            vol_ratio=2.0 → f8=+1.0 (twice normal volume = high interest)
```

**Feature 9: MA Signal**
```
ma10 = close.rolling(10).mean()
ma50 = close.rolling(50).mean()

if close > ma10 AND ma10 > ma50: → +1.0 (full bullish alignment)
if close > ma10 AND ma10 ≤ ma50: →  0.0 (mixed)
if close < ma10 AND ma10 < ma50: → -1.0 (full bearish alignment)
```

### Features 10–11: Fundamentals

**Feature 10: fundamentals_score**
```
From yfinance: P/E ratio, EPS growth, profit margin, revenue growth
scored_signals computed per metric:
  - P/E < 15: +0.2 (undervalued)
  - EPS growing > 10%: +0.1
  - Profit margin > 20%: +0.1
  - Revenue declining: -0.2
Final: sum of signals, clipped to [-1, +1]
```

**Feature 11: beta**
```
Beta = yfinance market beta (capped to [0, 3])
Normalized: beta / 3.0   → [0, 1]
Beta=1.0 → f11=0.33  (moves with market)
Beta=2.5 → f11=0.83  (amplified market moves)
```

### Features 12–18: Derived Intelligence

| # | Feature | Computation |
|---|---|---|
| 12 | news_sentiment | 24-bar momentum proxy: clip(mom24 × 8, -1, +1) |
| 13 | momentum_score | 5-bar composite: clip(mom5 × 8, -1, +1) |
| 14 | causality_score | Rolling 10-bar price-volume Pearson correlation |
| 15 | ultimatum_score | 0.0 at training; NLP price-target detection at inference |
| 16 | entity_count | 0.5 proxy at training; real entity count at inference |
| 17 | risk_tier_enc | vol20r>0.03→1.0, >0.02→0.67, >0.01→0.33, else 0.0 |
| 18 | noise_score | clip(1 - vol20r/0.03, 0, 1) — high vol = noisy |

### Features 19–22: Cyclical Time Encoding

**Why encode time cyclically?**
If you encode hour as a raw number (0–23), the model thinks hour 23 is "far" from hour 0, but they're adjacent in time. Sine/cosine encoding solves this:
```
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
dow_sin  = sin(2π × weekday / 5)   # weekday 0=Mon, 4=Fri
dow_cos  = cos(2π × weekday / 5)
```
This captures **market open/close patterns** and **Monday effect** without discontinuity.

### Features 23–26: Price History (v5)

**Feature 23: ret_1h — last-bar return**
```
ret_1h = (close[-1] - close[-2]) / close[-2]
Normalized: clip(ret_1h × 30, -1, +1)
            +3.3% move → f23 = +1.0
```

**Feature 24: ret_4h — 4-bar return**
```
ret_4h = (close[-1] - close[-5]) / close[-5]
Normalized: clip(ret_4h × 15, -1, +1)
```

**Feature 25: ATR norm — volatility regime**
```
True Range per bar = max(high-low, |high-prev_close|, |low-prev_close|)
ATR14 = Wilder's smoothed average of last 14 True Ranges
         atr[t] = atr[t-1] × 13/14 + TR[t] × 1/14

atr_norm = ATR14 / close
Normalized: clip(atr_norm / 0.04, 0, 1)
            1% hourly ATR → f25=0.25
            4% hourly ATR → f25=1.00 (extremely volatile)
```

**Feature 26: spread_norm — bar width**
```
spread = (high - low) / close
Normalized: clip(spread / 0.03, 0, 1)
```

### Features 27–36: v7 Advanced Signals

**Feature 27: Williams %R**
```
hi14 = high.rolling(14).max()
lo14 = low.rolling(14).min()
W%R  = -100 × (hi14 - close) / (hi14 - lo14)
       Range: [-100, 0]

Normalized: W%R / 50 + 1  → [-1, +1]
W%R = -20 → f27 = +0.60 (near recent high = overbought)
W%R = -80 → f27 = -0.60 (near recent low = oversold)

Unlike RSI (close-based), W%R is high-anchored — it measures
how close you are to the TOP of the recent range, not the middle.
```

**Feature 28: OBV (On-Balance Volume) Slope**
```
OBV accumulates volume directionally:
  OBV[t] = OBV[t-1] + volume[t]  if close > prev_close
  OBV[t] = OBV[t-1] - volume[t]  if close < prev_close

5-bar slope = (OBV[t] - OBV[t-5]) / (5 × mean_volume_20)
Normalized: clip(slope / 3, -1, +1)

Positive OBV slope = volume flowing IN = buyers accumulating
This is the ONLY feature that directly measures buyer vs seller pressure
```

**Feature 29: VWAP Deviation**
```
VWAP(24) = sum(close × volume, last 24 bars) / sum(volume, last 24 bars)

vwap_deviation = (close - VWAP) / ATR
Normalized: clip(deviation / 3, -1, +1)

Why important: Institutional traders target VWAP for entries/exits.
Price > VWAP means buyers are dominant → bullish signal
Price < VWAP means sellers are dominant → bearish signal
Distance from VWAP measured in ATR units = volatility-adjusted signal
```

**Feature 30: ROC-10 (Rate of Change)**
```
ROC10 = (close[t] - close[t-10]) / close[t-10]
Normalized: clip(ROC10 / 0.20, -1, +1)
20% move → f30 = ±1.0

Measures MEDIUM-TERM momentum (10 hours) — different from:
- ret_1h (feature 23): last hour  
- ret_4h (feature 24): last 4 hours
- mom5 (feature 1): sentiment 5-bar proxy
```

**Feature 31: Vol Regime Ratio**
```
vol5  = std(hourly_returns, last 5 bars)
vol20 = std(hourly_returns, last 20 bars)
ratio = vol5 / vol20

Normalized: clip(ratio / 3, 0, 1)

ratio > 1.0: volatility EXPANDING (possibly breaking out)
ratio < 1.0: volatility COMPRESSING (possible breakout pending)
ratio ≈ 0.3: extreme compression → Bollinger Squeeze setup
```

**Feature 32: 52-Week High Distance**
```
hi252 = close.rolling(252).max()   (252 hourly bars ≈ 10 trading days ≈ 2 weeks
                                     at 1h bars, 252 = ~3 trading months)
distance = (close - hi252) / hi252  → always ≤ 0

clip to [-1, 0]
distance = 0.00 → at the 52w high (psychological resistance)
distance = -0.20 → 20% below the high (room to run)
```

**Feature 33: Gap Signal**
```
gap = (open[t] - close[t-1]) / ATR

Normalized: clip(gap / 2, -1, +1)

Positive gap = opened higher than yesterday's close → overnight bullish catalyst
Negative gap = gapped down → overnight negative news absorbed
ATR normalization: same gap size matters differently for low- vs high-vol stocks
```

**Feature 34: EMA8/21 Cross**
```
ema8  = close.ewm(span=8).mean()
ema21 = close.ewm(span=21).mean()
cross = (ema8 - ema21) / ATR

Normalized: clip(cross / 2, -1, +1)

EMA8 > EMA21 → bullish cross → positive feature value
EMA8 < EMA21 → bearish cross → negative feature value
Compared to MA10/50 (features 9), this reacts much faster to trend changes
```

**Feature 35: Sentiment Delta (Velocity)**
```
Training proxy: diff of 5-bar momentum, clipped [-0.10, +0.10] / 0.10
Inference:      sentiment_score_now - sentiment_score_last_call

Why velocity matters: A sentiment score of 0.4 could mean:
  - Strong stable bullish signal (good)
  - Rapidly fading from 0.9 (about to reverse — bad)
  The DELTA tells you which it is.
```

**Feature 36: Source Diversity**
```
Training proxy: 0.5 (constant — models learns a weak weight)
Inference:      len(distinct_platforms) / 4, clipped to [0, 1]
Platforms:      reddit, google_news, stocktwits, yahoo, gdelt, alt-news

diversity = 1.0 → signal from all 4+ platforms = more reliable
diversity = 0.25 → only 1 platform → could be isolated noise
```

---

## Part 5: The Training Pipeline — Exact Steps

```
Step 1: Download 38 tickers × (1y + 2y) from yfinance
        (12 parallel threads, CSV disk cache with 12h expiry)

Step 2: For each ticker, compute all 36 features as pandas Series
        (vectorized operations — entire history at once)

Step 3: For each bar i from 50 to n-25:
          build feature row (36 floats at index i)
          compute label: look at close[i+1], close[i+4], close[i+24]
          if pct > dyn_threshold → "UP"
          if pct < -dyn_threshold → "DOWN"
          else → "STABLE"

Step 4: Stack all rows: X=shape(191435, 36), y_1h, y_4h, y_24h

Step 5: Chronological 80/20 split (split_idx = int(191435 × 0.80))

Step 6: For each timeframe (1h, 4h, 24h):
          Compute balanced sample weights
          Train XGBoost (600 trees, early stopping at round 40)
          Train RandomForest (400 trees)
          Train LightGBM  (600 leaves, early stopping at round 40)
          Soft-vote ensemble: 50% XGB + 25% RF + 25% LGB
          Isotonic calibration (see Part 6)
          Save to disk: model_1h.pkl, model_1h_rf.pkl, model_1h_lgb.pkl, calibrator_1h.pkl
```

---

## Part 6: Isotonic Calibration — Why and How

### The Problem

Raw XGBoost might output "P(UP)=0.87" when its actual hit rate is only 60%. This overconfidence is common in boosted trees. We need the probabilities to be **statistically honest**.

### What is Isotonic Regression?

Isotonic regression fits a **non-decreasing step function** from raw probabilities to observed frequencies. It's non-parametric — no assumptions about shape.

If the model says P(UP)=0.9 on 100 predictions, but UP only happened 65% of the time, isotonic regression corrects 0.9 → ~0.65.

### How it's applied (One-vs-Rest):

```python
for cls_idx in [0, 1, 2]:  # DOWN, STABLE, UP
    y_binary = (y_test == cls_idx).astype(int)   # binary: is this class or not?
    raw_probs = ensemble_probs[:, cls_idx]        # raw probability for this class

    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(raw_probs, y_binary)
    calibrated_probs[:, cls_idx] = iso.predict(raw_probs)

# Re-normalize: rows must sum to 1.0
calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
```

Three separate calibrators are fitted and saved (one per class per timeframe = 9 total).

### Effect:

```
Before: log-loss=1.43, acc=48%  (overconfident raw model)
After:  log-loss=1.01, acc=49%  (honest probabilities, similar accuracy)
```
The accuracy barely changes, but the probabilities are now reliable for decision-making.

---

## Part 7: Inference Pipeline — Real-Time Prediction

When you search for `PLTR`, this exact sequence runs:

### Step 1: Price Data Fetch
```python
stock_data = fetch_stock_data("PLTR", period="5d", interval="1h")
# Returns: [{"open": 82.1, "high": 83.4, "low": 81.8, "close": 82.9, "volume": 4821000, ...}, ...]
# ~120 bars × 5 days
```

### Step 2: Fundamentals
```python
info = yf.Ticker("PLTR").info
fundamentals = score_fundamentals(info)
# Returns: {"score": 0.12, "beta": 1.85, "signals": ["P/E 95x — rich valuation", ...]}
```

### Step 3: Post Ingestion (all 6 sources in parallel)
```python
await asyncio.gather(
    fetch_news_posts("PLTR", n=30),          # GDELT global news
    fetch_google_news("PLTR", n=30),         # Google News RSS
    fetch_stocktwits("PLTR", n=30),          # StockTwits + SeekingAlpha
    fetch_reddit_posts("PLTR", n=20),        # Reddit public API
    fetch_yahoo_finance_news("PLTR", n=20),  # Yahoo Finance news tab
    fetch_twitter_posts("PLTR", n=25),       # Alt-news: Finviz + Benzinga + SA RSS
)
# Deduplicate by first 80 chars of text
# Result: ~71 unique posts
```

### Step 4: FinBERT Sentiment on Each Post
```python
for post in raw_posts:
    # FinBERT (financial domain BERT, 110M parameters):
    result = finbert_pipeline(post["text"][:512])
    # → [{"label": "positive", "score": 0.87}, ...]

    # Convert to composite score:
    if label == "positive": composite = score        # +0.87
    if label == "negative": composite = -score       # -0.87
    if label == "neutral":  composite = 0.0

    # VADER fallback if FinBERT unavailable:
    vader_score = vader.polarity_scores(text)["compound"]  # [-1, +1]

    # Final composite: 70% FinBERT + 30% VADER (if both available)
    post["sentiment"] = {
        "composite_score": composite,
        "label": "POSITIVE" / "NEGATIVE" / "NEUTRAL",
        "emotion": classify_emotion(text),     # fear/greed/uncertainty/excitement
        "is_sarcastic": detect_sarcasm(text),
        "weight_multiplier": 0.5 if is_sarcastic else 1.0
    }
```

### Step 5: Authority Tagging
```python
for post in posts:
    post["authority"] = {
        "credibility_score": 0.45,  # baseline for unknown sources
        "type": "news",
        "name": "Reuters"
    }
# Known verified sources (CEO, analyst, institution) get 0.80–0.95
# Anonymous retail gets 0.30–0.50
```

### Step 6: Build the 36-Feature Vector
```python
# Technical features from price_history:
tech = compute_indicators(price_history)   
# Returns: {rsi: 48.2, macd: {...}, bollinger: {...}, ...}

# v7 features computed from price bars:
closes = [b["close"] for b in price_history]
highs  = [b["high"]  for b in price_history]
lows   = [b["low"]   for b in price_history]
opens  = [b["open"]  for b in price_history]
vols   = [b["volume"] for b in price_history]

# Williams %R:
hi14 = max(highs[-14:])
lo14 = min(lows[-14:])
wr = -100 × (hi14 - closes[-1]) / (hi14 - lo14)
williams_r = clip(wr/50 + 1, -1, +1)

# OBV slope:
obv = cumsum(volume × sign(close_change))
obv_slope = (obv[-1] - obv[-6]) / (5 × mean_volume)

# VWAP deviation:
vwap = sum(close[i]×vol[i] for last 24) / sum(vol for last 24)
vwap_deviation = (close[-1] - vwap) / ATR

# ... (all 10 v7 features computed similarly)

# Sentiment features from posts:
sentiment_score  = _weighted_sentiment(posts, "1h")   # time-decayed weighted avg
agreement_ratio  = count(dominant_sign) / len(posts)
source_diversity = len(distinct_platforms) / 4
# ...

# Build vector:
fv = build_feature_vector(tech, sentiment_score, ..., williams_r, obv_slope, ...)
# fv = [0.12, 0.71, 0.64, -0.04, 0.18, ..., 0.43, 0.22]  ← 36 floats
```

### Step 7: Granger Causality Test
```python
causality = compute_granger_causality(posts, price_history)
# Aligns post timestamps with price bars via ±12h window
# Builds two time series:
#   sentiment_ts: hourly average sentiment per matched bar
#   returns_ts:   hourly price return per matched bar
# Tests:
#   - Standard Granger: does sentiment at lag L predict returns at lag 0?
#   - Toda-Yamamoto: corrects for non-stationarity
#   - Transfer Entropy: non-linear information flow
#   - DTW: similarity with time warping
#   - Bootstrap CI: 200 iterations, 95% confidence bands
# Returns: {causality_score: 0.641, lag_hours: 2, p_value: 0.031, te_score: 0.275, ...}
```

### Step 8: ML Ensemble Inference
```python
X_np = np.array([fv], dtype=np.float32)       # shape (1, 36)
X_df = pd.DataFrame(X_np, columns=FEATURE_NAMES)  # named for LGB

# For each timeframe (1h, 4h, 24h):
xgb_probs = xgb_model.predict_proba(X_np)   # → [[P(DOWN), P(STABLE), P(UP)]]
xgb_probs = align_to_012_order(xgb_probs)   # ensure column order: DOWN=0, STABLE=1, UP=2
rf_probs  = rf_model.predict_proba(X_np)
lgb_probs = lgb_model.predict_proba(X_df)   # needs named DataFrame

# Soft vote:
raw_probs = 0.50 × xgb_probs + 0.25 × rf_probs + 0.25 × lgb_probs
# → [[0.17, 0.41, 0.42]] for DOWN/STABLE/UP

# Apply Isotonic calibration:
for cls in [0, 1, 2]:
    calibrated[:, cls] = calibrator[cls].predict(raw_probs[:, cls])
calibrated = calibrated / calibrated.sum()
# → [[0.15, 0.42, 0.43]]  ← honest probabilities
```

### Step 9: Ensemble Score and Direction
```python
# XGB score = P(UP) - P(DOWN)
xgb_score = prob_up - prob_down    # → +0.43 - 0.15 = +0.28

# LGB soft-vote at prediction time:
lgb_score = lgb_prob_up - lgb_prob_down
ensemble_score = xgb_score × 0.60 + lgb_score × 0.40

# Direction thresholding:
if ensemble_score > +0.12: direction = "UP"
if ensemble_score < -0.12: direction = "DOWN"
else:                       direction = "STABLE"

# Confidence (raw from XGBoost probabilities):
confidence = max(prob_up, prob_down, prob_stable) × (1 - entropy(probs)) × scaling
```

### Step 10: Confidence Adjustments (Applied in Order)

```
Base confidence:                  68  (from XGBoost max probability)
+ LightGBM agrees:               +8   (both models point same way)
+ Regime = TRENDING_UP:          +5   (HMM says trending market)
− Tech ↔ Sentiment conflict:     −15  (RSI=75 but sentiment negative)
− 24h high volatility daily>2.5%: −10  (uncertain noisy day)
+ All 3 timeframes agree:        +5   (1h, 4h, 24h all say UP)

Final confidence:                 61   (clipped to [20, 95])
```

### Step 11: Price Range Estimate
```python
# Base price change estimate:
tf_multiplier = {1h: 1.0, 4h: 2.0, 24h: 4.0}
magnitude = abs(ensemble_score) × beta × realized_vol × tf_multiplier × 100
# Example: 0.28 × 1.85 × 0.012 × 4.0 × 100 = +2.5%

# Monte Carlo uncertainty:
sigma = realized_vol × 100 × sqrt(tf_hours)   # for 24h: 0.012 × 100 × 4.9 = 5.9%
noise = np.random.normal(0, sigma, 1000)
P10 = -2.0%  (pessimistic 10th percentile)
P50 = +2.5%  (base estimate)
P90 = +7.2%  (optimistic 90th percentile)
```

---

## Part 8: Confidence Fusion Index (CFI)

The CFI is a composite meta-signal that answers: **How much should you trust this prediction?**

It uses **7 inputs** combined with a weighted additive formula (not multiplicative — multiplication would collapse to zero if any single signal is weak).

### The 7 Signals

```python
S1 (NLP Strength, w=0.22):
    # How strong and directional is the sentiment?
    vol_boost = log(n_posts + 1) / log(100)  # more posts = slightly higher weight
    nlp = abs(sentiment_score) × (0.6 + 0.4 × vol_boost)
    # Floor of 0.05 prevents zero-collapse

S2 (Credibility, w=0.20):
    # How authoritative are the sources on average?
    avg_cred = mean([post.authority.credibility_score for post in posts])
    agreement_amp = 0.7 + 0.3 × agreement_ratio
    credibility = avg_cred × agreement_amp

S3 (Causality, w=0.18):
    # Is sentiment statistically proven to cause price movement?
    if p_value < 0.01:  sig_mult = 1.4   # highly significant
    elif p_value < 0.05: sig_mult = 1.2  # significant
    elif p_value < 0.15: sig_mult = 0.9  # marginal
    else:                sig_mult = 0.5  # not significant
    causality = causality_score × sig_mult × (1 + te_score × 0.2)

S4 (Source Diversity, w=0.10):
    # Are signals coming from multiple independent platforms?
    distinct_platforms = {reddit, google_news, stocktwits, yahoo, gdelt, alt-news}
    diversity = len(distinct_platforms) / 5.0

S5 (Volume, w=0.10):
    # How many posts? More posts = higher confidence
    volume = log(n_posts + 1) / log(50)

S6 (Noise Ratio, w=0.10):
    # Is this a calm or noisy market? High vol = low CFI
    noise = 1 - realized_vol / 0.03   (clipped to [0, 1])

S7 (Agreement, w=0.10):
    # Do all posts point in the same direction?
    agreement = agreement_ratio
```

### CFI Formula

```python
CFI_raw = (0.22 × NLP        + 0.20 × Credibility + 0.18 × Causality +
           0.10 × Diversity  + 0.10 × Volume      + 0.10 × Noise     +
           0.10 × Agreement)

# Total weights = 1.00

# Confidence band: ±1σ across signal values
signal_std = std([NLP, Credibility, Causality, Diversity, Volume, Noise, Agreement])
CFI_low    = CFI_raw - signal_std
CFI_high   = CFI_raw + signal_std

# Bonuses:
regime_bonus    = +0.04 if market is TRENDING
consensus_bonus = +0.08 if all 3 timeframes predict same direction (non-STABLE)
bayesian_prior  = mean trust_tracker adjustments per source (data-grounded)

# Penalties:
bot_reduction      = bot_score × 0.25
conflict_reduction = n_conflicts × 0.05 (max 0.20)
anomaly_reduction  = 0.08 if any anomalous engagement spikes
thin_data_penalty  = 0.15 if n_posts < 3
hft_penalty        = 0.10 if bot_score > 0.70
trust_decay_penalty = low_trust_ratio × 0.12

total_penalty = sum of all penalties

CFI_final = clip(CFI_raw - total_penalty + bonuses, 0, 1) × 100
```

### CFI Grades

| Score | Grade | Tier |
|---|---|---|
| ≥ 90 | A+ | Very High |
| ≥ 75 | A | High |
| ≥ 60 | B | Moderate |
| ≥ 40 | C | Low |
| ≥ 20 | D | Very Low |
| < 20 | F | — |

---

## Part 9: SHAP Explainability

SHAP (SHapley Additive exPlanations) answers the question: **"Which of the 36 features pushed the model toward UP, and which pushed it toward DOWN?"**

### How SHAP Works (Simplified)

For each prediction, SHAP computes the **marginal contribution** of each feature by averaging the model output across all possible feature subsets — borrowed from cooperative game theory (Shapley values).

For a prediction of P(UP)=0.43:
```
Base rate:              P(UP) = 0.35  (average across training)
+ ema_cross:            +0.044  (EMA8 > EMA21 → bullish)
+ causality_score:      +0.031  (stat significant sentiment→price link)
+ ret_1h:               +0.028  (last hour was up)
+ vwap_deviation:       -0.018  (trading above VWAP → overbought risk)
+ williams_r:           +0.015  (not yet overbought)
...
= 0.43
```

### Implementation

```python
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(np.array([feature_vector]))
# → shape (3, 36): SHAP value per class per feature

# For UP direction:
shap_up = shap_values[2]  # index 2 = UP class
# zip with FEATURE_NAMES for human-readable output
```

### Temporal SHAP

The system also computes **rolling SHAP history** — tracking how feature contributions have changed across the last 5 predictions. This shows:
- Is sentiment_score becoming more or less important over time?
- Is the model shifting from technical to sentiment signals?

### Natural Language Summary

```python
top_features = sorted(shap_values, key=abs, reverse=True)[:3]
for feat, val in top_features:
    if val > 0:
        reasons.append(f"{feat} is strongly driving the UP signal (+{val:.3f})")
    else:
        reasons.append(f"{feat} is limiting confidence ({val:.3f})")
```

---

## Part 10: Bot Detection

Coordinated campaigns can flood a stock with fake positive posts. The bot detector prevents the model from being manipulated.

### Methods Used

**1. Temporal Burst Detection:**
```python
post_counts_by_hour = group_posts_by_hour(posts)
mean_per_hour = mean(post_counts_by_hour)
std_per_hour  = std(post_counts_by_hour)
z_scores = [(count - mean) / std for each hour]
burst_detected = any(z > 3.0)   # 3σ above normal = suspicious spike
severity = max(z - 3) / 10      # normalized severity [0, 1]
```

**2. Account Pattern Analysis:**
- Account age < 30 days: suspicious
- Following >> Followers: bot pattern
- Posting frequency > 20/hour: HFT posting bot

**3. OFT Proxy (Order Flow Toxicity):**
Based on bid/ask spread patterns detected in price data:
```python
IF spread is unusually wide AND volume spike occurs simultaneously:
    HFT activity suspected → hft_penalty += 0.10 to CFI
```

**Output:**
```python
{
    "bot_activity_score": 0.11,  # 0=clean, 1=fully bot
    "burst_detected": True,
    "burst_hours": [14],          # which hour had the spike
    "signal_validity": "VALID",   # VALID / SUSPICIOUS / INVALID
    "max_z_score": 3.5
}
```

---

## Part 11: Historical Pattern Matching

For every prediction, the system finds **real historical dates** where the market looked similar.

### Algorithm

```python
# Download 2 years of daily data for this ticker
hist = yf.Ticker(ticker).history(period="2y", interval="1d")

for each day i in history:
    # Compute 5-day realized volatility
    window_vol = std([daily returns for days i-4 to i])
    
    # Skip if volatility is too different from current
    if abs(window_vol - current_realized_vol) > 0.03:
        continue
    
    # Compute momentum similarity
    mom5 = (close[i] - close[i-4]) / close[i-4]
    
    # Similarity score (vol similarity + momentum alignment)
    similarity = 1.0 - (vol_diff × 8 + sentiment_diff × 2)
    similarity = clip(similarity, 0.55, 0.97)
    
    # Actual outcome: what happened the next day?
    pct_next = (close[i+1] - close[i]) / close[i]
    outcome = "BULLISH" if pct > +1% else ("BEARISH" if pct < -1% else "STABLE")
    
    matches.append(date, setup_label, similarity, outcome, pct_next)

# Return top 3 most similar windows
```

This grounds the prediction in **real market history** and shows the user how similar setups actually played out.

---

## Part 12: Macro Regime Overlay

### Live Market Data (updated every 10 minutes)

```python
# Live telemetry from yfinance:
vix   = yf.Ticker("^VIX").history("1d").Close.iloc[-1]
spy   = yf.Ticker("SPY").history("5d")
dxy   = yf.Ticker("DX-Y.NYB").history("5d")
tnx   = yf.Ticker("^TNX").history("1d").Close.iloc[-1]

# Tailwind scoring:
tailwind = 0
if vix < 18:    tailwind += 25  # calm market → positive
elif vix < 25:  tailwind += 10
elif vix > 35:  tailwind -= 30  # fear spike → negative

spy_ret = (spy.Close.iloc[-1] - spy.Close.iloc[-5]) / spy.Close.iloc[-5]
if spy_ret > 0.02:  tailwind += 25  # SPY up 2%+ in last 5d
elif spy_ret < -0.03: tailwind -= 25

# DXY: strong dollar = headwind for growth stocks
dxy_ret = (dxy.Close.iloc[-1] - dxy.Close.iloc[-5]) / dxy.Close.iloc[-5]
if dxy_ret > 0.01:  tailwind -= 15  # dollar strengthening

# 10Y yield: high yields = valuation headwind
if tnx > 5.0:  tailwind -= 20
elif tnx < 3.5: tailwind += 10
```

**Tailwind ∈ [-100, +100]** shifts confidence up or down based on the macro environment.

---

## Summary: Complete Data → Prediction Flow

```
yfinance 5d×1h bars                                    ┐
  → 36 features (43 math operations)                   │
  → XGBoost 600 trees                                  │ → P(UP/STABLE/DOWN)
  → RandomForest 400 trees                             │    per 1h, 4h, 24h
  → LightGBM 600 leaves                                │
  → Isotonic calibration                               ┘

6 live sources ~ 70 posts
  → FinBERT sentiment (per post)
  → Authority scoring
  → Time decay weighting
  → Sarcasm / hype dampening
  → Granger causality test                             ┐
      Granger + TY + TE + DTW + Bootstrap CI           │ → causality_score
                                                        ┘

Bot detection                                           → CFI penalty
Macro overlay (VIX/SPY/DXY/TNX)                       → tailwind adjustment
HMM regime detection                                   → confidence ±5pts
Entity graph propagation                               → peer ticker signals

                    ALL COMBINED INTO:
┌─────────────────────────────────────────────────────────┐
│ direction:    UP / STABLE / DOWN                        │
│ confidence:   68  (honest, after all adjustments)       │
│ score:        +0.28  (ensemble P(UP) - P(DOWN))         │
│ price range:  P10=-2.0%  P50=+2.5%  P90=+7.2%          │
│ CFI:          61.4 / 100  Grade B  Tier: Moderate       │
│ SHAP:         {ema_cross: +0.044, causality: +0.031, …} │
│ explanation:  "EMA8 crossed above EMA21. Granger…"      │
│ historical:   3 real matching dates from 2y of data     │
└─────────────────────────────────────────────────────────┘
```
