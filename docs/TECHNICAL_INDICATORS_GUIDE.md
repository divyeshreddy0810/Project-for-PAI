# Technical Indicators Code Explanation

## Overview
The `technical_indicators.py` program is a **Stock Regime Predictor** that combines:
1. **Sentiment analysis** from news headlines (from sentiment_analyzer.py output)
2. **Technical indicators** from historical price data
3. **Machine learning-style scoring** to predict if a stock is in BULL, BEAR, or SIDEWAYS regime

---

## Code Breakdown by Section

### 1. **IMPORTS & SETUP** (Lines 1-30)

```python
import json, pandas, numpy, yfinance, csv, datetime, etc.
```

**What it does:**
- Loads required libraries for data processing, mathematical calculations, and API calls
- Tries to import `yfinance` to fetch real market price data
- If yfinance not available, it prints a warning but continues with sentiment-only predictions

**Key Variables:**
- `SENTIMENT_FILE` = Path to your `latest.json` sentiment output
- `OUTPUT_DIR` = Where to save results
- `RSI_OVERBOUGHT = 70` / `RSI_OVERSOLD = 30` = Technical thresholds
- `SCORE_WEIGHTS` = How much each factor contributes to final score

---

### 2. **TechnicalIndicatorCalculator CLASS** (Lines 48-125)

A utility class that calculates 6 different technical indicators from price data.

#### **A. Simple Moving Average (SMA)** 
```python
def calculate_sma(prices, period=20)
```
- Calculates average price over last 20 days
- Shows the **smoothed trend** of price movement
- Example: 20-day SMA tells you the average price over 3 weeks

#### **B. Exponential Moving Average (EMA)**
```python
def calculate_ema(prices, period=20)
```
- Like SMA but **gives more weight to recent prices**
- More responsive to recent price changes than SMA
- Used for faster trend detection

#### **C. Relative Strength Index (RSI)**
```python
def calculate_rsi(prices, period=14)
```
- **What it measures:** How much momentum the stock has (0-100 scale)
- **Calculation:** Compares gains vs. losses over 14 days
- **Interpretation:**
  - RSI > 70 = Overbought (price went up too fast, may drop)
  - RSI < 30 = Oversold (price went down too much, may bounce up)
  - RSI 40-60 = Neutral/Balanced

#### **D. MACD (Moving Average Convergence Divergence)**
```python
def calculate_macd(prices)
```
- **What it measures:** Momentum & trend direction
- **How:** Compares fast (12-day) vs slow (26-day) moving averages
- **Result:** Gives MACD line + Signal line + Histogram
- **Interpretation:**
  - MACD > Signal Line = Bullish (uptrend)
  - MACD < Signal Line = Bearish (downtrend)

#### **E. Bollinger Bands**
```python
def calculate_bollinger_bands(prices, period=20)
```
- **What it measures:** Price volatility (how much price fluctuates)
- **Result:** Upper band, middle (SMA), lower band
- **Interpretation:**
  - Price near upper band = Overbought
  - Price near lower band = Oversold
  - Wide bands = High volatility

#### **F. Average True Range (ATR)**
```python
def calculate_atr(high, low, close)
```
- **What it measures:** How much the price moves per day
- Shows **volatility level** (bigger number = more volatile)

---

### 3. **SentimentAnalyzer CLASS** (Lines 128-198)

Loads and extracts sentiment data from your `latest.json` file.

#### **A. Constructor**
```python
def __init__(self, filepath)
```
- Opens your sentiment JSON file
- Stores it in memory for quick access

#### **B. Extract Sentiment for One Symbol**
```python
def get_asset_sentiment(symbol="^GSPC")
```
**Returns a dictionary with:**
- `overall_mean` = Average sentiment score (-1 to +1)
- `overall_median` = Middle sentiment value
- `overall_std` = How spread out sentiments are
- `total_headlines` = How many news articles were analyzed
- `sentiment_trend` = Is sentiment improving (+) or worsening (-)?
- `daily_means` = Sentiment by day

**Example output:**
```python
{
  'overall_mean': 0.082,      # Slightly positive
  'sentiment_trend': +0.0045,  # Getting slightly more positive each day
  'total_headlines': 42        # 42 articles analyzed
}
```

#### **C. Calculate Trend**
```python
def _calculate_trend(daily_means)
```
- Takes sentiment scores from each day
- Calculates **slope** (how fast sentiment is changing)
- Positive slope = Sentiment improving
- Negative slope = Sentiment worsening
- Zero slope = Sentiment stable

---

### 4. **RegimePredictor CLASS** (Lines 201-450)

The **main intelligence** of the program. Combines sentiment + technical data to predict stock regime.

#### **A. Initialize Predictor**
```python
def __init__(self, sentiment_file)
```
- Loads sentiment data
- Prepares technical calculator
- Stores analysis metadata (date range being analyzed)

#### **B. Fetch Price Data**
```python
def fetch_price_data(symbol, period='1mo')
```
**What it does:**
1. Converts sentiment window (like '1mo') to yfinance period
2. Downloads historical OHLCV data (Open, High, Low, Close, Volume)
3. Returns a DataFrame with ~20+ rows of price history
4. Prints status messages

**Example:**
```python
fetch_price_data('^GSPC', '1mo')
# Downloads 1 month of S&P 500 daily prices
# Returns DataFrame with 20 rows of OHLCV data
```

#### **C. Calculate All Indicators**
```python
def calculate_indicators(df)
```
**What it does:**
1. Calls all 6 indicator functions from `TechnicalIndicatorCalculator`
2. Returns dictionary with all calculated indicators:
   ```python
   {
     'sma_20': [series of values],
     'sma_50': [series of values],
     'rsi': [series of values],
     'macd': [series of values],
     'bollinger_bands': [upper, middle, lower],
     ...
   }
   ```

#### **D. Score Regime (THE CORE LOGIC)**
```python
def score_regime(sentiment, indicators, price_data)
```

**This is where the magic happens.** Scores each factor and combines them.

**Step 1: Helper Function for Safe Float Conversion**
```python
def safe_float(value):
    try:
        v = float(value)
        return v if not np.isnan(v) else None
    except:
        return None
```
- Safely converts pandas values to Python floats
- Returns None if value is missing/invalid

**Step 2-7: Score Six Components**

##### **Component 1: Price Trend (Weight: 1.5)**
```python
if close > sma_20 and sma_20 > sma_50:
    trend_score = 3.0  # Strong uptrend
elif close > sma_20:
    trend_score = 1.0  # Weak uptrend
elif close < sma_20:
    trend_score = -1.0  # Downtrend
```
**Logic:** If price is above moving averages → bullish, if below → bearish

##### **Component 2: RSI Score (Weight: 1.0)**
```python
if rsi > 70:
    rsi_score = -2.0   # Overbought - bearish
elif rsi > 60:
    rsi_score = 1.0    # Strong - bullish
elif 40 < rsi < 60:
    rsi_score = 0.5    # Neutral
elif rsi < 30:
    rsi_score = 2.0    # Oversold - potential bounce
```
**Logic:** Extreme values signal strength (or weakness)

##### **Component 3: MACD Score (Weight: 1.0)**
```python
if macd > signal and macd > 0:
    macd_score = 2.0   # Strong bullish
elif macd > signal:
    macd_score = 1.0   # Weak bullish
elif macd < signal:
    macd_score = -1.0  # Bearish
```
**Logic:** MACD > Signal = uptrend, MACD < Signal = downtrend

##### **Component 4: Sentiment Level (Weight: 1.5)**
```python
if sentiment_mean > 0.15:
    sentiment_score = 2.0   # Very positive
elif sentiment_mean > 0.05:
    sentiment_score = 1.0   # Positive
elif sentiment_mean < -0.15:
    sentiment_score = -2.0  # Very negative
```
**Logic:** Higher sentiment values = more bullish

##### **Component 5: Sentiment Trend (Weight: 1.0)**
```python
if sentiment_trend > 0.05:
    trend_score = 1.5   # Improving
elif sentiment_trend < 0:
    trend_score = -1.0  # Worsening
```
**Logic:** Is investor sentiment getting better or worse?

##### **Component 6: Headline Volume (Weight: 0.5)**
```python
if headline_count > 50:
    volume_score = 1.0   # High engagement
elif headline_count < 20:
    volume_score = -0.5  # Low engagement
```
**Logic:** More headlines = more investor attention/interest

**Step 8: Combine Scores**
```python
total_score = sum of (component_score × weight)
```
**Example:**
- Price trend: 3.0 × 1.5 = 4.5
- RSI: 1.0 × 1.0 = 1.0
- MACD: 1.0 × 1.0 = 1.0
- Sentiment level: 2.0 × 1.5 = 3.0
- Sentiment trend: 0.5 × 1.0 = 0.5
- Headline volume: 0.5 × 0.5 = 0.25
- **TOTAL SCORE: 10.25/10** (max possible)

#### **E. Determine Regime**
```python
def determine_regime(total_score)
```
**Mapping:**
- Score ≥ 3.0 → **🔼 BULL** (uptrend likely)
- Score ≤ -3.0 → **🔽 BEAR** (downtrend likely)
- -3.0 < Score < 3.0 → **↔️ SIDEWAYS** (no clear direction)

**Confidence Calculation:**
```python
confidence = min(abs(total_score) / 10, 0.99)
```
- Score of 10 = 99% confidence
- Score of 3 = 30% confidence
- Score of 0 = 0% confidence

#### **F. Full Prediction Pipeline**
```python
def predict(symbol, period=None)
```
**What it does (in order):**
1. Gets sentiment for the symbol
2. Fetches price data for that symbol
3. Calculates all technical indicators
4. Scores each component
5. Determines regime + confidence
6. Returns comprehensive result dictionary

---

### 5. **Result Formatting** (Lines 453-475)

#### **A. Console Display**
```python
def format_result(result)
```
Converts result dictionary to human-readable string with:
- Symbol & name
- Regime emoji + score
- Sentiment breakdown
- Technical indicator values
- Component score breakdown with visual bars
- Analysis date range

#### **B. CSV Export**
```python
def save_results_csv(results, filename)
```
Saves summary table to CSV file with columns:
- Symbol, Name, Regime, Confidence, Score
- Sentiment_Mean, Sentiment_Trend, RSI, MACD, Headlines

---

### 6. **Main Execution** (Lines 503-575)

#### **Step 1: Program Header**
```python
print("🚀 Stock Regime Predictor with Technical Indicators")
```

#### **Step 2: Load Sentiment Data**
```python
predictor = RegimePredictor(SENTIMENT_FILE)
```
- Loads your `latest.json` sentiment file

#### **Step 3: Display Available Symbols**
```python
print("Available symbols from sentiment analysis:")
for i, symbol in enumerate(symbols, 1):
    print(f"  {i}. {symbol} - {name}")
```
Shows numbered list of assets you can analyze

#### **Step 4: Get User Input**
```python
user_input = input("Enter symbols or indices: ")
```
Accepts:
- Numbers: `1,2,3` or `1-3` (converts to symbols)
- Symbols: `^GSPC,BTC-USD` (direct)
- Commands: `all`, `q` (quit)

#### **Step 5: Run Predictions**
```python
for symbol in selected_symbols:
    result = predictor.predict(symbol)
    print(format_result(result))
    results.append(result)
```
- Predicts regime for each selected symbol
- Prints formatted results

#### **Step 6: Save to CSV**
```python
save_results_csv(results, filename)
```
Saves all results to timestamped CSV file

---

## Complete Data Flow Diagram

```
Input: latest.json (sentiment data)
  ↓
SentimentAnalyzer
  ├── Extract sentiment metrics (mean, std, trend)
  └── Store daily breakdown
      ↓
User selects symbol(s)
  ↓
RegimePredictor.predict()
  ├── Get sentiment for symbol
  ├── Fetch price data (yfinance)
  └── Calculate indicators
      ├── SMA (moving averages)
      ├── RSI (momentum)
      ├── MACD (trend)
      ├── Bollinger Bands (volatility)
      └── ATR (price movement)
          ↓
      Score Regime
      ├── Price trend score → weight 1.5
      ├── RSI score → weight 1.0
      ├── MACD score → weight 1.0
      ├── Sentiment level → weight 1.5
      ├── Sentiment trend → weight 1.0
      └── Headline volume → weight 0.5
          ↓
      Combine scores → total_score (-10 to +10)
          ↓
      Map to regime
      ├── total_score ≥ 3.0 → BULL 🔼
      ├── -3.0 < score < 3.0 → SIDEWAYS ↔️
      └── total_score ≤ -3.0 → BEAR 🔽
          ↓
Output: Console display + CSV file
```

---

## Key Concepts Explained

### **Regime vs Sentiment**
- **Sentiment** = What people THINK about the stock (from news)
- **Technical Regime** = What the PRICE is actually doing (from price data)
- **Combined** = Both perspectives for balanced prediction

### **Why Weight Things Differently?**
- **Price Trend (1.5)** = Most reliable indicator
- **Sentiment Level (1.5)** = News reflects investor opinion
- **RSI, MACD (1.0 each)** = Good for momentum confirmation
- **Headline Volume (0.5)** = Just shows interest level

### **Why Safe Float Conversion?**
- Pandas returns special objects that don't play nicely with comparisons
- `np.isnan()` on pandas Series caused errors
- Converting to Python float first avoids the ambiguity

### **Confidence Score**
- Not just predicting direction, but also **certainty**
- Score of 9 = 90% confident it's BULL
- Score of 3 = 30% confident (weakly BULL)
- Helps you decide "should I trust this prediction?"

---

## Example Walkthrough

**Scenario:** Analyzing S&P 500 (^GSPC)

1. **Load Sentiment**
   - Mean: +0.082 (slightly positive)
   - Trend: +0.0045 (improving)
   - Headlines: 42

2. **Fetch Price Data**
   - Last close: $5,834
   - SMA20: $5,800 (below price → good)
   - SMA50: $5,770 (below SMA20 → good)

3. **Calculate Indicators**
   - RSI: 58 (neutral, not overbought)
   - MACD: 0.023 > Signal 0.019 (bullish)

4. **Score Components**
   - Price trend: 3.0 (above all SMAs) × 1.5 = 4.5
   - RSI: 0.5 (neutral) × 1.0 = 0.5
   - MACD: 2.0 (strong bullish) × 1.0 = 2.0
   - Sentiment level: 1.0 (positive) × 1.5 = 1.5
   - Sentiment trend: 0.5 (improving) × 1.0 = 0.5
   - Headline volume: 0.5 (decent) × 0.5 = 0.25

5. **Total Score:** 4.5 + 0.5 + 2.0 + 1.5 + 0.5 + 0.25 = **9.25**

6. **Result:** 🔼 **BULL** with 92% confidence!

---

## Common Questions

**Q: What if yfinance is not installed?**
A: The program still works but uses sentiment-only scoring, which is less accurate.

**Q: Why do some indicators show NaN (Not a Number)?**
A: If you don't have enough price history (e.g., less than 20 days for SMA20), those indicators can't be calculated. The code handles this gracefully.

**Q: Can I change the scoring weights?**
A: Yes! Modify `SCORE_WEIGHTS` at the top of the file. For example, if you trust price action more than sentiment, increase `'price_trend'` weight.

**Q: What's the difference between this and just looking at price?**
A: This combines multiple perspectives:
- Price tells you WHAT happened
- Sentiment tells you WHY it happened
- Regime tells you WHAT'S LIKELY TO HAPPEN NEXT

---
