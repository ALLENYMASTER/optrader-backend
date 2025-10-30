# 🐋 Optrader Cloud Backend

Options whale scanner API with **complete feature set** from us_whale_web.py:

✅ **Expected Move Calculation** - Weekly options (7-10 days, 68% probability)  
✅ **Strategy Type Detection** - DIRECTIONAL vs VOLATILITY vs STRADDLE  
✅ **Enhanced Sentiment Analysis** - Weighted by whale score  
✅ **Multiple Strategy Recommendations** - Up to 5 ranked strategies  
✅ **Market Hours Smart Scheduling** - Pre/Post/During market only  
✅ **Parallel Scanning** - Fast multi-ticker analysis  

---

## 📁 Project Structure

```
optrader_backend/
│
├── main.py                          # FastAPI application entry point
├── requirements.txt                 # Python dependencies
├── render.yaml                      # Render deployment config
├── README.md                        # This file
│
├── api/
│   ├── __init__.py
│   └── schemas.py                   # Pydantic data models
│
├── models/
│   ├── __init__.py
│   ├── options_scanner.py           # Whale activity detection
│   └── market_analyzer.py           # Market-wide analysis
│
└── storage/
    ├── __init__.py
    ├── ticker_manager.py            # User ticker list management
    └── cache_manager.py             # Simple caching system
```

---

## 🚀 Quick Start (Local Development)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Locally

```bash
# Development mode (auto-reload)
uvicorn main:app --reload --port 8000

# Production mode
python main.py
```

### 3. Access API

- **API Root**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

## ☁️ Deploy to Render

### Method 1: Using render.yaml (Recommended)

1. **Push code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Connect to Render**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Render will auto-detect `render.yaml`
   - Click "Apply"

### Method 2: Manual Setup

1. **Create New Web Service**
   - Go to Render Dashboard
   - Click "New +" → "Web Service"
   - Connect your repository

2. **Configure Settings**
   - **Name**: `optrader-api`
   - **Environment**: `Python 3`
   - **Region**: `Oregon (US West)`
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free` (for testing)

3. **Environment Variables**
   - `PYTHON_VERSION`: `3.11.0`
   - `PORT`: `8000` (auto-set by Render)

4. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (3-5 minutes)
   - Your API will be available at: `https://optrader-api.onrender.com`

---

## 📡 API Endpoints

### Ticker Management

#### Get User's Tickers
```http
GET /api/tickers?user_id=default
```

**Response:**
```json
{
  "user_id": "default",
  "tickers": ["SPY", "QQQ", "NVDA", "TSLA"],
  "count": 4,
  "timestamp": "2024-01-15T10:30:00"
}
```

#### Add Ticker
```http
POST /api/tickers/add?user_id=default
Content-Type: application/json

{
  "symbol": "AMZN"
}
```

#### Remove Ticker
```http
POST /api/tickers/remove?user_id=default
Content-Type: application/json

{
  "symbol": "TSLA"
}
```

#### Replace Ticker List
```http
POST /api/tickers/set?user_id=default
Content-Type: application/json

["NVDA", "AMD", "INTC", "TSM"]
```

---

### Market Scanning

#### Scan Market
```http
GET /api/scan?user_id=default&force=false&parallel=true
```

**Parameters:**
- `user_id`: User identifier (default: "default")
- `force`: Bypass cache (default: false)
- `parallel`: Use parallel scanning (default: true)

**Response:**
```json
{
  "user_id": "default",
  "results": [
    {
      "ticker": "NVDA",
      "current_price": 505.48,
      "volume_ratio": 1.25,
      "market_cap": 1245000000000,
      "sentiment": "STRONG BULLISH",
      "sentiment_score": 72.5,
      "bullish_percent": 72.5,
      "bearish_percent": 27.5,
      "total_calls_count": 45,
      "total_puts_count": 12,
      "total_calls_value": 15500000,
      "total_puts_value": 5800000,
      "whale_score": 385.2,
      "volatility_plays": 8,
      "directional_bets": 49,
      "unusual_activity": [
        {
          "option_type": "CALL",
          "strike": 510.0,
          "expiration": "2024-02-16",
          "volume": 2500,
          "lastPrice": 12.50,
          "impliedVolatility": 0.35,
          "whale_score": 385.2,
          "strategy_type": "DIRECTIONAL",
          "likely_direction": "LIKELY BUY"
        }
      ],
      "expected_move": {
        "atm_strike": 505.0,
        "atm_straddle_price": 18.50,
        "expected_move_dollar": 18.50,
        "expected_move_pct": 3.66,
        "upper_target": 523.98,
        "lower_target": 486.98,
        "confidence": "68% (1σ)"
      },
      "recommended_strategies": [
        {
          "strategy": "BUY CALL DEBIT SPREAD",
          "confidence": "HIGH",
          "rationale": "Strong bullish sentiment (73%) with 49 directional bets",
          "suggested_strikes": "Buy $510 / Sell $515",
          "risk_level": "MEDIUM",
          "profit_target": "+3.7%",
          "stop_loss": "-50% of premium",
          "time_horizon": "1-4 weeks"
        },
        {
          "strategy": "FOLLOW THE WHALE - BUY CALL",
          "confidence": "MEDIUM-HIGH",
          "rationale": "Massive whale trade (Score: 385) at $510",
          "suggested_strikes": "$510 CALL exp 2024-02-16",
          "risk_level": "HIGH",
          "profit_target": "+0.9% to strike",
          "stop_loss": "-30-50% of premium",
          "time_horizon": "14 days"
        }
      ],
      "timestamp": "2024-11-15T14:30:00"
    }
  ],
  "scan_time": "2024-11-15T14:35:00",
  "total_scanned": 15,
  "total_active": 12,
  "from_cache": false
}
```

#### Get Ticker Detail
```http
GET /api/ticker/NVDA?force=false
```

#### Market Summary
```http
GET /api/market-summary?user_id=default&force=false
```

**Response:**
```json
{
  "regime": "BULL MARKET",
  "regime_icon": "📈",
  "bullish_percent": 58.5,
  "bearish_percent": 41.5,
  "total_tickers": 15,
  "active_tickers": 12,
  "top_opportunities": [
    {
      "ticker": "NVDA",
      "sentiment": "STRONG BULLISH",
      "sentiment_score": 72.5,
      "strategy": "BUY CALL DEBIT SPREAD",
      "confidence": "HIGH",
      "current_price": 505.48
    }
  ],
  "risk_level": "MEDIUM"
}
```

#### Market Status (New!)
```http
GET /api/market-status
```

**Response:**
```json
{
  "current_time": "2024-11-15T14:30:00-05:00",
  "current_time_et": "2024-11-15 14:30:00 EST",
  "is_market_day": true,
  "is_market_open": true,
  "is_pre_market": false,
  "is_post_market": false,
  "should_scan_now": true,
  "scan_reason": "Market is open",
  "next_scan_time": "2024-11-15T15:00:00-05:00",
  "next_scan_time_et": "2024-11-15 15:00:00 EST",
  "market_open_time": "09:30 ET",
  "market_close_time": "16:00 ET"
}
```

---

### Utilities

#### Clear Cache
```http
POST /api/cache/clear?user_id=default
```

#### Get Default Tickers
```http
GET /api/default-tickers
```

#### Health Check
```http
GET /health
```

---

## 🧪 Testing API

### Using cURL

```bash
# Get tickers
curl https://optrader-api.onrender.com/api/tickers

# Scan market
curl https://optrader-api.onrender.com/api/scan?force=true

# Add ticker
curl -X POST https://optrader-api.onrender.com/api/tickers/add \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AMZN"}'

# Market summary
curl https://optrader-api.onrender.com/api/market-summary
```

### Using Python

```python
import requests

BASE_URL = "https://optrader-api.onrender.com"

# Get tickers
response = requests.get(f"{BASE_URL}/api/tickers")
print(response.json())

# Scan market
response = requests.get(f"{BASE_URL}/api/scan?force=true")
print(response.json())

# Add ticker
response = requests.post(
    f"{BASE_URL}/api/tickers/add",
    json={"symbol": "AMZN"}
)
print(response.json())
```

---

## 📊 Cache Strategy

- **Scan Results**: 15 minutes TTL
- **Ticker Details**: 10 minutes TTL
- **Market Summary**: 15 minutes TTL
- **Force refresh**: Use `?force=true` parameter

---

## ⚙️ Configuration

### Environment Variables

```bash
# Python version
PYTHON_VERSION=3.11.0

# Server port (auto-set by Render)
PORT=8000
```

---

## 🔧 Troubleshooting

### Issue: Slow first request after inactivity

**Cause**: Render free tier spins down after 15 min inactivity

**Solutions:**
1. Use `force=true` on first request
2. Implement wake-up ping from iPhone app
3. Upgrade to paid Render plan (always-on)

### Issue: yfinance rate limiting

**Solution**: Implement request throttling:
```python
import time
time.sleep(0.5)  # Between requests
```

### Issue: Module not found

**Solution**: Create `__init__.py` files:
```bash
touch api/__init__.py
touch models/__init__.py
touch storage/__init__.py
```

---

## 📈 Performance Optimization

### Current Performance
- **Sequential scan**: ~30-60 seconds for 15 tickers
- **Parallel scan**: ~10-20 seconds for 15 tickers
- **Memory**: ~150MB
- **Cache hit rate**: ~80% (with 15min TTL)

### Optimization Tips
1. Use parallel scanning (`parallel=true`)
2. Implement Redis for distributed cache
3. Use PostgreSQL for persistent storage
4. Add request rate limiting
5. Implement background job queue

---

## 🚧 Future Enhancements

- [ ] Redis cache integration
- [ ] PostgreSQL for user data
- [ ] WebSocket support for real-time updates
- [ ] Machine learning models (XGBoost/RandomForest)
- [ ] Historical data tracking
- [ ] User authentication
- [ ] Rate limiting per user
- [ ] Background job queue (Celery)

---

## 📝 Notes

- **Free Tier Limits**: Render free tier spins down after 15 min inactivity
- **Cold Start**: First request after spin-down takes 20-30 seconds
- **Data Source**: yfinance (Yahoo Finance) - unofficial API
- **Rate Limits**: Be respectful with yfinance requests

---

## 📞 Support

For issues or questions:
- Check `/docs` for interactive API documentation
- View logs in Render dashboard
- Test endpoints at `/health`

---

## 📄 License

MIT License - Feel free to use and modify
