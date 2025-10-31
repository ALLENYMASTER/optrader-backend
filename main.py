"""
Optrader Backend - FastAPI Server with Enhanced Features
Railway-optimized with rate limiting, caching, and error handling
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, time
import yfinance as yf
import pandas as pd
import numpy as np
import json
import asyncio
import aiohttp
from contextlib import asynccontextmanager
import pytz
import redis
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from concurrent.futures import ThreadPoolExecutor
import os
import random
import time as time_module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://default:EfKXKBdmYeyvsDmqcVmurRzMooouZhHO@redis.railway.internal:6379")
CACHE_TTL = int(os.getenv("CACHE_TTL", "1800"))  # 30 minutes
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "3"))  # seconds
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))
ALPHA_VANTAGE_KEY = os.getenv("T4MFI6GH6275NKLK", "")

# Log configuration on startup
logger.info(f"🔧 Configuration:")
logger.info(f"  - Cache TTL: {CACHE_TTL}s")
logger.info(f"  - Rate Limit Delay: {RATE_LIMIT_DELAY}s")
logger.info(f"  - Max Workers: {MAX_WORKERS}")
logger.info(f"  - Redis URL: {REDIS_URL[:20]}..." if REDIS_URL else "  - Redis: Not configured")
logger.info(f"  - Alpha Vantage: {'Configured ✓' if ALPHA_VANTAGE_KEY else 'Not configured ✗'}")

SCAN_CACHE_KEY = "scan_results"

# Initialize Redis (optional - will work without it)
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=5)
    redis_client.ping()
    logger.info("✅ Redis connected")
except Exception as e:
    redis_client = None
    logger.warning(f"⚠️ Redis not available - using in-memory cache: {e}")

# In-memory cache fallback
memory_cache = {}

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# ==================== Data Models ====================

class TickerData(BaseModel):
    ticker: str
    current_price: float
    volume_ratio: float
    market_cap: float
    sentiment: str
    sentiment_score: float
    bullish_percent: float
    bearish_percent: float
    total_calls_count: int
    total_puts_count: int
    total_calls_value: float
    total_puts_value: float
    whale_score: float
    volatility_plays: int
    directional_bets: int
    unusual_activity: List[Dict]
    expected_move: Optional[Dict]
    recommended_strategies: List[Dict]
    timestamp: str

class ScanResponse(BaseModel):
    user_id: str = "default"
    results: List[TickerData]
    scan_time: str
    total_scanned: int
    total_active: int
    from_cache: bool

class MarketStatus(BaseModel):
    current_time_et: str
    is_market_day: bool
    is_market_open: bool
    should_scan_now: bool
    scan_reason: str
    next_scan_time_et: str

class TickerRequest(BaseModel):
    symbol: str

# ==================== Lifespan Management ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("🚀 Optrader Backend Starting...")
    logger.info(f"🏢 Environment: {'Production' if os.getenv('RAILWAY_ENVIRONMENT') else 'Development'}")
    
    # ✅ REMOVED: Warmup causes immediate rate limiting on startup
    # YFinance will be initialized on first actual request
    logger.info("✅ Server ready - YFinance will initialize on first request")
    
    yield
    
    # Shutdown
    logger.info("👋 Optrader Backend Shutting Down...")
    executor.shutdown(wait=True)

# ==================== FastAPI App ====================

app = FastAPI(
    title="Optrader API",
    version="1.0.0",
    description="AI Options Trading Analysis & Recommendation Engine",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Cache Helpers ====================

def get_cache(key: str) -> Optional[str]:
    """Get value from cache (Redis or memory)"""
    if redis_client:
        try:
            return redis_client.get(key)
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
    return memory_cache.get(key)

def set_cache(key: str, value: str, ttl: int = CACHE_TTL):
    """Set value in cache (Redis or memory)"""
    if redis_client:
        try:
            redis_client.setex(key, ttl, value)
            return
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
    
    # Fallback to memory cache
    memory_cache[key] = value
    # Simple TTL implementation for memory cache
    asyncio.create_task(clear_memory_cache(key, ttl))

async def clear_memory_cache(key: str, ttl: int):
    """Clear memory cache after TTL"""
    await asyncio.sleep(ttl)
    if key in memory_cache:
        del memory_cache[key]

# ==================== YFinance Rate Limiting & Session Management ====================

# ✅ ADDED: User-Agent rotation to avoid rate limiting
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15"
]

def get_random_headers():
    """Get random headers to avoid rate limiting"""
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }

class RateLimiter:
    """Enhanced rate limiter for YFinance API calls"""
    def __init__(self, max_calls_per_minute=20):
        self.max_calls = max_calls_per_minute
        self.calls = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if rate limit exceeded"""
        async with self.lock:
            now = datetime.now()
            # Remove calls older than 1 minute
            self.calls = [t for t in self.calls if (now - t).seconds < 60]
            
            if len(self.calls) >= self.max_calls:
                wait_time = 60 - (now - self.calls[0]).seconds + random.uniform(1, 3)
                logger.warning(f"⏳ Rate limit reached. Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                self.calls = []
            
            self.calls.append(now)

rate_limiter = RateLimiter()

# ==================== Market Hours ====================

def get_market_status() -> MarketStatus:
    """Check if market is open and should scan"""
    et_tz = pytz.timezone('America/New_York')
    now_et = datetime.now(et_tz)
    
    # Check if weekday (Monday=0, Sunday=6)
    is_weekday = now_et.weekday() < 5
    
    # Market hours
    pre_market_start = time(7, 0)  # 7:00 AM ET
    market_open = time(9, 30)      # 9:30 AM ET
    market_close = time(16, 0)     # 4:00 PM ET
    post_market_end = time(20, 0)  # 8:00 PM ET
    
    current_time = now_et.time()
    
    # Determine scan status
    if not is_weekday:
        should_scan = False
        scan_reason = "Weekend - Markets Closed"
        next_scan = "Monday 7:00 AM ET"
    elif pre_market_start <= current_time < market_open:
        should_scan = True
        scan_reason = "Pre-Market Hours"
        next_scan = f"Every 30 minutes"
    elif market_open <= current_time < market_close:
        should_scan = True
        scan_reason = "Market Hours - Active Trading"
        next_scan = f"Every 15 minutes"
    elif market_close <= current_time < post_market_end:
        should_scan = True
        scan_reason = "Post-Market Hours"
        next_scan = f"Every 30 minutes"
    else:
        should_scan = False
        scan_reason = "Outside Trading Hours"
        next_scan = "Tomorrow 7:00 AM ET"
    
    return MarketStatus(
        current_time_et=now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
        is_market_day=is_weekday,
        is_market_open=is_weekday and market_open <= current_time < market_close,
        should_scan_now=should_scan,
        scan_reason=scan_reason,
        next_scan_time_et=next_scan
    )

# ==================== Options Analysis ====================

# ✅ ADDED: Session pool for yfinance
import requests
session_pool = []

def get_yfinance_session():
    """Get or create a session with random headers"""
    if not session_pool or random.random() < 0.3:  # 30% chance to create new session
        session = requests.Session()
        session.headers.update(get_random_headers())
        if len(session_pool) < 5:
            session_pool.append(session)
        return session
    return random.choice(session_pool)

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=2, min=4, max=30),
    reraise=True
)
def fetch_ticker_data(ticker: str) -> Optional[Dict]:
    """Fetch and analyze options data for a single ticker with retry logic"""
    try:
        # ✅ ENHANCED: Add random delay to avoid rate limiting
        delay = RATE_LIMIT_DELAY + random.uniform(0.5, 2.0)
        time_module.sleep(delay)
        
        # ✅ ADDED: Use session with random headers
        session = get_yfinance_session()
        stock = yf.Ticker(ticker, session=session)
        
        # Try to get info with timeout
        try:
            info = stock.info
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                logger.warning(f"⏳ {ticker}: Rate limited, will retry...")
                raise  # Let retry decorator handle it
            logger.warning(f"⚠️ {ticker}: Info fetch failed: {e}")
            return None
        
        # Get current price
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose', 0)
        if current_price <= 0:
            logger.warning(f"❌ {ticker}: Invalid price")
            return None
        
        # Get volume metrics
        volume = info.get('volume', 0)
        avg_volume = info.get('averageVolume', 1)
        market_cap = info.get('marketCap', 0)
        
        # Add small delay before options fetch
        time_module.sleep(random.uniform(1, 2))
        
        # Get options chain
        try:
            expirations = stock.options
        except Exception as e:
            if "429" in str(e):
                logger.warning(f"⏳ {ticker}: Rate limited on options fetch")
                raise
            logger.info(f"ℹ️ {ticker}: No options available")
            return None
            
        if not expirations:
            logger.info(f"ℹ️ {ticker}: No options available")
            return None
        
        # Collect options data (limit to next 90 days and fewer expirations)
        all_options = []
        cutoff_date = datetime.now() + timedelta(days=90)
        
        # ✅ REDUCED: Only fetch 3 nearest expirations to reduce API calls
        for exp_date in expirations[:3]:
            exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
            if exp_datetime > cutoff_date:
                continue
            
            try:
                # Add delay between expiration fetches
                time_module.sleep(random.uniform(0.5, 1.5))
                
                opt_chain = stock.option_chain(exp_date)
                
                # Process calls
                calls = opt_chain.calls.copy()
                calls['expiration'] = exp_date
                calls['type'] = 'call'
                calls['days_to_expiry'] = (exp_datetime - datetime.now()).days
                
                # Process puts
                puts = opt_chain.puts.copy()
                puts['expiration'] = exp_date
                puts['type'] = 'put'
                puts['days_to_expiry'] = (exp_datetime - datetime.now()).days
                
                all_options.append(pd.concat([calls, puts]))
                
            except Exception as e:
                if "429" in str(e):
                    logger.warning(f"⏳ {ticker}: Rate limited on {exp_date}")
                    raise
                logger.warning(f"⚠️ {ticker}: Error fetching {exp_date} options: {e}")
                continue
        
        if not all_options:
            return None
        
        # Combine all options
        options_df = pd.concat(all_options, ignore_index=True)
        
        # Analyze options for whale activity
        analysis = analyze_options_flow(ticker, options_df, current_price, volume, avg_volume, market_cap)
        
        return analysis
        
    except Exception as e:
        if "429" in str(e) or "Too Many Requests" in str(e):
            logger.error(f"🚫 {ticker}: Rate limited - {str(e)}")
            raise  # Let retry handle it
        logger.error(f"❌ {ticker}: Error - {str(e)}")
        return None

def analyze_options_flow(ticker: str, options_df: pd.DataFrame, current_price: float, 
                        volume: int, avg_volume: int, market_cap: float) -> Dict:
    """Analyze options flow for unusual activity and sentiment"""
    
    # Filter for liquid options
    liquid_options = options_df[
        (options_df['volume'] > 100) &
        (options_df['openInterest'] > 50)
    ].copy()
    
    if liquid_options.empty:
        return None
    
    # Calculate additional metrics
    liquid_options['volume_oi_ratio'] = liquid_options['volume'] / (liquid_options['openInterest'] + 1)
    liquid_options['moneyness'] = liquid_options['strike'] / current_price
    liquid_options['total_value'] = liquid_options['volume'] * liquid_options['lastPrice'] * 100
    
    # Identify unusual activity (volume > 2x OI or high value trades)
    unusual_mask = (
        (liquid_options['volume_oi_ratio'] > 2) |
        (liquid_options['total_value'] > 50000)
    )
    
    unusual_activity = liquid_options[unusual_mask].copy()
    
    if unusual_activity.empty:
        return None
    
    # Calculate whale scores
    unusual_activity['whale_score'] = (
        np.log1p(unusual_activity['total_value']) * 0.4 +
        unusual_activity['volume_oi_ratio'] * 100 * 0.3 +
        unusual_activity.get('impliedVolatility', 0.3) * 100 * 0.3
    )
    
    # Separate calls and puts
    unusual_calls = unusual_activity[unusual_activity['type'] == 'call']
    unusual_puts = unusual_activity[unusual_activity['type'] == 'put']
    
    # Calculate sentiment
    total_calls_value = unusual_calls['total_value'].sum() if not unusual_calls.empty else 0
    total_puts_value = unusual_puts['total_value'].sum() if not unusual_puts.empty else 0
    total_value = total_calls_value + total_puts_value
    
    if total_value > 0:
        bullish_pct = (total_calls_value / total_value) * 100
        bearish_pct = (total_puts_value / total_value) * 100
    else:
        bullish_pct = bearish_pct = 50
    
    # Determine sentiment
    if bullish_pct > 65:
        sentiment = "STRONG BULLISH"
        sentiment_score = bullish_pct
    elif bullish_pct > 55:
        sentiment = "BULLISH"
        sentiment_score = bullish_pct
    elif bearish_pct > 65:
        sentiment = "STRONG BEARISH"
        sentiment_score = -bearish_pct
    elif bearish_pct > 55:
        sentiment = "BEARISH"
        sentiment_score = -bearish_pct
    else:
        sentiment = "NEUTRAL"
        sentiment_score = bullish_pct - bearish_pct
    
    # Calculate expected move (simplified)
    atm_options = liquid_options[
        (liquid_options['moneyness'] > 0.97) & 
        (liquid_options['moneyness'] < 1.03)
    ]
    
    expected_move_data = None
    if not atm_options.empty:
        avg_iv = atm_options.get('impliedVolatility', pd.Series([0.3])).mean()
        days_to_exp = atm_options['days_to_expiry'].min()
        expected_move_pct = avg_iv * np.sqrt(days_to_exp / 365) * 100
        
        expected_move_data = {
            "atm_strike": current_price,
            "expected_move_pct": expected_move_pct,
            "upper_target": current_price * (1 + expected_move_pct / 100),
            "lower_target": current_price * (1 - expected_move_pct / 100),
            "confidence": "68% (1σ)"
        }
    
    # Generate strategy recommendations
    strategies = generate_strategies(
        sentiment_score, bullish_pct, bearish_pct, 
        unusual_activity, current_price, expected_move_data
    )
    
    # Prepare unusual activity for JSON
    unusual_json = []
    for _, row in unusual_activity.nlargest(10, 'whale_score').iterrows():
        unusual_json.append({
            "option_type": row['type'].upper(),
            "strike": float(row['strike']),
            "expiration": row['expiration'],
            "volume": int(row['volume']),
            "lastPrice": float(row['lastPrice']),
            "whale_score": float(row['whale_score']),
            "strategy_type": "DIRECTIONAL" if row['type'] == 'call' else "HEDGE"
        })
    
    return {
        "ticker": ticker,
        "current_price": current_price,
        "volume_ratio": volume / avg_volume if avg_volume > 0 else 0,
        "market_cap": market_cap,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "bullish_percent": bullish_pct,
        "bearish_percent": bearish_pct,
        "total_calls_count": len(unusual_calls),
        "total_puts_count": len(unusual_puts),
        "total_calls_value": total_calls_value,
        "total_puts_value": total_puts_value,
        "whale_score": unusual_activity['whale_score'].max(),
        "volatility_plays": 0,  # Simplified
        "directional_bets": len(unusual_activity),
        "unusual_activity": unusual_json,
        "expected_move": expected_move_data,
        "recommended_strategies": strategies,
        "timestamp": datetime.now().isoformat()
    }

def generate_strategies(sentiment_score: float, bullish_pct: float, bearish_pct: float,
                       unusual_activity: pd.DataFrame, current_price: float, 
                       expected_move: Optional[Dict]) -> List[Dict]:
    """Generate trading strategy recommendations"""
    strategies = []
    
    # Get top strikes from unusual activity
    top_calls = unusual_activity[unusual_activity['type'] == 'call'].nlargest(3, 'whale_score')
    top_puts = unusual_activity[unusual_activity['type'] == 'put'].nlargest(3, 'whale_score')
    
    # Strategy 1: Follow the whale
    if not top_calls.empty and sentiment_score > 60:
        top_strike = top_calls.iloc[0]['strike']
        strategies.append({
            "strategy": "FOLLOW WHALE - BUY CALLS",
            "confidence": "HIGH" if sentiment_score > 70 else "MEDIUM",
            "rationale": f"Heavy call buying detected at ${top_strike:.2f} strike",
            "suggested_strikes": f"${top_strike:.2f}",
            "risk_level": "MODERATE",
            "profit_target": "20-30%",
            "stop_loss": "-30%",
            "time_horizon": "1-2 weeks"
        })
    
    elif not top_puts.empty and sentiment_score < -60:
        top_strike = top_puts.iloc[0]['strike']
        strategies.append({
            "strategy": "FOLLOW WHALE - BUY PUTS",
            "confidence": "HIGH" if sentiment_score < -70 else "MEDIUM",
            "rationale": f"Heavy put buying detected at ${top_strike:.2f} strike",
            "suggested_strikes": f"${top_strike:.2f}",
            "risk_level": "MODERATE",
            "profit_target": "20-30%",
            "stop_loss": "-30%",
            "time_horizon": "1-2 weeks"
        })
    
    # Strategy 2: Spreads for high conviction
    if sentiment_score > 65:
        if expected_move:
            upper_target = expected_move['upper_target']
            strategies.append({
                "strategy": "BULL CALL SPREAD",
                "confidence": "HIGH",
                "rationale": "Strong bullish sentiment with defined risk",
                "suggested_strikes": f"Buy ${current_price:.2f} / Sell ${upper_target:.2f}",
                "risk_level": "LOW-MODERATE",
                "profit_target": "40-60%",
                "stop_loss": "Max loss defined",
                "time_horizon": "2-4 weeks"
            })
    
    elif sentiment_score < -65:
        if expected_move:
            lower_target = expected_move['lower_target']
            strategies.append({
                "strategy": "BEAR PUT SPREAD",
                "confidence": "HIGH",
                "rationale": "Strong bearish sentiment with defined risk",
                "suggested_strikes": f"Buy ${current_price:.2f} / Sell ${lower_target:.2f}",
                "risk_level": "LOW-MODERATE",
                "profit_target": "40-60%",
                "stop_loss": "Max loss defined",
                "time_horizon": "2-4 weeks"
            })
    
    # Strategy 3: Neutral/Volatility plays
    if abs(sentiment_score) < 20:
        strategies.append({
            "strategy": "IRON CONDOR",
            "confidence": "MEDIUM",
            "rationale": "Neutral sentiment - profit from range-bound movement",
            "suggested_strikes": "Sell ATM straddle, buy OTM protection",
            "risk_level": "MODERATE",
            "profit_target": "15-25%",
            "stop_loss": "Max loss defined",
            "time_horizon": "2-4 weeks"
        })
    
    # Always add a conservative strategy
    strategies.append({
        "strategy": "WAIT FOR CONFIRMATION",
        "confidence": "LOW",
        "rationale": "Monitor for additional signals before entering",
        "suggested_strikes": "N/A",
        "risk_level": "NONE",
        "profit_target": "N/A",
        "stop_loss": "N/A",
        "time_horizon": "Watch next 1-2 days"
    })
    
    return strategies[:5]  # Return top 5 strategies

# ==================== User Ticker Management ====================

def get_user_tickers(user_id: str = "default") -> List[str]:
    """Get user's watchlist"""
    default_tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY", "QQQ", "AMD", "META", "GOOGL", "AMZN"]
    
    cache_key = f"tickers_{user_id}"
    cached = get_cache(cache_key)
    
    if cached:
        return json.loads(cached)
    
    return default_tickers

def save_user_tickers(user_id: str, tickers: List[str]):
    """Save user's watchlist"""
    cache_key = f"tickers_{user_id}"
    set_cache(cache_key, json.dumps(tickers), ttl=86400)  # 24 hours

# ==================== API Endpoints ====================

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache": "redis" if redis_client else "memory",
        "version": "1.0.0",
        "config": {
            "cache_ttl": CACHE_TTL,
            "rate_limit_delay": RATE_LIMIT_DELAY,
            "max_workers": MAX_WORKERS
        }
    }

@app.get("/api/market-status", response_model=MarketStatus)
async def market_status():
    """Get current market status"""
    return get_market_status()

@app.get("/api/scan", response_model=ScanResponse)
async def scan_market(
    force: bool = False,
    ignore_hours: bool = True,
    max_workers: int = MAX_WORKERS,
    user_id: str = "default"
):
    """
    Scan market for whale activity
    - force: Force new scan (ignore cache)
    - ignore_hours: Scan even outside market hours
    - max_workers: Number of parallel workers (1-5)
    """
    
    # Check market hours
    market = get_market_status()
    if not ignore_hours and not market.should_scan_now:
        raise HTTPException(
            status_code=425,
            detail={
                "error": "Outside scanning window",
                "reason": market.scan_reason,
                "next_scan": market.next_scan_time_et
            }
        )
    
    # Check cache first
    if not force:
        cached = get_cache(f"{SCAN_CACHE_KEY}_{user_id}")
        if cached:
            data = json.loads(cached)
            logger.info(f"📦 Returning cached results for {user_id}")
            return ScanResponse(**data, from_cache=True)
    
    # Get user's tickers
    tickers = get_user_tickers(user_id)
    logger.info(f"📊 Scanning {len(tickers)} tickers for {user_id}")
    
    # Rate limiting check
    await rate_limiter.acquire()
    
    # ✅ ENHANCED: Process sequentially with delays to avoid rate limiting
    # Parallel processing causes too many simultaneous requests
    results = []
    loop = asyncio.get_event_loop()
    
    for i, ticker in enumerate(tickers):
        try:
            # Add progressive delay between tickers
            if i > 0:
                delay = 3 + random.uniform(1, 2)  # 3-5 seconds between tickers
                logger.info(f"⏳ Waiting {delay:.1f}s before {ticker}...")
                await asyncio.sleep(delay)
            
            # Fetch data in executor
            result = await loop.run_in_executor(executor, fetch_ticker_data, ticker)
            
            if result:
                results.append(result)
                logger.info(f"✅ {ticker}: Success (whale score: {result['whale_score']:.1f})")
            else:
                logger.info(f"⚠️ {ticker}: No data")
                
        except Exception as e:
            logger.error(f"❌ {ticker}: Failed - {e}")
            continue
    
    # Sort by whale score
    results.sort(key=lambda x: x['whale_score'], reverse=True)
    
    # Create response
    response_data = {
        "user_id": user_id,
        "results": [TickerData(**r) for r in results],
        "scan_time": datetime.now().isoformat(),
        "total_scanned": len(tickers),
        "total_active": len(results),
        "from_cache": False
    }
    
    # Cache results
    set_cache(f"{SCAN_CACHE_KEY}_{user_id}", json.dumps(response_data))
    
    logger.info(f"✅ Scan complete: {len(results)} active / {len(tickers)} total")
    return ScanResponse(**response_data)

@app.get("/api/tickers")
async def get_tickers(user_id: str = "default"):
    """Get user's ticker watchlist"""
    return {"tickers": get_user_tickers(user_id)}

@app.post("/api/tickers/add")
async def add_ticker(request: TickerRequest, user_id: str = "default"):
    """Add ticker to watchlist"""
    tickers = get_user_tickers(user_id)
    
    if request.symbol.upper() not in tickers:
        tickers.append(request.symbol.upper())
        save_user_tickers(user_id, tickers)
    
    return {"message": "Ticker added", "tickers": tickers}

@app.post("/api/tickers/remove")
async def remove_ticker(request: TickerRequest, user_id: str = "default"):
    """Remove ticker from watchlist"""
    tickers = get_user_tickers(user_id)
    
    if request.symbol.upper() in tickers:
        tickers.remove(request.symbol.upper())
        save_user_tickers(user_id, tickers)
    
    return {"message": "Ticker removed", "tickers": tickers}

@app.get("/api/ticker/{symbol}")
async def get_ticker_analysis(symbol: str):
    """Get detailed analysis for a single ticker"""
    await rate_limiter.acquire()
    
    # Check cache first
    cache_key = f"ticker_{symbol.upper()}"
    cached = get_cache(cache_key)
    
    if cached:
        return json.loads(cached)
    
    # Fetch fresh data
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, fetch_ticker_data, symbol.upper())
    
    if result is None:
        raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
    
    # Cache result
    set_cache(cache_key, json.dumps(result))
    
    return result

# ==================== Error Handlers ====================

@app.exception_handler(429)
async def rate_limit_handler(request, exc):
    """Handle rate limit errors from yfinance"""
    logger.error("🚫 Rate limit hit - implementing backoff")
    return {
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please try again in a few minutes.",
        "retry_after": 60
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"❌ Unexpected error: {exc}")
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again later."
    }

# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True if not os.getenv("RAILWAY_ENVIRONMENT") else False,
        workers=1
    )
