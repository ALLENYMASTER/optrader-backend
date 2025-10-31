"""
Optrader Backend - FastAPI Server with Polygon.io
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
import json
import asyncio
from contextlib import asynccontextmanager
import pytz
import redis
import logging
from concurrent.futures import ThreadPoolExecutor
import os

# ✅ 導入 Polygon.io fetcher（替代 yfinance）
from polygon_fetcher import PolygonDataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Configuration ====================

# ✅ Polygon.io API Key
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "E8RufTcxxWCESJPjjgtIviVNsbnl182n")
if not POLYGON_API_KEY:
    logger.error("❌ POLYGON_API_KEY not set! Get free key at https://massive.com")

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://default:EfKXKBdmYeyvsDmqcVmurRzMooouZhHO@redis.railway.internal:6379")
CACHE_TTL = int(os.getenv("CACHE_TTL", "900"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "3"))

logger.info(f"🔧 Configuration:")
logger.info(f"  - Cache TTL: {CACHE_TTL}s")
logger.info(f"  - Max Workers: {MAX_WORKERS}")
logger.info(f"  - Polygon API: {'Configured ✓' if POLYGON_API_KEY else 'Not configured ✗'}")

SCAN_CACHE_KEY = "scan_results"

# Initialize Redis
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=5)
    redis_client.ping()
    logger.info("✅ Redis connected")
except Exception as e:
    redis_client = None
    logger.warning(f"⚠️ Redis not available: {e}")

memory_cache = {}
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# ✅ Initialize Polygon.io client
polygon_fetcher = PolygonDataFetcher(POLYGON_API_KEY) if POLYGON_API_KEY else None

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

# ==================== Lifespan ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("🚀 Optrader Backend Starting (Polygon.io)...")
    
    # ✅ 驗證 Polygon.io 連接
    if polygon_fetcher:
        try:
            quote = polygon_fetcher.get_stock_quote("SPY")
            if quote:
                logger.info(f"✅ Polygon.io connected - SPY: ${quote['current_price']:.2f}")
            else:
                logger.warning("⚠️ Polygon.io test failed")
        except Exception as e:
            logger.warning(f"⚠️ Polygon.io warmup failed: {e}")
    else:
        logger.error("❌ Polygon.io not configured!")
    
    yield
    
    logger.info("👋 Shutting Down...")
    executor.shutdown(wait=True)

# ==================== FastAPI App ====================

app = FastAPI(
    title="Optrader API (Polygon.io)",
    version="2.0.0",
    description="Options Trading Analysis with Polygon.io",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Cache Helpers ====================

def get_cache(key: str) -> Optional[str]:
    """Get from cache"""
    if redis_client:
        try:
            return redis_client.get(key)
        except:
            pass
    return memory_cache.get(key)

def set_cache(key: str, value: str, ttl: int = CACHE_TTL):
    """Set cache"""
    if redis_client:
        try:
            redis_client.setex(key, ttl, value)
            return
        except:
            pass
    memory_cache[key] = value
    asyncio.create_task(clear_memory_cache(key, ttl))

async def clear_memory_cache(key: str, ttl: int):
    """Clear memory cache after TTL"""
    await asyncio.sleep(ttl)
    if key in memory_cache:
        del memory_cache[key]

# ==================== Market Hours ====================

def get_market_status() -> MarketStatus:
    """Check if market is open"""
    et_tz = pytz.timezone('America/New_York')
    now_et = datetime.now(et_tz)
    
    is_weekday = now_et.weekday() < 5
    
    pre_market_start = time(7, 0)
    market_open = time(9, 30)
    market_close = time(16, 0)
    post_market_end = time(20, 0)
    
    current_time = now_et.time()
    
    if not is_weekday:
        should_scan = False
        scan_reason = "Weekend - Markets Closed"
        next_scan = "Monday 7:00 AM ET"
    elif pre_market_start <= current_time < market_open:
        should_scan = True
        scan_reason = "Pre-Market Hours"
        next_scan = "Every 30 minutes"
    elif market_open <= current_time < market_close:
        should_scan = True
        scan_reason = "Market Hours - Active Trading"
        next_scan = "Every 15 minutes"
    elif market_close <= current_time < post_market_end:
        should_scan = True
        scan_reason = "Post-Market Hours"
        next_scan = "Every 30 minutes"
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

async def fetch_ticker_data(ticker: str) -> Optional[Dict]:
    """
    ✅ 使用 Polygon.io 獲取數據（替代 yfinance）
    """
    if not polygon_fetcher:
        logger.error("❌ Polygon.io not configured")
        return None
    
    try:
        logger.info(f"📊 Fetching {ticker} from Polygon.io...")
        
        # 使用 Polygon.io 獲取數據
        data = await polygon_fetcher.fetch_ticker_data_async(ticker)
        
        if not data:
            logger.warning(f"⚠️ {ticker}: No data available")
            return None
        
        quote = data['quote']
        options_df = data['options']
        
        if options_df.empty:
            logger.info(f"ℹ️ {ticker}: No options data")
            return None
        
        # 分析期權流
        analysis = analyze_options_flow(
            ticker=ticker,
            options_df=options_df,
            current_price=quote['current_price'],
            volume=quote['volume'],
            avg_volume=quote['volume'] / data['volume_ratio'] if data['volume_ratio'] > 0 else 1,
            market_cap=quote['market_cap']
        )
        
        return analysis
        
    except Exception as e:
        logger.error(f"❌ {ticker}: Error - {e}")
        return None

def analyze_options_flow(ticker: str, options_df: pd.DataFrame, current_price: float,
                        volume: int, avg_volume: int, market_cap: float) -> Dict:
    """分析期權流（邏輯保持不變）"""
    
    # 過濾流動性好的期權
    liquid_options = options_df[
        (options_df['volume'] > 100) &
        (options_df['openInterest'] > 50)
    ].copy()
    
    if liquid_options.empty:
        return None
    
    # 計算指標
    liquid_options['volume_oi_ratio'] = liquid_options['volume'] / (liquid_options['openInterest'] + 1)
    liquid_options['moneyness'] = liquid_options['strike'] / current_price
    liquid_options['total_value'] = liquid_options['volume'] * liquid_options['lastPrice'] * 100
    
    # 異常活動
    unusual_mask = (
        (liquid_options['volume_oi_ratio'] > 2) |
        (liquid_options['total_value'] > 50000)
    )
    
    unusual_activity = liquid_options[unusual_mask].copy()
    
    if unusual_activity.empty:
        return None
    
    # 鯨魚分數
    unusual_activity['whale_score'] = (
        np.log1p(unusual_activity['total_value']) * 0.4 +
        unusual_activity['volume_oi_ratio'] * 100 * 0.3 +
        unusual_activity['impliedVolatility'] * 100 * 0.3
    )
    
    # Call 和 Put 分離
    unusual_calls = unusual_activity[unusual_activity['type'] == 'call']
    unusual_puts = unusual_activity[unusual_activity['type'] == 'put']
    
    # 計算情緒
    total_calls_value = unusual_calls['total_value'].sum() if not unusual_calls.empty else 0
    total_puts_value = unusual_puts['total_value'].sum() if not unusual_puts.empty else 0
    total_value = total_calls_value + total_puts_value
    
    if total_value > 0:
        bullish_pct = (total_calls_value / total_value) * 100
        bearish_pct = (total_puts_value / total_value) * 100
    else:
        bullish_pct = bearish_pct = 50
    
    # 判斷情緒
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
    
    # 預期波動
    atm_options = liquid_options[
        (liquid_options['moneyness'] > 0.97) &
        (liquid_options['moneyness'] < 1.03)
    ]
    
    expected_move_data = None
    if not atm_options.empty:
        avg_iv = atm_options['impliedVolatility'].mean()
        days_to_exp = atm_options['days_to_expiry'].min()
        expected_move_pct = avg_iv * np.sqrt(days_to_exp / 365) * 100
        
        expected_move_data = {
            "atm_strike": current_price,
            "expected_move_pct": expected_move_pct,
            "upper_target": current_price * (1 + expected_move_pct / 100),
            "lower_target": current_price * (1 - expected_move_pct / 100),
            "confidence": "68% (1σ)"
        }
    
    # 生成策略
    strategies = generate_strategies(
        sentiment_score, bullish_pct, bearish_pct,
        unusual_activity, current_price, expected_move_data
    )
    
    # 準備異常活動數據
    unusual_json = []
    for _, row in unusual_activity.nlargest(10, 'whale_score').iterrows():
        unusual_json.append({
            "option_type": row['type'].upper(),
            "strike": float(row['strike']),
            "expiration": row['expiration'],
            "volume": int(row['volume']),
            "lastPrice": float(row['lastPrice']),
            "whale_score": float(row['whale_score']),
            "strategy_type": "DIRECTIONAL"
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
        "volatility_plays": 0,
        "directional_bets": len(unusual_activity),
        "unusual_activity": unusual_json,
        "expected_move": expected_move_data,
        "recommended_strategies": strategies,
        "timestamp": datetime.now().isoformat()
    }

def generate_strategies(sentiment_score: float, bullish_pct: float, bearish_pct: float,
                       unusual_activity: pd.DataFrame, current_price: float,
                       expected_move: Optional[Dict]) -> List[Dict]:
    """生成交易策略（保持不變）"""
    strategies = []
    
    top_calls = unusual_activity[unusual_activity['type'] == 'call'].nlargest(3, 'whale_score')
    top_puts = unusual_activity[unusual_activity['type'] == 'put'].nlargest(3, 'whale_score')
    
    if not top_calls.empty and sentiment_score > 60:
        top_strike = top_calls.iloc[0]['strike']
        strategies.append({
            "strategy": "FOLLOW WHALE - BUY CALLS",
            "confidence": "HIGH" if sentiment_score > 70 else "MEDIUM",
            "rationale": f"Heavy call buying at ${top_strike:.2f}",
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
            "rationale": f"Heavy put buying at ${top_strike:.2f}",
            "suggested_strikes": f"${top_strike:.2f}",
            "risk_level": "MODERATE",
            "profit_target": "20-30%",
            "stop_loss": "-30%",
            "time_horizon": "1-2 weeks"
        })
    
    strategies.append({
        "strategy": "WAIT FOR CONFIRMATION",
        "confidence": "LOW",
        "rationale": "Monitor for additional signals",
        "suggested_strikes": "N/A",
        "risk_level": "NONE",
        "profit_target": "N/A",
        "stop_loss": "N/A",
        "time_horizon": "1-2 days"
    })
    
    return strategies[:5]

# ==================== Ticker Management ====================

def get_user_tickers(user_id: str = "default") -> List[str]:
    """Get user watchlist"""
    default_tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY", "QQQ", "AMD", "META", "GOOGL", "AMZN"]
    
    cache_key = f"tickers_{user_id}"
    cached = get_cache(cache_key)
    
    if cached:
        return json.loads(cached)
    
    return default_tickers

def save_user_tickers(user_id: str, tickers: List[str]):
    """Save user watchlist"""
    cache_key = f"tickers_{user_id}"
    set_cache(cache_key, json.dumps(tickers), ttl=86400)

# ==================== API Endpoints ====================

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_source": "Polygon.io",
        "polygon_configured": polygon_fetcher is not None,
        "cache": "redis" if redis_client else "memory"
    }

@app.get("/api/market-status", response_model=MarketStatus)
async def market_status():
    """Get market status"""
    return get_market_status()

@app.get("/api/scan", response_model=ScanResponse)
async def scan_market(
    force: bool = False,
    ignore_hours: bool = False,
    max_workers: int = MAX_WORKERS,
    user_id: str = "default"
):
    """
    掃描市場（使用 Polygon.io）
    """
    
    if not polygon_fetcher:
        raise HTTPException(
            status_code=500,
            detail="Polygon.io not configured. Set POLYGON_API_KEY environment variable."
        )
    
    # 檢查市場時間
    if not ignore_hours:
        market = get_market_status()
        if not market.should_scan_now:
            logger.warning(f"❌ Scan rejected: {market.scan_reason}")
            raise HTTPException(
                status_code=425,
                detail={
                    "error": "Outside scanning window",
                    "reason": market.scan_reason,
                    "next_scan": market.next_scan_time_et,
                    "tip": "Add ignore_hours=true to scan anyway"
                }
            )
    
    # 檢查緩存
    if not force:
        cached = get_cache(f"{SCAN_CACHE_KEY}_{user_id}")
        if cached:
            data = json.loads(cached)
            logger.info(f"📦 Returning cached results")
            return ScanResponse(**data, from_cache=True)
    
    # 獲取股票列表
    tickers = get_user_tickers(user_id)
    logger.info(f"📊 Scanning {len(tickers)} tickers with Polygon.io")
    
    # ✅ Polygon.io Free Tier: 5 calls/minute
    # 順序處理，每個ticker之間間隔12秒以遵守限制
    results = []
    
    for i, ticker in enumerate(tickers):
        try:
            # Rate limiting: 5 calls/min = 12s per call
            if i > 0:
                await asyncio.sleep(12)
            
            result = await fetch_ticker_data(ticker)
            
            if result:
                results.append(result)
                logger.info(f"✅ {ticker}: Whale score {result['whale_score']:.1f}")
            
        except Exception as e:
            logger.error(f"❌ {ticker}: Failed - {e}")
            continue
    
    # 排序
    results.sort(key=lambda x: x['whale_score'], reverse=True)
    
    # 創建響應
    response_data = {
        "user_id": user_id,
        "results": [TickerData(**r) for r in results],
        "scan_time": datetime.now().isoformat(),
        "total_scanned": len(tickers),
        "total_active": len(results),
        "from_cache": False
    }
    
    # 緩存結果
    set_cache(f"{SCAN_CACHE_KEY}_{user_id}", json.dumps(response_data))
    
    logger.info(f"✅ Scan complete: {len(results)}/{len(tickers)}")
    return ScanResponse(**response_data)

@app.get("/api/tickers")
async def get_tickers(user_id: str = "default"):
    """Get tickers"""
    return {"tickers": get_user_tickers(user_id)}

@app.post("/api/tickers/add")
async def add_ticker(request: TickerRequest, user_id: str = "default"):
    """Add ticker"""
    tickers = get_user_tickers(user_id)
    if request.symbol.upper() not in tickers:
        tickers.append(request.symbol.upper())
        save_user_tickers(user_id, tickers)
    return {"message": "Ticker added", "tickers": tickers}

@app.post("/api/tickers/remove")
async def remove_ticker(request: TickerRequest, user_id: str = "default"):
    """Remove ticker"""
    tickers = get_user_tickers(user_id)
    if request.symbol.upper() in tickers:
        tickers.remove(request.symbol.upper())
        save_user_tickers(user_id, tickers)
    return {"message": "Ticker removed", "tickers": tickers}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
