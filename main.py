"""
Optrader Cloud Backend - FastAPI
Main entry point for options whale scanning API
"""
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import os
from datetime import datetime

# Import modules
from models.options_scanner import OptionsWhaleScanner
from models.market_analyzer import MarketAnalyzer
from storage.ticker_manager import TickerManager
from storage.cache_manager import CacheManager
from api.schemas import (
    TickerRequest, TickerListResponse, ScanResultResponse,
    MarketSummaryResponse, TickerDetailResponse, HealthResponse
)
from config.settings import settings, MarketHours, get_market_status

# ============================================================================
#  APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Optrader API",
    description="Options Whale Scanner & AI Trading Recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for iPhone/web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
#  GLOBAL INSTANCES
# ============================================================================

ticker_manager = TickerManager()
cache_manager = CacheManager()
scanner = OptionsWhaleScanner()
market_analyzer = MarketAnalyzer()

# Default tickers
DEFAULT_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "IBIT", "KRE", "SOXX", "GLD", "SLV", "NVDA", "AAPL", 
    "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NFLX", "AVGO", "TSM", "AMD", "MRVL", "INTC", 
    "ARM", "QCOM", "ASML", "AMAT", "LRCX", "WMT", "COST", "ORCL", "PLTR", "PANW", "CRWD", 
    "JPM", "BAC", "SCHW", "BLK", "AFRM", "HOOD", "SOFI", "AAL", "DAL", "UAL", "CCL", "RTX", 
    "ISRG", "LLY", "TEM", "DOCS", "NNE", "SMR", "SCCO", "FCX", "MP", "EXPE", "UBER", "COIN", 
    "BABA", "BIDU", "IONQ", "QBTS", "RGTI", "RR", "LUNR", "RKLB", "OKLO", "DHI"
]

# ============================================================================
#  STARTUP & SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("=" * 80)
    print("🚀 Optrader API Starting...")
    print("=" * 80)
    
    # Initialize default tickers
    for user_id in ["default"]:
        if not ticker_manager.get_tickers(user_id):
            ticker_manager.set_tickers(user_id, DEFAULT_TICKERS)
    
    print(f"✅ Default tickers loaded: {len(DEFAULT_TICKERS)}")
    print(f"✅ Cache manager initialized")
    print(f"✅ Options scanner ready")
    print("=" * 80)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Optrader API shutting down...")
    cache_manager.clear_all()

# ============================================================================
#  ROOT & HEALTH ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "app": "Optrader API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "tickers": "/api/tickers",
            "scan": "/api/scan",
            "market_summary": "/api/market-summary",
            "market_status": "/api/market-status",
            "ticker_detail": "/api/ticker/{symbol}",
            "health": "/health"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/favicon.ico")
async def favicon():
    """Return 204 No Content for favicon requests to avoid 404 errors"""
    from fastapi.responses import Response
    return Response(status_code=204)

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_size": cache_manager.get_cache_size(),
        "uptime": "running"
    }

# ============================================================================
#  TICKER MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/api/tickers", response_model=TickerListResponse, tags=["Tickers"])
async def get_tickers(user_id: str = Query("default", description="User ID")):
    """
    Get user's ticker list
    
    - **user_id**: User identifier (default: "default")
    """
    tickers = ticker_manager.get_tickers(user_id)
    return {
        "user_id": user_id,
        "tickers": tickers,
        "count": len(tickers),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/tickers/add", response_model=TickerListResponse, tags=["Tickers"])
async def add_ticker(
    request: TickerRequest,
    user_id: str = Query("default", description="User ID")
):
    """
    Add ticker to user's list
    
    - **symbol**: Stock ticker symbol (e.g., "NVDA")
    - **user_id**: User identifier
    """
    symbol = request.symbol.upper().strip()
    
    # Validate ticker format
    if not symbol or len(symbol) > 5:
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")
    
    success = ticker_manager.add_ticker(user_id, symbol)
    
    if not success:
        raise HTTPException(status_code=400, detail=f"Ticker {symbol} already exists")
    
    # Clear cache for this user
    cache_manager.delete(f"scan_{user_id}")
    
    tickers = ticker_manager.get_tickers(user_id)
    return {
        "user_id": user_id,
        "tickers": tickers,
        "count": len(tickers),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/tickers/remove", response_model=TickerListResponse, tags=["Tickers"])
async def remove_ticker(
    request: TickerRequest,
    user_id: str = Query("default", description="User ID")
):
    """
    Remove ticker from user's list
    
    - **symbol**: Stock ticker symbol to remove
    - **user_id**: User identifier
    """
    symbol = request.symbol.upper().strip()
    success = ticker_manager.remove_ticker(user_id, symbol)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Ticker {symbol} not found")
    
    # Clear cache
    cache_manager.delete(f"scan_{user_id}")
    
    tickers = ticker_manager.get_tickers(user_id)
    return {
        "user_id": user_id,
        "tickers": tickers,
        "count": len(tickers),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/tickers/set", response_model=TickerListResponse, tags=["Tickers"])
async def set_tickers(
    tickers: List[str],
    user_id: str = Query("default", description="User ID")
):
    """
    Replace entire ticker list
    
    - **tickers**: List of ticker symbols
    - **user_id**: User identifier
    """
    # Validate and clean tickers
    clean_tickers = [t.upper().strip() for t in tickers if t.strip()]
    
    if len(clean_tickers) == 0:
        raise HTTPException(status_code=400, detail="Ticker list cannot be empty")
    
    ticker_manager.set_tickers(user_id, clean_tickers)
    
    # Clear cache
    cache_manager.delete(f"scan_{user_id}")
    
    return {
        "user_id": user_id,
        "tickers": clean_tickers,
        "count": len(clean_tickers),
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
#  SCANNING ENDPOINTS
# ============================================================================

@app.get("/api/scan", response_model=ScanResultResponse, tags=["Scanning"])
async def scan_market(
    user_id: str = Query("default", description="User ID"),
    force: bool = Query(False, description="Force refresh (bypass cache)"),
    parallel: bool = Query(True, description="Use parallel scanning"),
    ignore_hours: bool = Query(False, description="Ignore market hours check")
):
    """
    Scan user's tickers for whale activity
    
    - **user_id**: User identifier
    - **force**: Bypass cache and force new scan
    - **parallel**: Use parallel scanning (faster, default: True)
    - **ignore_hours**: Bypass market hours check (default: False)
    
    Returns sorted results: BULLISH → BEARISH
    
    **Smart Scheduling:**
    - Scans only during: Pre-market (1h before), Market hours (every 1h), Post-market (1h after)
    - Use `ignore_hours=true` to force scan anytime
    """
    
    # Check market hours (unless ignored)
    if not ignore_hours:
        should_scan, reason = MarketHours.should_scan_now()
        if not should_scan:
            next_scan = MarketHours.next_scan_time()
            raise HTTPException(
                status_code=425,  # Too Early
                detail={
                    "error": "Outside scanning window",
                    "reason": reason,
                    "next_scan_time": next_scan.isoformat(),
                    "next_scan_et": next_scan.strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "hint": "Use ignore_hours=true to force scan"
                }
            )
    
    # Check cache
    cache_key = f"scan_{user_id}"
    if not force:
        cached = cache_manager.get(cache_key)
        if cached:
            cached['from_cache'] = True
            return cached
    
    # Get user's tickers
    tickers = ticker_manager.get_tickers(user_id)
    
    if not tickers:
        raise HTTPException(
            status_code=400, 
            detail="No tickers configured. Add tickers first."
        )
    
    # Scan tickers
    print(f"📊 Scanning {len(tickers)} tickers for user: {user_id}")
    
    if parallel:
        results = scanner.scan_all_parallel(tickers, max_workers=5)
    else:
        results = scanner.scan_all_sequential(tickers)
    
    # Sort by sentiment score (bullish → bearish)
    results.sort(key=lambda x: x.get('sentiment_score', 0), reverse=True)
    
    response = {
        "user_id": user_id,
        "results": results,
        "scan_time": datetime.now().isoformat(),
        "total_scanned": len(tickers),
        "total_active": len(results),
        "from_cache": False
    }
    
    # Cache for 15 minutes
    cache_manager.set(cache_key, response, ttl=900)
    
    return response

@app.get("/api/ticker/{symbol}", response_model=TickerDetailResponse, tags=["Scanning"])
async def get_ticker_detail(
    symbol: str,
    force: bool = Query(False, description="Force refresh")
):
    """
    Get detailed analysis for a specific ticker
    
    - **symbol**: Stock ticker symbol
    - **force**: Bypass cache
    """
    symbol = symbol.upper().strip()
    
    # Check cache
    cache_key = f"ticker_{symbol}"
    if not force:
        cached = cache_manager.get(cache_key)
        if cached:
            return cached
    
    # Scan single ticker
    result = scanner.scan_single_ticker(symbol)
    
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No unusual activity found for {symbol} or ticker not found"
        )
    
    response = {
        "symbol": symbol,
        "data": result,
        "timestamp": datetime.now().isoformat()
    }
    
    # Cache for 10 minutes
    cache_manager.set(cache_key, response, ttl=600)
    
    return response

# ============================================================================
#  MARKET ANALYSIS ENDPOINTS
# ============================================================================

@app.get("/api/market-summary", response_model=MarketSummaryResponse, tags=["Market Analysis"])
async def market_summary(
    user_id: str = Query("default", description="User ID"),
    force: bool = Query(False, description="Force refresh")
):
    """
    Get overall market sentiment and top opportunities
    
    - **user_id**: User identifier
    - **force**: Bypass cache
    
    Returns aggregated market analysis with top 5 opportunities
    """
    
    # Check cache
    cache_key = f"market_summary_{user_id}"
    if not force:
        cached = cache_manager.get(cache_key)
        if cached:
            return cached
    
    # Get scan results
    scan_data = await scan_market(user_id=user_id, force=force)
    results = scan_data['results']
    
    if not results:
        return {
            "user_id": user_id,
            "regime": "UNKNOWN",
            "regime_icon": "❓",
            "bullish_percent": 0.0,
            "bearish_percent": 0.0,
            "total_tickers": 0,
            "active_tickers": 0,
            "top_opportunities": [],
            "risk_level": "UNKNOWN",
            "timestamp": datetime.now().isoformat()
        }
    
    # Use MarketAnalyzer
    summary = market_analyzer.analyze(results)
    summary['user_id'] = user_id
    summary['timestamp'] = datetime.now().isoformat()
    
    # Cache for 15 minutes
    cache_manager.set(cache_key, summary, ttl=900)
    
    return summary

# ============================================================================
#  UTILITY ENDPOINTS
# ============================================================================

@app.post("/api/cache/clear", tags=["Utilities"])
async def clear_cache(user_id: Optional[str] = Query(None, description="User ID (optional)")):
    """
    Clear cache
    
    - **user_id**: Clear cache for specific user, or all if not provided
    """
    if user_id:
        cache_manager.delete(f"scan_{user_id}")
        cache_manager.delete(f"market_summary_{user_id}")
        return {"message": f"Cache cleared for user: {user_id}"}
    else:
        cache_manager.clear_all()
        return {"message": "All cache cleared"}

@app.get("/api/default-tickers", tags=["Utilities"])
async def get_default_tickers():
    """Get default ticker list"""
    return {
        "default_tickers": DEFAULT_TICKERS,
        "count": len(DEFAULT_TICKERS)
    }

@app.get("/api/market-status", tags=["Utilities"])
async def market_status():
    """
    Get current market status and next scan time
    
    Returns market hours, trading day info, and next recommended scan time
    """
    return get_market_status()

# ============================================================================
#  ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    print(f"❌ Unexpected error: {exc}")
    return {
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
#  MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    print("\n" + "=" * 80)
    print("🚀 Starting Optrader API Server")
    print("=" * 80)
    print(f"📡 Port: {port}")
    print(f"📚 Docs: http://localhost:{port}/docs")
    print(f"🔄 Redoc: http://localhost:{port}/redoc")
    print("=" * 80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
