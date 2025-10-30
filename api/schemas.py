"""
API Data Models (Pydantic Schemas)
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# ============================================================================
#  REQUEST MODELS
# ============================================================================

class TickerRequest(BaseModel):
    """Request to add/remove a single ticker"""
    symbol: str = Field(..., min_length=1, max_length=5, description="Stock ticker symbol")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "NVDA"
            }
        }

# ============================================================================
#  RESPONSE MODELS
# ============================================================================

class TickerListResponse(BaseModel):
    """User's ticker list response"""
    user_id: str
    tickers: List[str]
    count: int
    timestamp: str

class Strategy(BaseModel):
    """Trading strategy recommendation"""
    strategy: str
    confidence: str
    rationale: str
    suggested_strikes: str
    risk_level: str
    profit_target: str
    stop_loss: str
    time_horizon: str

class ExpectedMove(BaseModel):
    """Expected move calculation for weekly options"""
    atm_strike: float
    atm_straddle_price: float
    expected_move_dollar: float
    expected_move_pct: float
    upper_target: float
    lower_target: float
    confidence: str

class TickerData(BaseModel):
    """Individual ticker analysis result"""
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
    unusual_activity: List[Dict[str, Any]]
    expected_move: Optional[ExpectedMove] = None
    recommended_strategies: List[Strategy]
    timestamp: str

class ScanResultResponse(BaseModel):
    """Market scan results"""
    user_id: str
    results: List[TickerData]
    scan_time: str
    total_scanned: int
    total_active: int
    from_cache: bool = False

class TickerDetailResponse(BaseModel):
    """Detailed ticker analysis"""
    symbol: str
    data: TickerData
    timestamp: str

class Opportunity(BaseModel):
    """Top trading opportunity"""
    ticker: str
    sentiment: str
    sentiment_score: float
    strategy: str
    confidence: str
    current_price: float

class MarketSummaryResponse(BaseModel):
    """Overall market summary"""
    user_id: str
    regime: str
    regime_icon: str
    bullish_percent: float
    bearish_percent: float
    total_tickers: int
    active_tickers: int
    top_opportunities: List[Opportunity]
    risk_level: str
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    cache_size: int
    uptime: str
