"""
Configuration and Settings
Includes market hours management and environment variables
"""
from datetime import datetime, time, timedelta
from typing import Optional, Tuple
import pytz
import os

class Settings:
    """Application settings and configuration"""
    
    # API Settings
    APP_NAME = "Optrader API"
    VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # Cache
    CACHE_TTL_SCAN = int(os.getenv("CACHE_TTL_SCAN", 900))  # 15 min
    CACHE_TTL_TICKER = int(os.getenv("CACHE_TTL_TICKER", 600))  # 10 min
    CACHE_TTL_MARKET = int(os.getenv("CACHE_TTL_MARKET", 900))  # 15 min
    
    # Scanning
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", 5))
    DEFAULT_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "IBIT", "KRE", "SOXX", "GLD", "SLV", "NVDA", "AAPL", 
    "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NFLX", "AVGO", "TSM", "AMD", "MRVL", "INTC", 
    "ARM", "QCOM", "ASML", "AMAT", "LRCX", "WMT", "COST", "ORCL", "PLTR", "PANW", "CRWD", 
    "JPM", "BAC", "SCHW", "BLK", "AFRM", "HOOD", "SOFI", "AAL", "DAL", "UAL", "CCL", "RTX", 
    "ISRG", "LLY", "TEM", "DOCS", "NNE", "SMR", "SCCO", "FCX", "MP", "EXPE", "UBER", "COIN", 
    "BABA", "BIDU", "IONQ", "QBTS", "RGTI", "RR", "LUNR", "RKLB", "OKLO", "DHI"
]
    
    # Market Hours (US Eastern Time)
    TIMEZONE = pytz.timezone('America/New_York')
    MARKET_OPEN = time(9, 30)   # 9:30 AM ET
    MARKET_CLOSE = time(16, 0)  # 4:00 PM ET
    
    # Scanning Windows (relative to market hours)
    PRE_MARKET_HOURS = 1   # Scan 1 hour before open
    POST_MARKET_HOURS = 1  # Scan 1 hour after close
    MARKET_SCAN_INTERVAL_HOURS = 1  # Scan every 1 hour during market
    
    # Days market is open
    MARKET_DAYS = [0, 1, 2, 3, 4]  # Monday=0, Friday=4
    
    # US Market Holidays 2024-2025
    MARKET_HOLIDAYS = [
        # 2024
        "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29",
        "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
        "2024-11-28", "2024-12-25",
        # 2025
        "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18",
        "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
        "2025-11-27", "2025-12-25"
    ]

settings = Settings()


class MarketHours:
    """Market hours and scheduling logic"""
    
    @staticmethod
    def get_current_time() -> datetime:
        """Get current time in US Eastern timezone"""
        return datetime.now(settings.TIMEZONE)
    
    @staticmethod
    def is_market_day(dt: Optional[datetime] = None) -> bool:
        """
        Check if given date is a market trading day
        
        Args:
            dt: DateTime to check (default: now)
            
        Returns:
            True if market is open this day
        """
        if dt is None:
            dt = MarketHours.get_current_time()
        
        # Check if weekend
        if dt.weekday() not in settings.MARKET_DAYS:
            return False
        
        # Check if holiday
        date_str = dt.strftime("%Y-%m-%d")
        if date_str in settings.MARKET_HOLIDAYS:
            return False
        
        return True
    
    @staticmethod
    def is_market_open(dt: Optional[datetime] = None) -> bool:
        """
        Check if market is currently open
        
        Args:
            dt: DateTime to check (default: now)
            
        Returns:
            True if market is currently open
        """
        if dt is None:
            dt = MarketHours.get_current_time()
        
        # Must be a market day
        if not MarketHours.is_market_day(dt):
            return False
        
        # Check if within market hours
        current_time = dt.time()
        return settings.MARKET_OPEN <= current_time < settings.MARKET_CLOSE
    
    @staticmethod
    def is_pre_market_window(dt: Optional[datetime] = None) -> bool:
        """
        Check if within pre-market scanning window
        (1 hour before market open)
        
        Args:
            dt: DateTime to check (default: now)
            
        Returns:
            True if in pre-market window
        """
        if dt is None:
            dt = MarketHours.get_current_time()
        
        if not MarketHours.is_market_day(dt):
            return False
        
        # Calculate pre-market start time
        pre_market_start = (
            datetime.combine(dt.date(), settings.MARKET_OPEN) - 
            timedelta(hours=settings.PRE_MARKET_HOURS)
        ).time()
        
        current_time = dt.time()
        return pre_market_start <= current_time < settings.MARKET_OPEN
    
    @staticmethod
    def is_post_market_window(dt: Optional[datetime] = None) -> bool:
        """
        Check if within post-market scanning window
        (1 hour after market close)
        
        Args:
            dt: DateTime to check (default: now)
            
        Returns:
            True if in post-market window
        """
        if dt is None:
            dt = MarketHours.get_current_time()
        
        if not MarketHours.is_market_day(dt):
            return False
        
        # Calculate post-market end time
        post_market_end = (
            datetime.combine(dt.date(), settings.MARKET_CLOSE) + 
            timedelta(hours=settings.POST_MARKET_HOURS)
        ).time()
        
        current_time = dt.time()
        return settings.MARKET_CLOSE <= current_time < post_market_end
    
    @staticmethod
    def should_scan_now(dt: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Determine if scanning should happen now
        
        Args:
            dt: DateTime to check (default: now)
            
        Returns:
            Tuple of (should_scan, reason)
        """
        if dt is None:
            dt = MarketHours.get_current_time()
        
        # Check if market day
        if not MarketHours.is_market_day(dt):
            next_day = MarketHours.next_market_day(dt)
            return False, f"Market closed (next: {next_day.strftime('%Y-%m-%d')})"
        
        # Check pre-market window
        if MarketHours.is_pre_market_window(dt):
            return True, "Pre-market window (1h before open)"
        
        # Check market hours
        if MarketHours.is_market_open(dt):
            return True, "Market is open"
        
        # Check post-market window
        if MarketHours.is_post_market_window(dt):
            return True, "Post-market window (1h after close)"
        
        # Outside all windows
        if dt.time() < settings.MARKET_OPEN:
            return False, f"Before market hours (opens {settings.MARKET_OPEN.strftime('%H:%M')} ET)"
        else:
            return False, f"After market hours (closed {settings.MARKET_CLOSE.strftime('%H:%M')} ET)"
    
    @staticmethod
    def next_scan_time(dt: Optional[datetime] = None) -> datetime:
        """
        Calculate next recommended scan time
        
        Args:
            dt: DateTime to check (default: now)
            
        Returns:
            Next recommended scan datetime
        """
        if dt is None:
            dt = MarketHours.get_current_time()
        
        current_time = dt.time()
        
        # If not a market day, go to next market day pre-market
        if not MarketHours.is_market_day(dt):
            next_day = MarketHours.next_market_day(dt)
            pre_market_time = (
                datetime.combine(next_day.date(), settings.MARKET_OPEN) - 
                timedelta(hours=settings.PRE_MARKET_HOURS)
            )
            return settings.TIMEZONE.localize(pre_market_time)
        
        # Before pre-market: wait for pre-market
        pre_market_start = (
            datetime.combine(dt.date(), settings.MARKET_OPEN) - 
            timedelta(hours=settings.PRE_MARKET_HOURS)
        ).time()
        
        if current_time < pre_market_start:
            next_time = datetime.combine(dt.date(), pre_market_start)
            return settings.TIMEZONE.localize(next_time)
        
        # In pre-market: scan at market open
        if MarketHours.is_pre_market_window(dt):
            next_time = datetime.combine(dt.date(), settings.MARKET_OPEN)
            return settings.TIMEZONE.localize(next_time)
        
        # During market hours: next hour interval
        if MarketHours.is_market_open(dt):
            next_hour = (dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            if next_hour.time() < settings.MARKET_CLOSE:
                return next_hour
            else:
                # Next scan at market close
                next_time = datetime.combine(dt.date(), settings.MARKET_CLOSE)
                return settings.TIMEZONE.localize(next_time)
        
        # After market close but before post-market end: already scanned
        if MarketHours.is_post_market_window(dt):
            # Next scan tomorrow pre-market
            next_day = MarketHours.next_market_day(dt + timedelta(days=1))
            pre_market_time = (
                datetime.combine(next_day.date(), settings.MARKET_OPEN) - 
                timedelta(hours=settings.PRE_MARKET_HOURS)
            )
            return settings.TIMEZONE.localize(pre_market_time)
        
        # After post-market: next day pre-market
        next_day = MarketHours.next_market_day(dt + timedelta(days=1))
        pre_market_time = (
            datetime.combine(next_day.date(), settings.MARKET_OPEN) - 
            timedelta(hours=settings.PRE_MARKET_HOURS)
        )
        return settings.TIMEZONE.localize(pre_market_time)
    
    @staticmethod
    def next_market_day(dt: Optional[datetime] = None) -> datetime:
        """
        Find next market day
        
        Args:
            dt: Starting date (default: now)
            
        Returns:
            Next market day datetime
        """
        if dt is None:
            dt = MarketHours.get_current_time()
        
        next_day = dt + timedelta(days=1)
        
        # Keep searching until we find a market day (max 10 days)
        for _ in range(10):
            if MarketHours.is_market_day(next_day):
                return next_day
            next_day += timedelta(days=1)
        
        return next_day
    
    @staticmethod
    def get_market_status() -> dict:
        """
        Get current market status
        
        Returns:
            Dictionary with market status information
        """
        now = MarketHours.get_current_time()
        should_scan, reason = MarketHours.should_scan_now(now)
        next_scan = MarketHours.next_scan_time(now)
        
        return {
            "current_time": now.isoformat(),
            "current_time_et": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "is_market_day": MarketHours.is_market_day(now),
            "is_market_open": MarketHours.is_market_open(now),
            "is_pre_market": MarketHours.is_pre_market_window(now),
            "is_post_market": MarketHours.is_post_market_window(now),
            "should_scan_now": should_scan,
            "scan_reason": reason,
            "next_scan_time": next_scan.isoformat(),
            "next_scan_time_et": next_scan.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "market_open_time": settings.MARKET_OPEN.strftime("%H:%M ET"),
            "market_close_time": settings.MARKET_CLOSE.strftime("%H:%M ET")
        }


# Convenience functions
def get_market_status() -> dict:
    """Get current market status"""
    return MarketHours.get_market_status()

def should_scan_now() -> Tuple[bool, str]:
    """Check if should scan now"""
    return MarketHours.should_scan_now()

def next_scan_time() -> datetime:
    """Get next scan time"""
    return MarketHours.next_scan_time()
