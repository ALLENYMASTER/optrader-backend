"""
Alternative Data Fetcher - Multiple Sources Strategy
Handles yfinance rate limiting with fallback mechanisms
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import asyncio
import aiohttp
import json
import time
import random
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)

# ==================== Data Classes ====================

@dataclass
class OptionsData:
    """Structured options data"""
    ticker: str
    current_price: float
    calls: pd.DataFrame
    puts: pd.DataFrame
    volume_ratio: float
    market_cap: float
    
# ==================== Proxy & Headers Strategy ====================

class SmartDataFetcher:
    """Smart fetcher with multiple strategies to avoid rate limiting"""
    
    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        ]
        self.session_pool = []
        self.last_request_time = {}
        self.min_delay = 2  # Minimum delay between requests
        self.max_retries = 3
        
        # Initialize session pool
        self._init_sessions()
        
    def _init_sessions(self):
        """Initialize multiple yfinance sessions with different settings"""
        for _ in range(3):
            session = self._create_session()
            self.session_pool.append(session)
    
    def _create_session(self):
        """Create a session with random headers"""
        import requests
        session = requests.Session()
        session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        return session
    
    def _get_session(self):
        """Get a random session from pool"""
        return random.choice(self.session_pool)
    
    def _rate_limit_wait(self, ticker: str):
        """Implement smart rate limiting per ticker"""
        current_time = time.time()
        
        if ticker in self.last_request_time:
            elapsed = current_time - self.last_request_time[ticker]
            if elapsed < self.min_delay:
                sleep_time = self.min_delay - elapsed + random.uniform(0.5, 1.5)
                time.sleep(sleep_time)
        
        self.last_request_time[ticker] = time.time()
    
    def fetch_with_retry(self, ticker: str, attempt: int = 0) -> Optional[OptionsData]:
        """Fetch data with exponential backoff retry"""
        if attempt >= self.max_retries:
            logger.error(f"❌ {ticker}: Max retries exceeded")
            return None
        
        try:
            # Rate limiting
            self._rate_limit_wait(ticker)
            
            # Use random session
            session = self._get_session()
            
            # Create ticker with session
            stock = yf.Ticker(ticker, session=session)
            
            # Try to get basic info first (less likely to be rate limited)
            info = stock.info
            
            if not info:
                raise ValueError("No info available")
            
            # Extract price with multiple fallbacks
            current_price = (
                info.get('currentPrice') or
                info.get('regularMarketPrice') or
                info.get('previousClose') or
                info.get('ask') or
                info.get('bid', 0)
            )
            
            if current_price <= 0:
                raise ValueError("Invalid price")
            
            # Get volume metrics
            volume = info.get('volume', 0)
            avg_volume = info.get('averageVolume', 1)
            market_cap = info.get('marketCap', 0)
            
            # Get options chain with delay
            time.sleep(random.uniform(1, 2))
            expirations = stock.options
            
            if not expirations:
                logger.info(f"ℹ️ {ticker}: No options available")
                return None
            
            # Collect limited options data
            all_calls = []
            all_puts = []
            
            # Only fetch nearest 3 expirations to minimize API calls
            for exp_date in expirations[:3]:
                try:
                    # Add delay between expiration fetches
                    time.sleep(random.uniform(0.5, 1))
                    
                    opt_chain = stock.option_chain(exp_date)
                    
                    calls = opt_chain.calls.copy()
                    calls['expiration'] = exp_date
                    calls['days_to_expiry'] = (
                        datetime.strptime(exp_date, '%Y-%m-%d') - datetime.now()
                    ).days
                    
                    puts = opt_chain.puts.copy()
                    puts['expiration'] = exp_date
                    puts['days_to_expiry'] = (
                        datetime.strptime(exp_date, '%Y-%m-%d') - datetime.now()
                    ).days
                    
                    all_calls.append(calls)
                    all_puts.append(puts)
                    
                except Exception as e:
                    logger.warning(f"⚠️ {ticker}: Failed to fetch {exp_date}: {e}")
                    continue
            
            if not all_calls:
                return None
            
            # Combine dataframes
            calls_df = pd.concat(all_calls, ignore_index=True)
            puts_df = pd.concat(all_puts, ignore_index=True)
            
            return OptionsData(
                ticker=ticker,
                current_price=current_price,
                calls=calls_df,
                puts=puts_df,
                volume_ratio=volume / avg_volume if avg_volume > 0 else 0,
                market_cap=market_cap
            )
            
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                # Rate limited - exponential backoff
                wait_time = (2 ** attempt) * 10 + random.uniform(5, 10)
                logger.warning(f"⏳ {ticker}: Rate limited. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                return self.fetch_with_retry(ticker, attempt + 1)
            else:
                logger.error(f"❌ {ticker}: Error - {e}")
                return None

# ==================== Batch Processing Strategy ====================

class BatchProcessor:
    """Process tickers in intelligent batches"""
    
    def __init__(self, fetcher: SmartDataFetcher):
        self.fetcher = fetcher
        self.batch_size = 2  # Process 2 tickers at a time
        self.batch_delay = 10  # Delay between batches
        
    async def process_tickers(self, tickers: List[str]) -> List[Dict]:
        """Process tickers in batches with delays"""
        results = []
        
        # Split into batches
        batches = [tickers[i:i + self.batch_size] 
                  for i in range(0, len(tickers), self.batch_size)]
        
        logger.info(f"📦 Processing {len(tickers)} tickers in {len(batches)} batches")
        
        for i, batch in enumerate(batches):
            logger.info(f"🔄 Processing batch {i+1}/{len(batches)}: {batch}")
            
            # Process batch in parallel using threads (yfinance is sync)
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
            
            # Delay between batches (except last)
            if i < len(batches) - 1:
                logger.info(f"⏸️ Batch delay: {self.batch_delay}s")
                await asyncio.sleep(self.batch_delay)
        
        return results
    
    async def _process_batch(self, batch: List[str]) -> List[Dict]:
        """Process a single batch of tickers"""
        loop = asyncio.get_event_loop()
        tasks = []
        
        for ticker in batch:
            # Run sync fetch in thread pool
            task = loop.run_in_executor(None, self._fetch_and_analyze, ticker)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    
    def _fetch_and_analyze(self, ticker: str) -> Optional[Dict]:
        """Fetch and analyze single ticker"""
        data = self.fetcher.fetch_with_retry(ticker)
        
        if data is None:
            return None
        
        # Analyze the data
        return self._analyze_options(data)
    
    def _analyze_options(self, data: OptionsData) -> Dict:
        """Analyze options data for signals"""
        
        # Filter for liquid options
        liquid_calls = data.calls[
            (data.calls['volume'] > 50) & 
            (data.calls['openInterest'] > 25)
        ].copy() if not data.calls.empty else pd.DataFrame()
        
        liquid_puts = data.puts[
            (data.puts['volume'] > 50) & 
            (data.puts['openInterest'] > 25)
        ].copy() if not data.puts.empty else pd.DataFrame()
        
        # Calculate metrics
        if not liquid_calls.empty:
            liquid_calls['volume_oi_ratio'] = liquid_calls['volume'] / (liquid_calls['openInterest'] + 1)
            liquid_calls['total_value'] = liquid_calls['volume'] * liquid_calls['lastPrice'] * 100
            liquid_calls['whale_score'] = (
                np.log1p(liquid_calls['total_value']) * 0.5 +
                liquid_calls['volume_oi_ratio'] * 50
            )
        
        if not liquid_puts.empty:
            liquid_puts['volume_oi_ratio'] = liquid_puts['volume'] / (liquid_puts['openInterest'] + 1)
            liquid_puts['total_value'] = liquid_puts['volume'] * liquid_puts['lastPrice'] * 100
            liquid_puts['whale_score'] = (
                np.log1p(liquid_puts['total_value']) * 0.5 +
                liquid_puts['volume_oi_ratio'] * 50
            )
        
        # Calculate sentiment
        total_call_value = liquid_calls['total_value'].sum() if not liquid_calls.empty else 0
        total_put_value = liquid_puts['total_value'].sum() if not liquid_puts.empty else 0
        total_value = total_call_value + total_put_value
        
        if total_value > 0:
            bullish_pct = (total_call_value / total_value) * 100
            bearish_pct = (total_put_value / total_value) * 100
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
            sentiment_score = 0
        
        # Find top unusual activity
        top_whale = None
        if not liquid_calls.empty or not liquid_puts.empty:
            all_options = pd.concat([liquid_calls, liquid_puts], ignore_index=True)
            if not all_options.empty and 'whale_score' in all_options.columns:
                top_whale = all_options.nlargest(1, 'whale_score').iloc[0]
        
        return {
            "ticker": data.ticker,
            "current_price": data.current_price,
            "volume_ratio": data.volume_ratio,
            "market_cap": data.market_cap,
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "bullish_percent": bullish_pct,
            "bearish_percent": bearish_pct,
            "total_calls_count": len(liquid_calls),
            "total_puts_count": len(liquid_puts),
            "total_calls_value": total_call_value,
            "total_puts_value": total_put_value,
            "whale_score": top_whale['whale_score'] if top_whale is not None else 0,
            "top_whale_strike": float(top_whale['strike']) if top_whale is not None else 0,
            "timestamp": datetime.now().isoformat()
        }

# ==================== Cache Layer ====================

class DataCache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self, ttl_seconds: int = 900):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[Dict]:
        """Get from cache if not expired"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, data: Dict):
        """Set cache with timestamp"""
        self.cache[key] = (data, time.time())
    
    def clear_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired = [k for k, (_, t) in self.cache.items() 
                  if current_time - t >= self.ttl]
        for key in expired:
            del self.cache[key]

# ==================== Main Scanner ====================

class EnhancedOptionsScanner:
    """Main scanner with all strategies combined"""
    
    def __init__(self):
        self.fetcher = SmartDataFetcher()
        self.processor = BatchProcessor(self.fetcher)
        self.cache = DataCache()
        
    async def scan_market(self, tickers: List[str], use_cache: bool = True) -> Dict:
        """Scan market with smart strategies"""
        
        # Check cache first
        if use_cache:
            cached_results = []
            uncached_tickers = []
            
            for ticker in tickers:
                cached = self.cache.get(ticker)
                if cached:
                    cached_results.append(cached)
                else:
                    uncached_tickers.append(ticker)
            
            if cached_results and not uncached_tickers:
                logger.info(f"✅ All {len(cached_results)} results from cache")
                return {
                    "results": cached_results,
                    "from_cache": True,
                    "total": len(cached_results)
                }
            
            tickers = uncached_tickers
            logger.info(f"📊 Fetching {len(tickers)} tickers (cached: {len(cached_results)})")
        else:
            cached_results = []
        
        # Process uncached tickers
        results = await self.processor.process_tickers(tickers)
        
        # Update cache
        for result in results:
            self.cache.set(result['ticker'], result)
        
        # Combine with cached results
        all_results = cached_results + results
        
        # Sort by whale score
        all_results.sort(key=lambda x: x.get('whale_score', 0), reverse=True)
        
        # Clear expired cache entries
        self.cache.clear_expired()
        
        return {
            "results": all_results,
            "from_cache": use_cache and len(cached_results) > 0,
            "total": len(all_results),
            "cached": len(cached_results),
            "fetched": len(results)
        }

# ==================== Usage Example ====================

async def example_usage():
    """Example of how to use the enhanced scanner"""
    
    # Initialize scanner
    scanner = EnhancedOptionsScanner()
    
    # Define tickers
    tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "TSM", "META", "GOOGL"]
    
    # Scan market
    print("🚀 Starting market scan...")
    results = await scanner.scan_market(tickers)
    
    print(f"\n📊 Scan Results:")
    print(f"Total: {results['total']}")
    print(f"From Cache: {results['from_cache']}")
    
    # Display top movers
    for result in results['results'][:3]:
        print(f"\n{result['ticker']}:")
        print(f"  Price: ${result['current_price']:.2f}")
        print(f"  Sentiment: {result['sentiment']} ({result['sentiment_score']:.1f}%)")
        print(f"  Whale Score: {result['whale_score']:.1f}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())

