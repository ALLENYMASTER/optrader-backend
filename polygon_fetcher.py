"""
Polygon.io Data Fetcher
"""

from polygon import RESTClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import time

logger = logging.getLogger(__name__)

class PolygonDataFetcher:
    """使用 Polygon.io API 獲取股票和期權數據"""
    
    def __init__(self, api_key: str):
        """
        初始化 Polygon.io 客戶端
        
        Args:
            api_key: Polygon.io API key
        """
        if not api_key:
            raise ValueError("Polygon.io API key is required")
            
        self.client = RESTClient(api_key)
        self.api_key = api_key
        logger.info("✅ Polygon.io client initialized")
        
    # ==================== Stock Quote ====================
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
    def get_stock_quote(self, ticker: str) -> Optional[Dict]:
        """
        獲取股票當前報價
        
        Args:
            ticker: 股票代碼 (e.g., 'AAPL')
            
        Returns:
            {
                'ticker': 'AAPL',
                'current_price': 180.50,
                'bid': 180.45,
                'ask': 180.55,
                'volume': 50000000,
                'prev_close': 179.80,
                'market_cap': 2800000000000,
                'timestamp': '2025-10-31T...'
            }
        """
        try:
            logger.info(f"📊 Fetching quote for {ticker}")
            
            # 獲取最新報價
            quote = self.client.get_last_quote(ticker)
            
            if not quote:
                logger.warning(f"⚠️ {ticker}: No quote data")
                return None
            
            # 獲取前一交易日數據
            prev_close_data = self.client.get_previous_close_agg(ticker)
            
            # 獲取公司詳情
            try:
                details = self.client.get_ticker_details(ticker)
                market_cap = getattr(details, 'market_cap', 0)
            except:
                market_cap = 0
            
            # 提取價格
            current_price = quote.ask_price if quote.ask_price and quote.ask_price > 0 else quote.bid_price
            
            if not current_price or current_price <= 0:
                logger.warning(f"⚠️ {ticker}: Invalid price")
                return None
            
            result = {
                'ticker': ticker,
                'current_price': float(current_price),
                'bid': float(quote.bid_price) if quote.bid_price else 0,
                'ask': float(quote.ask_price) if quote.ask_price else 0,
                'volume': int(prev_close_data[0].volume) if prev_close_data else 0,
                'prev_close': float(prev_close_data[0].close) if prev_close_data else 0,
                'market_cap': float(market_cap),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ {ticker}: ${result['current_price']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ {ticker}: Quote fetch failed - {e}")
            return None
    
    # ==================== Historical Data ====================
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
    def get_aggregate_bars(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """
        獲取歷史K線數據（用於計算平均成交量）
        
        Args:
            ticker: 股票代碼
            days: 回溯天數
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"📈 Fetching {days} days bars for {ticker}")
            
            aggs = self.client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=from_date,
                to=to_date,
                limit=days
            )
            
            if not aggs:
                logger.warning(f"⚠️ {ticker}: No historical data")
                return pd.DataFrame()
            
            data = []
            for agg in aggs:
                data.append({
                    'date': datetime.fromtimestamp(agg.timestamp / 1000),
                    'open': float(agg.open),
                    'high': float(agg.high),
                    'low': float(agg.low),
                    'close': float(agg.close),
                    'volume': int(agg.volume)
                })
            
            df = pd.DataFrame(data)
            logger.info(f"✅ {ticker}: {len(df)} bars retrieved")
            return df
            
        except Exception as e:
            logger.error(f"❌ {ticker}: Historical data failed - {e}")
            return pd.DataFrame()
    
    # ==================== Options Chain ====================
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
    def get_options_chain(self, ticker: str, expiration_date: str = None) -> pd.DataFrame:
        """
        獲取期權鏈數據
        
        Args:
            ticker: 股票代碼
            expiration_date: 到期日 (YYYY-MM-DD)，None = 獲取最近到期日
            
        Returns:
            DataFrame with columns: ticker, strike, expiration, option_type, lastPrice,
                                   bid, ask, volume, openInterest, impliedVolatility, etc.
        """
        try:
            logger.info(f"🔗 Fetching options chain for {ticker}")
            
            # 獲取期權合約列表
            contracts = list(self.client.list_options_contracts(
                underlying_ticker=ticker,
                expiration_date=expiration_date,
                limit=1000
            ))
            
            if not contracts:
                logger.warning(f"⚠️ {ticker}: No options contracts")
                return pd.DataFrame()
            
            logger.info(f"📋 {ticker}: Found {len(contracts)} contracts")
            
            # 收集期權數據
            options_data = []
            processed = 0
            
            for contract in contracts:
                try:
                    option_ticker = contract.ticker
                    
                    # 使用 snapshot 獲取實時數據
                    snapshot = self.client.get_snapshot_option(
                        underlyingAsset=ticker,
                        optionContract=option_ticker
                    )
                    
                    if not snapshot:
                        continue
                    
                    # 提取數據
                    details = snapshot.details
                    greeks = getattr(snapshot, 'greeks', None)
                    last_quote = getattr(snapshot, 'last_quote', None)
                    day_info = getattr(snapshot, 'day', None)
                    
                    # 組裝數據
                    option_data = {
                        'ticker': option_ticker,
                        'underlying': ticker,
                        'strike': float(details.strike_price),
                        'expiration': details.expiration_date,
                        'option_type': details.contract_type.lower(),  # 'call' or 'put'
                        'lastPrice': float(last_quote.ask) if last_quote and last_quote.ask else 0,
                        'bid': float(last_quote.bid) if last_quote and last_quote.bid else 0,
                        'ask': float(last_quote.ask) if last_quote and last_quote.ask else 0,
                        'volume': int(day_info.volume) if day_info and hasattr(day_info, 'volume') else 0,
                        'openInterest': int(day_info.open_interest) if day_info and hasattr(day_info, 'open_interest') else 0,
                        'impliedVolatility': float(greeks.implied_volatility) if greeks and hasattr(greeks, 'implied_volatility') else 0,
                        'delta': float(greeks.delta) if greeks and hasattr(greeks, 'delta') else 0,
                        'gamma': float(greeks.gamma) if greeks and hasattr(greeks, 'gamma') else 0,
                        'theta': float(greeks.theta) if greeks and hasattr(greeks, 'theta') else 0,
                        'vega': float(greeks.vega) if greeks and hasattr(greeks, 'vega') else 0
                    }
                    
                    options_data.append(option_data)
                    processed += 1
                    
                    # 每處理50個合約休息一下
                    if processed % 50 == 0:
                        time.sleep(0.5)
                    
                except Exception as e:
                    logger.debug(f"⚠️ Skipping contract {option_ticker}: {e}")
                    continue
            
            if not options_data:
                logger.warning(f"⚠️ {ticker}: No valid options data")
                return pd.DataFrame()
            
            df = pd.DataFrame(options_data)
            
            # 計算到期天數
            df['days_to_expiry'] = df['expiration'].apply(
                lambda x: (datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days
            )
            
            # 添加 'type' 列以匹配原有代碼
            df['type'] = df['option_type']
            
            logger.info(f"✅ {ticker}: {len(df)} options loaded")
            return df
            
        except Exception as e:
            logger.error(f"❌ {ticker}: Options chain failed - {e}")
            return pd.DataFrame()
    
    # ==================== Expiration Dates ====================
    
    def get_near_term_expirations(self, ticker: str, max_days: int = 90) -> List[str]:
        """
        獲取近期到期日列表
        
        Args:
            ticker: 股票代碼
            max_days: 最大天數
            
        Returns:
            到期日列表 ['2025-11-15', '2025-11-22', ...]
        """
        try:
            cutoff_date = (datetime.now() + timedelta(days=max_days)).strftime('%Y-%m-%d')
            
            logger.info(f"📅 Fetching expirations for {ticker} (max {max_days} days)")
            
            contracts = list(self.client.list_options_contracts(
                underlying_ticker=ticker,
                expiration_date_lte=cutoff_date,
                limit=100
            ))
            
            if not contracts:
                logger.warning(f"⚠️ {ticker}: No expirations found")
                return []
            
            # 提取唯一的到期日並排序
            expirations = sorted(set(c.expiration_date for c in contracts))
            
            logger.info(f"✅ {ticker}: {len(expirations)} expirations found")
            return expirations[:6]  # 返回前6個到期日
            
        except Exception as e:
            logger.error(f"❌ {ticker}: Expiration fetch failed - {e}")
            return []
    
    # ==================== Async Wrapper ====================
    
    async def fetch_ticker_data_async(self, ticker: str) -> Optional[Dict]:
        """
        異步獲取完整的股票和期權數據
        
        Args:
            ticker: 股票代碼
            
        Returns:
            {
                'ticker': 'AAPL',
                'quote': {...},
                'volume_ratio': 1.5,
                'options': DataFrame
            }
        """
        loop = asyncio.get_event_loop()
        
        try:
            # 1. 獲取股票報價
            quote = await loop.run_in_executor(None, self.get_stock_quote, ticker)
            
            if not quote:
                return None
            
            # 2. 獲取歷史數據計算平均成交量
            bars = await loop.run_in_executor(None, self.get_aggregate_bars, ticker, 30)
            avg_volume = bars['volume'].mean() if not bars.empty else 1
            
            # 3. 計算成交量比率
            volume_ratio = quote['volume'] / avg_volume if avg_volume > 0 else 0
            
            # 4. 獲取近期到期日
            expirations = await loop.run_in_executor(
                None,
                self.get_near_term_expirations,
                ticker,
                90
            )
            
            if not expirations:
                logger.info(f"ℹ️ {ticker}: No options available")
                return None
            
            # 5. 獲取期權數據（只取前3個到期日以節省API調用）
            all_options = []
            for i, exp_date in enumerate(expirations[:3]):
                logger.info(f"📊 {ticker}: Fetching options for {exp_date} ({i+1}/3)")
                
                options_df = await loop.run_in_executor(
                    None,
                    self.get_options_chain,
                    ticker,
                    exp_date
                )
                
                if not options_df.empty:
                    all_options.append(options_df)
                
                # Rate limiting: 每個到期日之間間隔
                if i < len(expirations[:3]) - 1:
                    await asyncio.sleep(1)
            
            if not all_options:
                logger.warning(f"⚠️ {ticker}: No options data retrieved")
                return None
            
            # 6. 合併所有期權數據
            options_df = pd.concat(all_options, ignore_index=True)
            
            logger.info(f"✅ {ticker}: Complete data fetched")
            
            return {
                'ticker': ticker,
                'quote': quote,
                'volume_ratio': volume_ratio,
                'options': options_df
            }
            
        except Exception as e:
            logger.error(f"❌ {ticker}: Async fetch failed - {e}")
            return None


# ==================== 輔助函數 ====================

def filter_liquid_options(options_df: pd.DataFrame,
                         min_volume: int = 100,
                         min_oi: int = 50) -> pd.DataFrame:
    """
    過濾流動性好的期權
    
    Args:
        options_df: 期權數據
        min_volume: 最小成交量
        min_oi: 最小未平倉合約
        
    Returns:
        過濾後的 DataFrame
    """
    if options_df.empty:
        return options_df
    
    return options_df[
        (options_df['volume'] > min_volume) &
        (options_df['openInterest'] > min_oi)
    ].copy()


# ==================== 測試代碼 ====================

async def test_polygon_fetcher():
    """測試 Polygon.io 數據獲取功能"""
    
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        print("❌ Please set POLYGON_API_KEY environment variable")
        print("Get free key at: https://polygon.io/dashboard/signup")
        return
    
    print("🚀 Testing Polygon.io Data Fetcher\n")
    
    fetcher = PolygonDataFetcher(api_key)
    
    # 測試股票報價
    print("=" * 50)
    print("📊 Testing Stock Quote...")
    print("=" * 50)
    quote = fetcher.get_stock_quote("AAPL")
    if quote:
        print(f"✅ AAPL Quote:")
        print(f"   Price: ${quote['current_price']:.2f}")
        print(f"   Volume: {quote['volume']:,}")
        print(f"   Market Cap: ${quote['market_cap']:,.0f}")
    else:
        print("❌ Quote fetch failed")
    
    # 測試歷史數據
    print("\n" + "=" * 50)
    print("📈 Testing Historical Data...")
    print("=" * 50)
    bars = fetcher.get_aggregate_bars("AAPL", 5)
    if not bars.empty:
        print(f"✅ AAPL Historical Data:")
        print(f"   Bars: {len(bars)}")
        print(f"   Avg Volume: {bars['volume'].mean():,.0f}")
    else:
        print("❌ Historical data fetch failed")
    
    # 測試到期日
    print("\n" + "=" * 50)
    print("📅 Testing Expirations...")
    print("=" * 50)
    expirations = fetcher.get_near_term_expirations("AAPL")
    if expirations:
        print(f"✅ AAPL Expirations:")
        for exp in expirations[:3]:
            print(f"   {exp}")
    else:
        print("❌ Expirations fetch failed")
    
    # 測試期權鏈
    print("\n" + "=" * 50)
    print("🔗 Testing Options Chain...")
    print("=" * 50)
    if expirations:
        options = fetcher.get_options_chain("AAPL", expirations[0])
        if not options.empty:
            print(f"✅ AAPL Options Chain:")
            print(f"   Total Contracts: {len(options)}")
            liquid = filter_liquid_options(options)
            print(f"   Liquid Contracts: {len(liquid)}")
            print(f"   Calls: {len(options[options['type'] == 'call'])}")
            print(f"   Puts: {len(options[options['type'] == 'put'])}")
        else:
            print("❌ Options chain fetch failed")
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    # 設置日誌級別
    logging.basicConfig(level=logging.INFO)
    
    # 運行測試
    asyncio.run(test_polygon_fetcher())
