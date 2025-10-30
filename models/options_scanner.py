"""
Enhanced Options Whale Scanner - ROBUST VERSION
Handles yfinance failures with multiple fallback methods
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Any
import warnings
import time
import traceback
warnings.filterwarnings('ignore')

class OptionsWhaleScanner:
    """Scanner with robust error handling and fallback methods"""
    
    def __init__(self):
        self.results = []
        # Create a session with custom headers
        import requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scan_single_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Scan a single ticker - ALWAYS returns data
        """
        try:
            options_data = self._get_options_data_safe(ticker)
            
            if options_data and options_data.get('current_price', 0) > 0:
                return self._identify_whale_activity(options_data)
            else:
                # Return minimal valid data structure
                return self._get_minimal_response(ticker)
                
        except Exception as e:
            print(f"⚠️ Error scanning {ticker}: {e}")
            return self._get_minimal_response(ticker)
    
    def _get_minimal_response(self, ticker: str) -> Dict[str, Any]:
        """Return minimal valid response when data unavailable"""
        # Try to at least get the current price
        price = self._get_price_fallback(ticker)
        
        return {
            'ticker': ticker,
            'current_price': price,
            'volume_ratio': 0,
            'market_cap': 0,
            'sentiment': 'DATA_LIMITED' if price > 0 else 'NO_DATA',
            'sentiment_score': 0,
            'bullish_percent': 50,
            'bearish_percent': 50,
            'total_calls_count': 0,
            'total_puts_count': 0,
            'total_calls_value': 0,
            'total_puts_value': 0,
            'whale_score': 0,
            'volatility_plays': 0,
            'directional_bets': 0,
            'unusual_activity': [],
            'expected_move': None,
            'recommended_strategies': [{
                'strategy': 'LIMITED DATA - MANUAL RESEARCH RECOMMENDED',
                'confidence': 'LOW',
                'rationale': 'Unable to fetch complete options data. Check yfinance or try again later.',
                'suggested_strikes': 'Visit broker platform for current options',
                'risk_level': 'UNKNOWN',
                'profit_target': 'N/A',
                'stop_loss': 'N/A', 
                'time_horizon': 'N/A'
            }],
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_price_fallback(self, ticker: str) -> float:
        """Try multiple methods to get current price"""
        try:
            # Method 1: Download function (most reliable)
            data = yf.download(ticker, period="1d", progress=False, threads=False)
            if not data.empty and 'Close' in data.columns:
                return float(data['Close'].iloc[-1])
        except:
            pass
        
        try:
            # Method 2: Ticker with session
            stock = yf.Ticker(ticker, session=self.session)
            hist = stock.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except:
            pass
        
        try:
            # Method 3: Fast info
            stock = yf.Ticker(ticker)
            fast_info = stock.fast_info
            if hasattr(fast_info, 'lastPrice'):
                return float(fast_info.lastPrice)
        except:
            pass
        
        return 0  # Unable to get price
    
    def _get_options_data_safe(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Safely fetch options data with multiple fallback methods"""
        
        # Add initial delay to avoid rate limiting
        time.sleep(0.5)
        
        try:
            # Use session for better reliability
            stock = yf.Ticker(ticker, session=self.session)
            
            # Get price first (most important)
            current_price = 0
            volume = 0
            avg_volume = 1
            market_cap = 0
            
            # Try multiple methods for price
            try:
                # Method 1: Regular info
                info = stock.info
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                volume = info.get('volume', 0)
                avg_volume = info.get('averageVolume', 1)
                market_cap = info.get('marketCap', 0)
            except:
                pass
            
            if current_price == 0:
                # Method 2: History
                try:
                    hist = stock.history(period="5d")
                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                        volume = int(hist['Volume'].iloc[-1])
                        avg_volume = int(hist['Volume'].mean())
                except:
                    pass
            
            if current_price == 0:
                # Method 3: Download
                try:
                    data = yf.download(ticker, period="5d", progress=False, threads=False)
                    if not data.empty:
                        current_price = float(data['Close'].iloc[-1])
                        volume = int(data['Volume'].iloc[-1])
                        avg_volume = int(data['Volume'].mean())
                except:
                    pass
            
            if current_price == 0:
                print(f"⚠️ Unable to get price for {ticker}")
                return None
            
            # Now try to get options data (but don't fail if unavailable)
            calls_df = pd.DataFrame()
            puts_df = pd.DataFrame()
            
            try:
                time.sleep(0.5)  # Rate limit protection
                expirations = stock.options
                
                if expirations and len(expirations) > 0:
                    # Get just the first expiration to minimize API calls
                    exp_date = expirations[0]
                    
                    try:
                        time.sleep(0.5)
                        opt_chain = stock.option_chain(exp_date)
                        
                        calls_df = opt_chain.calls.copy()
                        calls_df['expiration'] = exp_date
                        calls_df['type'] = 'call'
                        
                        puts_df = opt_chain.puts.copy()
                        puts_df['expiration'] = exp_date
                        puts_df['type'] = 'put'
                        
                    except Exception as e:
                        print(f"⚠️ Could not get options for {ticker}: {e}")
            except Exception as e:
                print(f"⚠️ No options available for {ticker}: {e}")
            
            # Return what we have
            return {
                'ticker': ticker,
                'current_price': current_price,
                'volume': volume,
                'avg_volume': avg_volume,
                'volume_ratio': volume / avg_volume if avg_volume > 0 else 0,
                'market_cap': market_cap,
                'calls': calls_df,
                'puts': puts_df
            }
            
        except Exception as e:
            print(f"❌ Failed to get data for {ticker}: {e}")
            return None
    
    def scan_all_sequential(self, tickers: list) -> list:
        """Scan tickers sequentially - more reliable"""
        results = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] Scanning {ticker}...")
            result = self.scan_single_ticker(ticker)
            results.append(result)  # Always append result
            time.sleep(1)  # Add delay between tickers
        
        return results
    
    def scan_all_parallel(self, tickers: list, max_workers: int = 2) -> list:
        """Scan tickers in parallel - use sequential for now due to rate limits"""
        # For now, just use sequential to avoid rate limiting issues
        return self.scan_all_sequential(tickers)
    
    def _identify_whale_activity(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process options data - always returns valid structure
        """
        ticker = options_data['ticker']
        current_price = options_data['current_price']
        calls = options_data.get('calls', pd.DataFrame())
        puts = options_data.get('puts', pd.DataFrame())
        
        # Calculate basic metrics
        total_calls_vol = 0
        total_puts_vol = 0
        
        if not calls.empty and 'volume' in calls.columns:
            total_calls_vol = calls['volume'].sum()
        
        if not puts.empty and 'volume' in puts.columns:
            total_puts_vol = puts['volume'].sum()
        
        total_vol = total_calls_vol + total_puts_vol
        
        # Calculate sentiment
        if total_vol > 0:
            bullish_percent = (total_calls_vol / total_vol * 100)
            bearish_percent = (total_puts_vol / total_vol * 100)
        else:
            bullish_percent = 50
            bearish_percent = 50
        
        # Determine sentiment
        net_sentiment = bullish_percent - bearish_percent
        
        if net_sentiment > 20:
            sentiment = "BULLISH"
            sentiment_score = min(100, 50 + net_sentiment/2)
        elif net_sentiment < -20:
            sentiment = "BEARISH"
            sentiment_score = max(-100, -50 + net_sentiment/2)
        else:
            sentiment = "NEUTRAL"
            sentiment_score = net_sentiment/2
        
        # Process unusual activity if data available
        unusual_list = []
        if not calls.empty and not puts.empty:
            try:
                all_options = pd.concat([calls, puts], ignore_index=True)
                
                # Look for high volume options
                if 'volume' in all_options.columns and 'openInterest' in all_options.columns:
                    all_options['vol_oi_ratio'] = all_options['volume'] / all_options['openInterest'].replace(0, 1)
                    
                    # Get top 5 by volume
                    top_options = all_options.nlargest(5, 'volume')
                    
                    for _, row in top_options.iterrows():
                        if row['volume'] > 0:
                            unusual_list.append({
                                'option_type': row['type'].upper(),
                                'strike': float(row['strike']),
                                'expiration': row.get('expiration', 'N/A'),
                                'volume': int(row['volume']),
                                'openInterest': int(row.get('openInterest', 0)),
                                'vol_oi_ratio': float(row.get('vol_oi_ratio', 0)),
                                'lastPrice': float(row.get('lastPrice', 0)),
                                'distance_from_price': ((row['strike'] - current_price) / current_price * 100)
                            })
            except Exception as e:
                print(f"⚠️ Error processing options for {ticker}: {e}")
        
        # Generate basic strategy
        strategies = []
        if sentiment == "BULLISH":
            strategies.append({
                'strategy': 'BULLISH MOMENTUM - CONSIDER CALLS',
                'confidence': 'MEDIUM',
                'rationale': f'Call volume dominates ({bullish_percent:.0f}% bullish)',
                'suggested_strikes': f'ATM or slightly OTM calls around ${current_price:.0f}',
                'risk_level': 'MEDIUM',
                'profit_target': '+5-10%',
                'stop_loss': '-30% of premium',
                'time_horizon': '1-4 weeks'
            })
        elif sentiment == "BEARISH":
            strategies.append({
                'strategy': 'BEARISH PRESSURE - CONSIDER PUTS',
                'confidence': 'MEDIUM',
                'rationale': f'Put volume dominates ({bearish_percent:.0f}% bearish)',
                'suggested_strikes': f'ATM or slightly OTM puts around ${current_price:.0f}',
                'risk_level': 'MEDIUM',
                'profit_target': '-5-10%',
                'stop_loss': '-30% of premium',
                'time_horizon': '1-4 weeks'
            })
        else:
            strategies.append({
                'strategy': 'NEUTRAL MARKET - RANGE-BOUND STRATEGIES',
                'confidence': 'LOW',
                'rationale': f'Balanced call/put activity ({bullish_percent:.0f}%/{bearish_percent:.0f}%)',
                'suggested_strikes': 'Consider iron condors or calendar spreads',
                'risk_level': 'LOW-MEDIUM',
                'profit_target': 'Collect premium',
                'stop_loss': 'Exit if price breaks range',
                'time_horizon': '1-2 weeks'
            })
        
        return {
            'ticker': ticker,
            'current_price': current_price,
            'volume_ratio': options_data.get('volume_ratio', 0),
            'market_cap': options_data.get('market_cap', 0),
            'sentiment': sentiment,
            'sentiment_score': float(sentiment_score),
            'bullish_percent': float(bullish_percent),
            'bearish_percent': float(bearish_percent),
            'total_calls_count': int(total_calls_vol),
            'total_puts_count': int(total_puts_vol),
            'total_calls_value': 0,
            'total_puts_value': 0,
            'whale_score': len(unusual_list) * 10,  # Simple score based on activity
            'volatility_plays': 0,
            'directional_bets': len(unusual_list),
            'unusual_activity': unusual_list,
            'expected_move': None,
            'recommended_strategies': strategies,
            'timestamp': datetime.now().isoformat()
        }
