"""
Enhanced Options Whale Scanner - FIXED VERSION
Now returns data even when no unusual activity is detected
With more lenient whale detection thresholds
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class OptionsWhaleScanner:
    """Enhanced scanner with complete feature set from us_whale_web.py"""
    
    def __init__(self):
        self.results = []
    
    def scan_single_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Scan a single ticker for whale activity
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with comprehensive analysis results or None
        """
        try:
            options_data = self._get_options_data(ticker)
            if not options_data:
                return None
            
            whale_activity = self._identify_whale_activity(options_data)
            return whale_activity
            
        except Exception as e:
            print(f"❌ Error scanning {ticker}: {e}")
            return None
    
    def scan_all_sequential(self, tickers: list) -> list:
        """Scan tickers sequentially"""
        results = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] Scanning {ticker}...")
            result = self.scan_single_ticker(ticker)
            if result:
                results.append(result)
        
        return results
    
    def scan_all_parallel(self, tickers: list, max_workers: int = 2) -> list:
        """Scan tickers in parallel with aggressive rate limiting"""
        results = []
        
        # CRITICAL: Reduce to max 2 concurrent requests to avoid 429
        safe_workers = min(max_workers, 2)
        
        with ThreadPoolExecutor(max_workers=safe_workers) as executor:
            future_to_ticker = {}
            
            # Submit with longer delay to avoid rate limiting
            for i, ticker in enumerate(tickers):
                # Add 500ms delay between submissions (increased from 200ms)
                if i > 0:
                    import time
                    time.sleep(0.5)
                
                future = executor.submit(self.scan_single_ticker, ticker)
                future_to_ticker[future] = ticker
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        unusual_count = len(result.get('unusual_activity', []))
                        print(f"✅ {ticker}: {unusual_count} unusual trades")
                except Exception as e:
                    print(f"❌ {ticker}: {e}")
        
        return results
    
    def _get_options_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive options data from yfinance with aggressive retry logic"""
        import time
        
        max_retries = 3
        base_retry_delay = 3  # Increased from 2 seconds
        
        for attempt in range(max_retries):
            try:
                # Add longer delay to avoid rate limiting
                if attempt > 0:
                    delay = base_retry_delay * (attempt + 1)
                    print(f"⏳ Retrying {ticker} after {delay}s delay...")
                    time.sleep(delay)
                
                # Initial delay before any request
                time.sleep(0.5)
                
                stock = yf.Ticker(ticker)
                info = stock.info
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                
                if current_price == 0:
                    return None
                
                # Get additional market metrics
                volume = info.get('volume', 0)
                avg_volume = info.get('averageVolume', 1)
                market_cap = info.get('marketCap', 0)
                
                # Add longer delay before options request
                time.sleep(0.5)
                
                expirations = stock.options
                if not expirations:
                    return None
                
                all_calls = []
                all_puts = []
                cutoff_date = datetime.now() + timedelta(days=90)
                
                # Limit to first 4 expirations (reduced from 6)
                for exp_date in expirations[:4]:
                    exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                    if exp_datetime > cutoff_date:
                        continue
                    
                    # Add longer delay between expiration fetches
                    time.sleep(0.5)  # Increased from 0.2s
                    
                    opt_chain = stock.option_chain(exp_date)
                    
                    calls = opt_chain.calls.copy()
                    calls['expiration'] = exp_date
                    calls['type'] = 'call'
                    calls['days_to_expiry'] = (exp_datetime - datetime.now()).days
                    all_calls.append(calls)
                    
                    puts = opt_chain.puts.copy()
                    puts['expiration'] = exp_date
                    puts['type'] = 'put'
                    puts['days_to_expiry'] = (exp_datetime - datetime.now()).days
                    all_puts.append(puts)
                
                calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
                puts_df = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()
                
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
                error_msg = str(e)
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    print(f"⚠️ Rate limited on {ticker}, attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        # Exponential backoff with longer delays
                        backoff_delay = base_retry_delay * (2 ** attempt)
                        print(f"⏳ Backing off for {backoff_delay}s...")
                        time.sleep(backoff_delay)
                elif attempt < max_retries - 1:
                    print(f"⚠️ Error fetching {ticker} data: {e}")
        
        return None
    
    def _identify_whale_activity(self, options_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Identify whale activity with dynamic thresholds
        FIXED: Always returns data, even when no whale activity detected
        """
        ticker = options_data['ticker']
        current_price = options_data['current_price']
        calls = options_data['calls']
        puts = options_data['puts']
        
        # Initialize default response structure
        default_response = {
            'ticker': ticker,
            'current_price': current_price,
            'volume_ratio': options_data.get('volume_ratio', 0),
            'market_cap': options_data.get('market_cap', 0),
            'sentiment': 'NEUTRAL',
            'sentiment_score': 0,
            'bullish_percent': 0,
            'bearish_percent': 0,
            'total_calls_count': 0,
            'total_puts_count': 0,
            'total_calls_value': 0,
            'total_puts_value': 0,
            'whale_score': 0,
            'volatility_plays': 0,
            'directional_bets': 0,
            'unusual_activity': [],
            'expected_move': None,
            'recommended_strategies': [],
            'timestamp': datetime.now().isoformat()
        }
        
        if calls.empty and puts.empty:
            print(f"ℹ️ {ticker}: No options data available")
            return default_response
        
        # Combine calls and puts
        all_options = pd.concat([calls, puts], ignore_index=True) if not calls.empty and not puts.empty else (calls if not puts.empty else puts)
        
        # Feature engineering
        all_options['option_type'] = all_options['type'].str.upper()
        all_options['moneyness'] = all_options.apply(
            lambda x: (x['strike'] - current_price) / current_price if x['option_type'] == 'CALL'
            else (current_price - x['strike']) / current_price, axis=1
        )
        all_options['dollar_volume'] = all_options['lastPrice'] * all_options['volume'] * 100
        all_options['notional_value'] = all_options['strike'] * all_options['volume'] * 100
        all_options['oi_value'] = all_options['lastPrice'] * all_options['openInterest'] * 100
        all_options['distance_from_price'] = ((all_options['strike'] - current_price) / current_price) * 100
        
        # More lenient whale detection criteria (reduced thresholds)
        # Old: volume > 100 and vol/oi > 0.3
        # New: volume > 50 and vol/oi > 0.2
        volume_threshold = 50  # Reduced from 100
        vol_oi_ratio_threshold = 0.2  # Reduced from 0.3
        
        all_options['vol_oi_ratio'] = all_options['volume'] / all_options['openInterest'].replace(0, 1)
        all_options['is_unusual'] = (
            (all_options['volume'] > volume_threshold) & 
            (all_options['vol_oi_ratio'] > vol_oi_ratio_threshold)
        )
        
        # If still no unusual activity with lenient criteria, get top 10 by volume
        unusual_df = all_options[all_options['is_unusual']].copy()
        
        if unusual_df.empty:
            # Get top 10 most active options as "notable" activity
            notable_df = all_options.nlargest(10, 'volume').copy()
            if not notable_df.empty:
                unusual_df = notable_df
                print(f"ℹ️ {ticker}: No whale activity detected, showing top {len(notable_df)} active options")
        
        # Calculate metrics even with limited data
        total_calls_vol = calls['volume'].sum() if not calls.empty else 0
        total_puts_vol = puts['volume'].sum() if not puts.empty else 0
        total_vol = total_calls_vol + total_puts_vol
        
        bullish_percent = (total_calls_vol / total_vol * 100) if total_vol > 0 else 50
        bearish_percent = (total_puts_vol / total_vol * 100) if total_vol > 0 else 50
        
        # Sentiment calculation
        net_sentiment = bullish_percent - bearish_percent
        
        if net_sentiment > 30:
            sentiment = "STRONG BULLISH"
            sentiment_score = min(100, 50 + net_sentiment)
        elif net_sentiment > 10:
            sentiment = "BULLISH"
            sentiment_score = 50 + (net_sentiment / 2)
        elif net_sentiment < -30:
            sentiment = "STRONG BEARISH"
            sentiment_score = max(-100, -50 + net_sentiment)
        elif net_sentiment < -10:
            sentiment = "BEARISH"
            sentiment_score = -50 + (net_sentiment / 2)
        else:
            sentiment = "NEUTRAL"
            sentiment_score = net_sentiment
        
        # Calculate whale scores and strategy types
        if not unusual_df.empty:
            # Whale score calculation
            unusual_df['whale_score'] = (
                unusual_df['dollar_volume'] / 10000 * 
                unusual_df['vol_oi_ratio'] * 
                (1 + abs(unusual_df['moneyness']))
            )
            
            # Strategy type detection
            unusual_df['strategy_type'] = unusual_df.apply(
                lambda x: 'DIRECTIONAL' if abs(x['moneyness']) < 0.10
                else 'VOLATILITY' if 0.10 <= abs(x['moneyness']) < 0.20
                else 'HEDGE' if abs(x['moneyness']) >= 0.20
                else 'UNKNOWN', axis=1
            )
            
            volatility_plays = len(unusual_df[unusual_df['strategy_type'] == 'VOLATILITY'])
            directional_bets = len(unusual_df[unusual_df['strategy_type'] == 'DIRECTIONAL'])
            
            # Sort by whale score
            unusual_df = unusual_df.sort_values('whale_score', ascending=False)
            total_whale_score = unusual_df['whale_score'].sum()
        else:
            volatility_plays = 0
            directional_bets = 0
            total_whale_score = 0
        
        # Calculate expected move
        weekly_options = self._get_weekly_options(calls, puts, current_price)
        expected_move_data = None
        if weekly_options:
            expected_move_data = self._calculate_expected_move(
                weekly_options['calls'], 
                weekly_options['puts'], 
                current_price
            )
        
        # Generate strategy recommendations
        recommended_strategies = self._generate_strategy_recommendations(
            unusual_df, sentiment_score, bullish_percent, bearish_percent,
            volatility_plays, directional_bets, expected_move_data, current_price
        )
        
        # Convert unusual activity to list of dicts
        unusual_list = []
        if not unusual_df.empty:
            for _, row in unusual_df.head(20).iterrows():  # Top 20 unusual trades
                unusual_list.append({
                    'option_type': row['option_type'],
                    'strike': float(row['strike']),
                    'expiration': row['expiration'],
                    'days_to_expiry': int(row['days_to_expiry']),
                    'volume': int(row['volume']),
                    'openInterest': int(row['openInterest']),
                    'vol_oi_ratio': float(row['vol_oi_ratio']),
                    'lastPrice': float(row['lastPrice']),
                    'impliedVolatility': float(row.get('impliedVolatility', 0)),
                    'dollar_volume': float(row['dollar_volume']),
                    'moneyness': float(row['moneyness']),
                    'distance_from_price': float(row['distance_from_price']),
                    'whale_score': float(row.get('whale_score', 0)),
                    'strategy_type': row.get('strategy_type', 'UNKNOWN')
                })
        
        # Always return data, even if no unusual activity
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
            'total_calls_value': float(calls['dollar_volume'].sum() if not calls.empty and 'dollar_volume' in calls.columns else 0),
            'total_puts_value': float(puts['dollar_volume'].sum() if not puts.empty and 'dollar_volume' in puts.columns else 0),
            'whale_score': float(total_whale_score),
            'volatility_plays': volatility_plays,
            'directional_bets': directional_bets,
            'unusual_activity': unusual_list,
            'expected_move': expected_move_data,
            'recommended_strategies': recommended_strategies,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_weekly_options(self, calls: pd.DataFrame, puts: pd.DataFrame, 
                           current_price: float) -> Optional[Dict[str, pd.DataFrame]]:
        """Get options expiring within 7-10 days for expected move calculation"""
        if calls.empty or puts.empty:
            return None
        
        weekly_calls = calls[(calls['days_to_expiry'] >= 5) & (calls['days_to_expiry'] <= 10)]
        weekly_puts = puts[(puts['days_to_expiry'] >= 5) & (puts['days_to_expiry'] <= 10)]
        
        if weekly_calls.empty or weekly_puts.empty:
            return None
        
        return {'calls': weekly_calls, 'puts': weekly_puts}
    
    def _calculate_expected_move(self, calls: pd.DataFrame, puts: pd.DataFrame, 
                                current_price: float) -> Optional[Dict[str, Any]]:
        """
        Calculate expected move using ATM straddle and OTM strangles
        Returns 68% probability range (1 standard deviation)
        """
        if calls.empty or puts.empty:
            return None
        
        try:
            # Find ATM strike
            all_strikes = pd.concat([calls['strike'], puts['strike']]).unique()
            atm_strike = min(all_strikes, key=lambda x: abs(x - current_price))
            
            # ATM straddle
            atm_call = calls[calls["strike"] == atm_strike]
            atm_put = puts[puts["strike"] == atm_strike]
            
            if atm_call.empty or atm_put.empty:
                return None
            
            atm_straddle_price = (
                atm_call["lastPrice"].iloc[0] + atm_put["lastPrice"].iloc[0]
            )
            
            # OTM1 strangle
            call_strikes = calls["strike"][calls["strike"] > atm_strike].sort_values()
            put_strikes = puts["strike"][puts["strike"] < atm_strike].sort_values(ascending=False)
            
            if call_strikes.empty or put_strikes.empty:
                otm1_strangle_price = 0
            else:
                otm1_call_strike = call_strikes.iloc[0]
                otm1_put_strike = put_strikes.iloc[0]
                otm1_call = calls[calls["strike"] == otm1_call_strike]
                otm1_put = puts[puts["strike"] == otm1_put_strike]
                otm1_strangle_price = (
                    otm1_call["lastPrice"].iloc[0] + otm1_put["lastPrice"].iloc[0]
                ) if not (otm1_call.empty or otm1_put.empty) else 0
            
            # OTM2 strangle
            if len(call_strikes) > 1 and len(put_strikes) > 1:
                otm2_call_strike = call_strikes.iloc[1]
                otm2_put_strike = put_strikes.iloc[1]
                otm2_call = calls[calls["strike"] == otm2_call_strike]
                otm2_put = puts[puts["strike"] == otm2_put_strike]
                otm2_strangle_price = (
                    otm2_call["lastPrice"].iloc[0] + otm2_put["lastPrice"].iloc[0]
                ) if not (otm2_call.empty or otm2_put.empty) else 0
            else:
                otm2_strangle_price = 0
            
            # Weighted expected move
            expected_move = (
                atm_straddle_price * 0.6 + 
                otm1_strangle_price * 0.3 + 
                otm2_strangle_price * 0.1
            )
            
            upper_bound = current_price + expected_move
            lower_bound = current_price - expected_move
            expected_move_pct = expected_move / current_price * 100
            
            return {
                'atm_strike': float(atm_strike),
                'atm_straddle_price': float(atm_straddle_price),
                'expected_move_dollar': float(expected_move),
                'expected_move_pct': float(expected_move_pct),
                'upper_target': float(upper_bound),
                'lower_target': float(lower_bound),
                'confidence': '68% (1σ)'
            }
        except Exception as e:
            print(f"Error calculating expected move: {e}")
            return None
    
    def _generate_strategy_recommendations(self, unusual_activity: pd.DataFrame, 
                                          sentiment_score: float, bullish_pct: float, 
                                          bearish_pct: float, volatility_plays: int,
                                          directional_bets: int, expected_move: Optional[Dict],
                                          current_price: float) -> List[Dict[str, Any]]:
        """Generate multiple trading strategy recommendations"""
        recommendations = []
        
        # Always provide at least one basic recommendation based on sentiment
        if sentiment_score > 20:
            recommendations.append({
                'strategy': 'BULLISH OUTLOOK - CONSIDER CALLS',
                'confidence': 'MEDIUM' if sentiment_score > 50 else 'LOW',
                'rationale': f'Market showing bullish sentiment ({bullish_pct:.0f}%)',
                'suggested_strikes': 'Consider ATM or slightly OTM calls',
                'risk_level': 'MEDIUM',
                'profit_target': '+5-10%',
                'stop_loss': '-30% of premium',
                'time_horizon': '2-4 weeks'
            })
        elif sentiment_score < -20:
            recommendations.append({
                'strategy': 'BEARISH OUTLOOK - CONSIDER PUTS',
                'confidence': 'MEDIUM' if sentiment_score < -50 else 'LOW',
                'rationale': f'Market showing bearish sentiment ({bearish_pct:.0f}%)',
                'suggested_strikes': 'Consider ATM or slightly OTM puts',
                'risk_level': 'MEDIUM',
                'profit_target': '-5-10%',
                'stop_loss': '-30% of premium',
                'time_horizon': '2-4 weeks'
            })
        else:
            recommendations.append({
                'strategy': 'NEUTRAL MARKET - CONSIDER RANGE STRATEGIES',
                'confidence': 'LOW',
                'rationale': 'Market showing neutral sentiment',
                'suggested_strikes': 'Consider iron condors or calendar spreads',
                'risk_level': 'LOW-MEDIUM',
                'profit_target': 'Collect premium',
                'stop_loss': 'Exit if price breaks range',
                'time_horizon': '1-2 weeks'
            })
        
        # Add more sophisticated strategies if unusual activity exists
        if not unusual_activity.empty:
            calls = unusual_activity[unusual_activity['option_type'] == 'CALL']
            puts = unusual_activity[unusual_activity['option_type'] == 'PUT']
            
            top_call_strikes = calls.nlargest(3, 'whale_score')['strike'].tolist() if not calls.empty else []
            top_put_strikes = puts.nlargest(3, 'whale_score')['strike'].tolist() if not puts.empty else []
            
            avg_iv = unusual_activity['impliedVolatility'].mean() if not unusual_activity.empty else 0
            
            # Strategy 1: Strong Directional
            if sentiment_score > 65 and directional_bets > volatility_plays * 2:
                recommendations.append({
                    'strategy': 'BUY CALL DEBIT SPREAD',
                    'confidence': 'HIGH',
                    'rationale': f'Strong bullish sentiment ({bullish_pct:.0f}%) with {directional_bets} directional bets',
                    'suggested_strikes': f'Buy ${top_call_strikes[0]:.0f} / Sell ${top_call_strikes[1]:.0f}' if len(top_call_strikes) >= 2 else 'Buy ATM Call',
                    'risk_level': 'MEDIUM',
                    'profit_target': f'+{expected_move["expected_move_pct"]:.1f}%' if expected_move else '+5-10%',
                    'stop_loss': '-50% of premium',
                    'time_horizon': '1-4 weeks'
                })
            
            elif sentiment_score < -65 and directional_bets > volatility_plays * 2:
                recommendations.append({
                    'strategy': 'BUY PUT DEBIT SPREAD',
                    'confidence': 'HIGH',
                    'rationale': f'Strong bearish sentiment ({bearish_pct:.0f}%) with {directional_bets} directional bets',
                    'suggested_strikes': f'Buy ${top_put_strikes[0]:.0f} / Sell ${top_put_strikes[1]:.0f}' if len(top_put_strikes) >= 2 else 'Buy ATM Put',
                    'risk_level': 'MEDIUM',
                    'profit_target': f'-{expected_move["expected_move_pct"]:.1f}%' if expected_move else '-5-10%',
                    'stop_loss': '-50% of premium',
                    'time_horizon': '1-4 weeks'
                })
            
            # Strategy 2: High Volatility
            if volatility_plays >= directional_bets and volatility_plays > 3:  # Reduced from 5
                if -20 < sentiment_score < 20:
                    recommendations.append({
                        'strategy': 'IRON CONDOR',
                        'confidence': 'MEDIUM-HIGH',
                        'rationale': f'{volatility_plays} volatility plays with neutral sentiment',
                        'suggested_strikes': f'Sell ${expected_move["lower_target"]:.0f} Put / ${expected_move["upper_target"]:.0f} Call' if expected_move else 'Sell ±5% wings',
                        'risk_level': 'MEDIUM',
                        'profit_target': '40-60% of max profit',
                        'stop_loss': 'Exit if price breaks range',
                        'time_horizon': '1-2 weeks'
                    })
            
            # Strategy 3: Follow the Whale
            if len(unusual_activity) > 0:
                top_whale_trade = unusual_activity.nlargest(1, 'whale_score').iloc[0]
                if top_whale_trade['whale_score'] > 100:  # Reduced from 200
                    direction = "BULLISH" if top_whale_trade['option_type'] == 'CALL' else "BEARISH"
                    recommendations.append({
                        'strategy': f'FOLLOW THE WHALE - {"BUY CALL" if direction == "BULLISH" else "BUY PUT"}',
                        'confidence': 'MEDIUM',
                        'rationale': f'Notable whale trade (Score: {top_whale_trade["whale_score"]:.0f}) at ${top_whale_trade["strike"]:.0f}',
                        'suggested_strikes': f'${top_whale_trade["strike"]:.0f} {top_whale_trade["option_type"]} exp {top_whale_trade["expiration"]}',
                        'risk_level': 'MEDIUM-HIGH',
                        'profit_target': f'{top_whale_trade["distance_from_price"]:+.1f}% to strike',
                        'stop_loss': '-30-50% of premium',
                        'time_horizon': f'{top_whale_trade["days_to_expiry"]} days'
                    })
        
        # Sort by confidence
        confidence_order = {'HIGH': 3, 'MEDIUM-HIGH': 2.5, 'MEDIUM': 2, 'LOW-MEDIUM': 1.5, 'LOW': 1}
        recommendations.sort(key=lambda x: confidence_order.get(x['confidence'], 0), reverse=True)
        
        return recommendations[:3]  # Top 3 strategies
