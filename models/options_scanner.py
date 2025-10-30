"""
Enhanced Options Whale Scanner
Based on us_whale_web.py with full feature set:
- Expected move calculation (weekly options)
- Strategy type detection (DIRECTIONAL vs VOLATILITY vs STRADDLE)
- Enhanced sentiment analysis
- Multiple strategy recommendations
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Any
import time
from tenacity import retry, stop_after_attempt, wait_exponential
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
    
    def scan_all_parallel(self, tickers: list, max_workers: int = 5) -> list:
        """Scan tickers in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.scan_single_ticker, ticker): ticker 
                for ticker in tickers
            }
            
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
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _get_options_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive options data from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            
            if current_price == 0:
                return None
            
            # Get additional market metrics
            volume = info.get('volume', 0)
            avg_volume = info.get('averageVolume', 1)
            market_cap = info.get('marketCap', 0)
            
            expirations = stock.options
            if not expirations:
                return None
            
            all_calls = []
            all_puts = []
            cutoff_date = datetime.now() + timedelta(days=90)
            
            # Fetch first 6 expirations (up to 90 days)
            for exp_date in expirations[:6]:
                exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                if exp_datetime > cutoff_date:
                    continue
                
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
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def _identify_whale_activity(self, options_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Identify unusual options activity with enhanced analysis"""
        if not options_data:
            return None
        
        ticker = options_data['ticker']
        current_price = options_data['current_price']
        calls = options_data['calls']
        puts = options_data['puts']
        
        # Calculate expected move for weekly options
        weekly_options = self._get_weekly_options(calls, puts, current_price)
        expected_move_data = self._calculate_expected_move(
            weekly_options['calls'], 
            weekly_options['puts'], 
            current_price
        ) if weekly_options else None
        
        def find_unusual_activity(df, option_type):
            """Find unusual trades with enhanced metrics"""
            if df.empty:
                return pd.DataFrame()
            
            # Enhanced calculations (from us_whale_web.py)
            df['total_volume_value'] = df['volume'] * df['lastPrice'] * 100
            df['oi_value'] = df['openInterest'] * df['lastPrice'] * 100
            df['volume_oi_ratio'] = df['volume'] / (df['openInterest'] + 1)
            
            # IV percentile
            df['iv_percentile'] = df['impliedVolatility'].rank(pct=True)
            
            # Moneyness
            if option_type == 'CALL':
                df['moneyness'] = df['strike'] / current_price
            else:
                df['moneyness'] = current_price / df['strike']
            
            # ATM detection
            df['is_atm'] = (df['strike'] >= current_price * 0.95) & (df['strike'] <= current_price * 1.05)
            
            # Strategy type detection
            df['strategy_type'] = 'DIRECTIONAL'
            
            # High IV + ATM + High volume = Volatility play
            df.loc[
                (df['iv_percentile'] > 0.7) & 
                (df['is_atm']) & 
                (df['volume_oi_ratio'] > 1.0),
                'strategy_type'
            ] = 'VOLATILITY'
            
            # Straddle/Strangle detection
            df.loc[
                (df['is_atm']) & 
                (df['volume_oi_ratio'] > 2.0),
                'strategy_type'
            ] = 'STRADDLE/STRANGLE'
            
            # Filter unusual activity (#: If market is quiet)
            unusual = df[
                (df['volume'] > 100) &  # 50
                (df['total_volume_value'] > 50000) &  # 25000
                (df['volume_oi_ratio'] > 0.3) &  # 0.2
                (df['lastPrice'] > 0.05)
            ].copy()
            
            if not unusual.empty:
                unusual['option_type'] = option_type
                unusual['distance_from_price'] = ((unusual['strike'] - current_price) / current_price * 100).round(2)
                
                # Determine likely direction
                unusual['likely_direction'] = 'UNKNOWN'
                
                if option_type == 'CALL':
                    unusual.loc[(unusual['strike'] < current_price) & (unusual['volume_oi_ratio'] > 1.5), 'likely_direction'] = 'LIKELY SELL'
                    unusual.loc[(unusual['strike'] >= current_price) & (unusual['volume_oi_ratio'] > 1.5), 'likely_direction'] = 'LIKELY BUY'
                else:
                    unusual.loc[(unusual['strike'] > current_price) & (unusual['volume_oi_ratio'] > 1.5), 'likely_direction'] = 'LIKELY SELL'
                    unusual.loc[(unusual['strike'] <= current_price) & (unusual['volume_oi_ratio'] > 1.5), 'likely_direction'] = 'LIKELY BUY'
                
                # Sentiment adjustment
                if option_type == 'CALL':
                    unusual['sentiment_adjustment'] = np.where(
                        unusual['strike'] >= current_price * 0.95,
                        1.0,
                        0.3
                    )
                else:
                    unusual['sentiment_adjustment'] = np.where(
                        unusual['strike'] <= current_price * 1.05,
                        1.0,
                        0.3
                    )
                
                # Enhanced whale score
                unusual['whale_score'] = (
                    np.log1p(unusual['total_volume_value']) * 0.4 +
                    unusual['volume_oi_ratio'] * 100 * 0.3 +
                    unusual['iv_percentile'] * 100 * 0.3
                ) * unusual['sentiment_adjustment']
            
            return unusual
        
        unusual_calls = find_unusual_activity(calls, 'CALL')
        unusual_puts = find_unusual_activity(puts, 'PUT')
        unusual_activity = pd.concat([unusual_calls, unusual_puts], ignore_index=True)
        
        if unusual_activity.empty:
            return None
        
        unusual_activity = unusual_activity.sort_values('whale_score', ascending=False)
        
        # Count strategy types
        volatility_plays = len(unusual_activity[unusual_activity['strategy_type'].str.contains('VOLATILITY|STRADDLE', na=False)])
        directional_bets = len(unusual_activity[unusual_activity['strategy_type'] == 'DIRECTIONAL'])
        
        # Enhanced sentiment calculation
        likely_buy_calls = unusual_calls[
            (unusual_calls['likely_direction'] == 'LIKELY BUY') | 
            (unusual_calls['strike'] >= current_price * 0.95)
        ] if not unusual_calls.empty else pd.DataFrame()
        
        likely_buy_puts = unusual_puts[
            (unusual_puts['likely_direction'] == 'LIKELY BUY') |
            (unusual_puts['strike'] <= current_price * 1.05)
        ] if not unusual_puts.empty else pd.DataFrame()
        
        total_calls_value = unusual_calls['total_volume_value'].sum() if not unusual_calls.empty else 0
        total_puts_value = unusual_puts['total_volume_value'].sum() if not unusual_puts.empty else 0
        
        # Weight by whale score and sentiment adjustment
        weighted_calls = (unusual_calls['whale_score'] * unusual_calls['sentiment_adjustment']).sum() if not unusual_calls.empty else 0
        weighted_puts = (unusual_puts['whale_score'] * unusual_puts['sentiment_adjustment']).sum() if not unusual_puts.empty else 0
        total_weighted = weighted_calls + weighted_puts
        
        if total_weighted > 0:
            bullish_pct = (weighted_calls / total_weighted * 100)
            bearish_pct = (weighted_puts / total_weighted * 100)
        else:
            bullish_pct = bearish_pct = 0
        
        # Determine sentiment
        if bullish_pct > 65:
            sentiment = 'STRONG BULLISH'
            sentiment_score = bullish_pct
        elif bullish_pct > 55:
            sentiment = 'BULLISH'
            sentiment_score = bullish_pct
        elif bearish_pct > 65:
            sentiment = 'STRONG BEARISH'
            sentiment_score = -bearish_pct
        elif bearish_pct > 55:
            sentiment = 'BEARISH'
            sentiment_score = -bearish_pct
        else:
            sentiment = 'NEUTRAL'
            sentiment_score = bullish_pct - bearish_pct
        
        # Generate multiple strategy recommendations
        recommended_strategies = self._generate_strategy_recommendations(
            unusual_activity, sentiment_score, bullish_pct, bearish_pct,
            volatility_plays, directional_bets, expected_move_data, current_price
        )
        
        # Convert unusual_activity to list of dicts
        unusual_list = unusual_activity.head(20).to_dict('records')
        
        # Clean up numpy types
        for item in unusual_list:
            for key, value in item.items():
                if isinstance(value, (np.integer, np.floating)):
                    item[key] = float(value) if isinstance(value, np.floating) else int(value)
                elif pd.isna(value):
                    item[key] = None
        
        return {
            'ticker': ticker,
            'current_price': float(current_price),
            'volume_ratio': float(options_data['volume_ratio']),
            'market_cap': float(options_data['market_cap']),
            'sentiment': sentiment,
            'sentiment_score': float(sentiment_score),
            'bullish_percent': float(bullish_pct),
            'bearish_percent': float(bearish_pct),
            'total_calls_count': len(unusual_calls),
            'total_puts_count': len(unusual_puts),
            'total_calls_value': float(total_calls_value),
            'total_puts_value': float(total_puts_value),
            'whale_score': float(unusual_activity['whale_score'].max()),
            'volatility_plays': int(volatility_plays),
            'directional_bets': int(directional_bets),
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
        
        if unusual_activity.empty:
            return recommendations
        
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
        if volatility_plays >= directional_bets and volatility_plays > 5:
            if -20 < sentiment_score < 20:
                recommendations.append({
                    'strategy': 'IRON CONDOR',
                    'confidence': 'HIGH',
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
            if top_whale_trade['whale_score'] > 200:
                direction = "BULLISH" if top_whale_trade['option_type'] == 'CALL' else "BEARISH"
                recommendations.append({
                    'strategy': f'FOLLOW THE WHALE - {"BUY CALL" if direction == "BULLISH" else "BUY PUT"}',
                    'confidence': 'MEDIUM-HIGH',
                    'rationale': f'Massive whale trade (Score: {top_whale_trade["whale_score"]:.0f}) at ${top_whale_trade["strike"]:.0f}',
                    'suggested_strikes': f'${top_whale_trade["strike"]:.0f} {top_whale_trade["option_type"]} exp {top_whale_trade["expiration"]}',
                    'risk_level': 'HIGH',
                    'profit_target': f'{top_whale_trade["distance_from_price"]:+.1f}% to strike',
                    'stop_loss': '-30-50% of premium',
                    'time_horizon': f'{top_whale_trade["days_to_expiry"]} days'
                })
        
        # Sort by confidence
        confidence_order = {'HIGH': 3, 'MEDIUM-HIGH': 2.5, 'MEDIUM': 2, 'LOW-MEDIUM': 1.5, 'LOW': 1}
        recommendations.sort(key=lambda x: confidence_order.get(x['confidence'], 0), reverse=True)
        
        return recommendations[:5]  # Top 5 strategies
