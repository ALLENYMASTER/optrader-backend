"""
Market Analyzer
Analyzes overall market sentiment and generates top opportunities
"""

class MarketAnalyzer:
    """Analyze market-wide sentiment and opportunities"""
    
    def analyze(self, scan_results: list) -> dict:
        """
        Analyze overall market from scan results
        
        Args:
            scan_results: List of ticker analysis results
            
        Returns:
            Market summary dictionary
        """
        if not scan_results:
            return self._empty_summary()
        
        # Calculate overall sentiment
        total_bullish = sum(r['bullish_percent'] for r in scan_results)
        total_bearish = sum(r['bearish_percent'] for r in scan_results)
        total = total_bullish + total_bearish
        
        overall_bullish = (total_bullish / total * 100) if total > 0 else 0
        overall_bearish = (total_bearish / total * 100) if total > 0 else 0
        
        # Determine market regime
        regime_data = self._determine_regime(overall_bullish, overall_bearish)
        
        # Find top opportunities
        top_opps = self._find_top_opportunities(scan_results)
        
        return {
            'regime': regime_data['regime'],
            'regime_icon': regime_data['icon'],
            'bullish_percent': float(overall_bullish),
            'bearish_percent': float(overall_bearish),
            'total_tickers': len(scan_results),
            'active_tickers': len(scan_results),
            'top_opportunities': top_opps,
            'risk_level': regime_data['risk_level']
        }
    
    def _determine_regime(self, bullish_pct: float, bearish_pct: float) -> dict:
        """Determine market regime from sentiment percentages"""
        if bullish_pct > 65:
            return {
                'regime': 'STRONG BULL MARKET',
                'icon': '🚀📈',
                'risk_level': 'MEDIUM'
            }
        elif bullish_pct > 55:
            return {
                'regime': 'BULL MARKET',
                'icon': '📈',
                'risk_level': 'MEDIUM'
            }
        elif bearish_pct > 65:
            return {
                'regime': 'STRONG BEAR MARKET',
                'icon': '🐻📉',
                'risk_level': 'HIGH'
            }
        elif bearish_pct > 55:
            return {
                'regime': 'BEAR MARKET',
                'icon': '📉',
                'risk_level': 'HIGH'
            }
        else:
            return {
                'regime': 'NEUTRAL/CHOPPY MARKET',
                'icon': '↔️',
                'risk_level': 'HIGH'
            }
    
    def _find_top_opportunities(self, results: list, top_n: int = 5) -> list:
        """Find top N trading opportunities"""
        # Sort by absolute sentiment score and whale score
        scored_results = []
        for result in results:
            score = abs(result['sentiment_score']) * (result['whale_score'] / 100)
            scored_results.append((score, result))
        
        scored_results.sort(reverse=True, key=lambda x: x[0])
        
        opportunities = []
        for _, result in scored_results[:top_n]:
            strategy = result.get('top_strategy', {})
            
            opportunities.append({
                'ticker': result['ticker'],
                'sentiment': result['sentiment'],
                'sentiment_score': result['sentiment_score'],
                'strategy': strategy.get('name', 'N/A') if strategy else 'N/A',
                'confidence': strategy.get('confidence', 'UNKNOWN') if strategy else 'UNKNOWN',
                'current_price': result['current_price']
            })
        
        return opportunities
    
    def _empty_summary(self) -> dict:
        """Return empty summary when no data"""
        return {
            'regime': 'UNKNOWN',
            'regime_icon': '❓',
            'bullish_percent': 0.0,
            'bearish_percent': 0.0,
            'total_tickers': 0,
            'active_tickers': 0,
            'top_opportunities': [],
            'risk_level': 'UNKNOWN'
        }
