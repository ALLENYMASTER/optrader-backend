"""
Ticker Manager
Manages user-specific ticker lists
"""
from typing import List, Dict

class TickerManager:
    """Manage ticker lists per user"""
    
    def __init__(self):
        self.user_tickers: Dict[str, List[str]] = {}
    
    def get_tickers(self, user_id: str) -> List[str]:
        """
        Get ticker list for user
        
        Args:
            user_id: User identifier
            
        Returns:
            List of ticker symbols
        """
        return self.user_tickers.get(user_id, [])
    
    def set_tickers(self, user_id: str, tickers: List[str]):
        """
        Set entire ticker list for user
        
        Args:
            user_id: User identifier
            tickers: List of ticker symbols
        """
        # Clean and validate tickers
        clean_tickers = [t.upper().strip() for t in tickers if t.strip()]
        self.user_tickers[user_id] = clean_tickers
    
    def add_ticker(self, user_id: str, ticker: str) -> bool:
        """
        Add single ticker to user's list
        
        Args:
            user_id: User identifier
            ticker: Ticker symbol to add
            
        Returns:
            True if added, False if already exists
        """
        ticker = ticker.upper().strip()
        
        if user_id not in self.user_tickers:
            self.user_tickers[user_id] = []
        
        if ticker in self.user_tickers[user_id]:
            return False  # Already exists
        
        self.user_tickers[user_id].append(ticker)
        return True
    
    def remove_ticker(self, user_id: str, ticker: str) -> bool:
        """
        Remove ticker from user's list
        
        Args:
            user_id: User identifier
            ticker: Ticker symbol to remove
            
        Returns:
            True if removed, False if not found
        """
        ticker = ticker.upper().strip()
        
        if user_id not in self.user_tickers:
            return False
        
        if ticker not in self.user_tickers[user_id]:
            return False
        
        self.user_tickers[user_id].remove(ticker)
        return True
    
    def has_ticker(self, user_id: str, ticker: str) -> bool:
        """Check if user has ticker in list"""
        ticker = ticker.upper().strip()
        return ticker in self.user_tickers.get(user_id, [])
    
    def ticker_count(self, user_id: str) -> int:
        """Get ticker count for user"""
        return len(self.user_tickers.get(user_id, []))
    
    def all_users(self) -> List[str]:
        """Get list of all user IDs"""
        return list(self.user_tickers.keys())
