"""
Cache Manager
Simple in-memory cache with TTL support
For production: Replace with Redis
"""
from datetime import datetime, timedelta
from typing import Any, Optional, Dict

class CacheManager:
    """Simple in-memory cache with TTL"""
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def set(self, key: str, value: Any, ttl: int = 900):
        """
        Set cache entry with TTL
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: 15 min)
        """
        self.cache[key] = {
            'value': value,
            'expires_at': datetime.now() + timedelta(seconds=ttl),
            'created_at': datetime.now()
        }
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value if not expired
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if expired/not found
        """
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if datetime.now() > entry['expires_at']:
            del self.cache[key]
            return None
        
        return entry['value']
    
    def delete(self, key: str) -> bool:
        """
        Delete cache entry
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear_all(self):
        """Clear entire cache"""
        self.cache.clear()
    
    def clear_expired(self):
        """Remove all expired entries"""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now > entry['expires_at']
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)
    
    def get_cache_size(self) -> int:
        """Get number of cached entries"""
        self.clear_expired()  # Clean up first
        return len(self.cache)
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        self.clear_expired()
        
        total_size = len(self.cache)
        oldest = None
        newest = None
        
        if self.cache:
            created_times = [entry['created_at'] for entry in self.cache.values()]
            oldest = min(created_times)
            newest = max(created_times)
        
        return {
            'total_entries': total_size,
            'oldest_entry': oldest.isoformat() if oldest else None,
            'newest_entry': newest.isoformat() if newest else None
        }
