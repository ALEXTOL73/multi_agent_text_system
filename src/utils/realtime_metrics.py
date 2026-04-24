"""
Real-time metrics storage for web monitoring
Stores metrics in memory for immediate access
"""

import threading
from datetime import datetime
from typing import Dict, List, Any, Optional

class RealtimeMetricsStore:
    """
    Thread-safe storage for metrics in real-time
    """
    
    def __init__(self):
        self._metrics: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._last_update = None
    
    def add_metrics(self, filename: str, metrics: Dict[str, Any]) -> None:
        """
        Add or update metrics for a file
        
        Args:
            filename: Name of the processed file
            metrics: Dictionary with all metrics
        """
        with self._lock:
            self._metrics[filename] = {
                **metrics,
                'timestamp': datetime.now().isoformat(),
                'filename': filename
            }
            self._last_update = datetime.now().isoformat()
    
    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """
        Get all stored metrics
        
        Returns:
            List of all metrics dictionaries
        """
        with self._lock:
            return list(self._metrics.values())
    
    def get_metrics(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get metrics for specific file
        
        Args:
            filename: Name of the file
            
        Returns:
            Metrics dictionary or None if not found
        """
        with self._lock:
            return self._metrics.get(filename)
    
    def get_last_update(self) -> Optional[str]:
        """
        Get last update timestamp
        
        Returns:
            ISO timestamp string or None
        """
        with self._lock:
            return self._last_update
    
    def clear_metrics(self) -> None:
        """Clear all stored metrics"""
        with self._lock:
            self._metrics.clear()
            self._last_update = None
    
    def get_metrics_count(self) -> int:
        """
        Get count of stored metrics
        
        Returns:
            Number of stored metric records
        """
        with self._lock:
            return len(self._metrics)

# Global instance for real-time metrics
realtime_store = RealtimeMetricsStore()
