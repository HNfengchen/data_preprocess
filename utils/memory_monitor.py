import gc
import psutil


class MemoryMonitor:
    """Monitor current process memory, trigger GC when over limit"""

    def __init__(self, limit_mb: int = 2048):
        self.limit_mb = limit_mb

    def get_usage_mb(self) -> float:
        """Return current process RSS in MB"""
        return psutil.Process().memory_info().rss / 1024 / 1024

    def check_and_collect(self) -> bool:
        """
        Trigger GC if memory exceeds limit.
        Returns True if GC was triggered, False otherwise.
        """
        if self.get_usage_mb() > self.limit_mb:
            gc.collect()
            return True
        return False
