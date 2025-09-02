#!/usr/bin/env python3
"""
Memory monitoring utility for SLAM processes
"""

import psutil
import threading
import time
import os

class MemoryMonitor:
    def __init__(self, process_id=None):
        self.process_id = process_id or os.getpid()
        self.peak_memory = 0
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start monitoring memory usage in a separate thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return peak memory usage"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        return self.peak_memory
        
    def _monitor_loop(self):
        """Monitor memory usage in a loop"""
        try:
            process = psutil.Process(self.process_id)
            while self.monitoring:
                try:
                    # Get memory info in GB
                    memory_info = process.memory_info()
                    memory_gb = memory_info.rss / (1024**3)  # Convert bytes to GB
                    
                    if memory_gb > self.peak_memory:
                        self.peak_memory = memory_gb
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                    
                time.sleep(0.1)  # Check every 100ms
                
        except Exception as e:
            print(f"Memory monitoring error: {e}")
            
    def get_current_memory(self):
        """Get current memory usage in GB"""
        try:
            process = psutil.Process(self.process_id)
            memory_info = process.memory_info()
            return memory_info.rss / (1024**3)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

# Global memory monitor instance
memory_monitor = None

def start_memory_monitoring():
    """Start global memory monitoring"""
    global memory_monitor
    memory_monitor = MemoryMonitor()
    memory_monitor.start_monitoring()
    
def stop_memory_monitoring():
    """Stop global memory monitoring and return peak usage"""
    global memory_monitor
    if memory_monitor:
        return memory_monitor.stop_monitoring()
    return 0

def get_peak_memory():
    """Get current peak memory usage"""
    global memory_monitor
    if memory_monitor:
        return memory_monitor.peak_memory
    return 0
