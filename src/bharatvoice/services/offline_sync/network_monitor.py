<<<<<<< HEAD
"""
Network monitoring system for BharatVoice Assistant.

This module provides intelligent network connectivity monitoring,
intermittent connectivity handling, and network quality assessment.
"""

import asyncio
import logging
import socket
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

from pydantic import BaseModel


logger = logging.getLogger(__name__)


class NetworkStatus(str, Enum):
    """Network connectivity status."""
    ONLINE = "online"
    OFFLINE = "offline"
    LIMITED = "limited"
    UNSTABLE = "unstable"


class ConnectionQuality(str, Enum):
    """Network connection quality."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNAVAILABLE = "unavailable"


@dataclass
class NetworkMetrics:
    """Network performance metrics."""
    latency_ms: float
    packet_loss_percent: float
    bandwidth_mbps: float
    jitter_ms: float
    timestamp: datetime


class NetworkEvent(BaseModel):
    """Network connectivity event."""
    event_type: str  # "connected", "disconnected", "quality_changed"
    timestamp: datetime
    previous_status: Optional[NetworkStatus] = None
    current_status: NetworkStatus
    metrics: Optional[NetworkMetrics] = None
    duration_seconds: Optional[float] = None


class NetworkMonitor:
    """
    Intelligent network monitoring system that tracks connectivity,
    quality, and provides graceful handling of intermittent connections.
    """
    
    def __init__(
        self,
        check_interval_seconds: int = 30,
        quality_check_interval_seconds: int = 300,
        connectivity_timeout_seconds: int = 5,
        enable_quality_monitoring: bool = True,
        test_hosts: List[str] = None
    ):
        """
        Initialize network monitor.
        
        Args:
            check_interval_seconds: Interval between connectivity checks
            quality_check_interval_seconds: Interval between quality assessments
            connectivity_timeout_seconds: Timeout for connectivity tests
            enable_quality_monitoring: Whether to monitor connection quality
            test_hosts: List of hosts to test connectivity against
        """
        self.check_interval_seconds = check_interval_seconds
        self.quality_check_interval_seconds = quality_check_interval_seconds
        self.connectivity_timeout_seconds = connectivity_timeout_seconds
        self.enable_quality_monitoring = enable_quality_monitoring
        
        # Default test hosts (reliable public DNS servers)
        self.test_hosts = test_hosts or [
            "8.8.8.8",      # Google DNS
            "1.1.1.1",      # Cloudflare DNS
            "208.67.222.222" # OpenDNS
        ]
        
        # Current state
        self.current_status = NetworkStatus.OFFLINE
        self.current_quality = ConnectionQuality.UNAVAILABLE
        self.last_check_time = datetime.now()
        self.last_quality_check_time = datetime.now()
        
        # Event tracking
        self.network_events: List[NetworkEvent] = []
        self.event_callbacks: List[Callable[[NetworkEvent], None]] = []
        
        # Statistics
        self.connection_stats = {
            'total_checks': 0,
            'successful_checks': 0,
            'failed_checks': 0,
            'status_changes': 0,
            'total_downtime_seconds': 0.0,
            'total_uptime_seconds': 0.0,
            'average_latency_ms': 0.0,
            'quality_degradations': 0
        }
        
        # Monitoring tasks
        self._monitoring_task = None
        self._quality_task = None
        self._is_monitoring = False
        
        # Recent metrics for trend analysis
        self.recent_metrics: List[NetworkMetrics] = []
        self.max_recent_metrics = 100
        
        logger.info("NetworkMonitor initialized successfully")
    
    async def start_monitoring(self):
        """Start network monitoring tasks."""
        if self._is_monitoring:
            logger.warning("Network monitoring already started")
            return
        
        self._is_monitoring = True
        
        # Start connectivity monitoring
        self._monitoring_task = asyncio.create_task(self._connectivity_monitoring_loop())
        
        # Start quality monitoring if enabled
        if self.enable_quality_monitoring:
            self._quality_task = asyncio.create_task(self._quality_monitoring_loop())
        
        # Perform initial check
        await self.check_connectivity()
        
        logger.info("Network monitoring started")
    
    async def stop_monitoring(self):
        """Stop network monitoring tasks."""
        if not self._is_monitoring:
            return
        
        self._is_monitoring = False
        
        # Cancel monitoring tasks
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self._quality_task and not self._quality_task.done():
            self._quality_task.cancel()
            try:
                await self._quality_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Network monitoring stopped")
    
    async def _connectivity_monitoring_loop(self):
        """Background connectivity monitoring loop."""
        while self._is_monitoring:
            try:
                await self.check_connectivity()
                await asyncio.sleep(self.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connectivity monitoring loop: {e}")
                await asyncio.sleep(self.check_interval_seconds)
    
    async def _quality_monitoring_loop(self):
        """Background quality monitoring loop."""
        while self._is_monitoring:
            try:
                if self.current_status == NetworkStatus.ONLINE:
                    await self.assess_connection_quality()
                await asyncio.sleep(self.quality_check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in quality monitoring loop: {e}")
                await asyncio.sleep(self.quality_check_interval_seconds)
    
    async def check_connectivity(self) -> NetworkStatus:
        """
        Check network connectivity status.
        
        Returns:
            Current network status
        """
        try:
            previous_status = self.current_status
            check_start_time = time.time()
            
            # Test connectivity to multiple hosts
            successful_connections = 0
            total_latency = 0.0
            
            for host in self.test_hosts:
                try:
                    start_time = time.time()
                    
                    # Create socket connection
                    sock = socket.create_connection(
                        (host, 53),  # DNS port
                        timeout=self.connectivity_timeout_seconds
                    )
                    sock.close()
                    
                    end_time = time.time()
                    latency = (end_time - start_time) * 1000  # Convert to ms
                    total_latency += latency
                    successful_connections += 1
                    
                except (socket.error, OSError, socket.timeout):
                    continue
            
            # Determine network status
            if successful_connections == 0:
                self.current_status = NetworkStatus.OFFLINE
            elif successful_connections == len(self.test_hosts):
                self.current_status = NetworkStatus.ONLINE
            elif successful_connections >= len(self.test_hosts) // 2:
                self.current_status = NetworkStatus.LIMITED
            else:
                self.current_status = NetworkStatus.UNSTABLE
            
            # Update statistics
            self.connection_stats['total_checks'] += 1
            if successful_connections > 0:
                self.connection_stats['successful_checks'] += 1
                if successful_connections > 0:
                    avg_latency = total_latency / successful_connections
                    self.connection_stats['average_latency_ms'] = (
                        (self.connection_stats['average_latency_ms'] * 
                         (self.connection_stats['successful_checks'] - 1) + avg_latency) /
                        self.connection_stats['successful_checks']
                    )
            else:
                self.connection_stats['failed_checks'] += 1
            
            # Track status changes
            if previous_status != self.current_status:
                self.connection_stats['status_changes'] += 1
                
                # Calculate uptime/downtime
                check_duration = time.time() - check_start_time
                if previous_status == NetworkStatus.OFFLINE:
                    self.connection_stats['total_downtime_seconds'] += check_duration
                else:
                    self.connection_stats['total_uptime_seconds'] += check_duration
                
                # Create network event
                event = NetworkEvent(
                    event_type="status_changed",
                    timestamp=datetime.now(),
                    previous_status=previous_status,
                    current_status=self.current_status,
                    duration_seconds=check_duration
                )
                
                await self._handle_network_event(event)
                
                logger.info(f"Network status changed: {previous_status} -> {self.current_status}")
            
            self.last_check_time = datetime.now()
            return self.current_status
            
        except Exception as e:
            logger.error(f"Error checking network connectivity: {e}")
            self.current_status = NetworkStatus.OFFLINE
            return self.current_status
    
    async def assess_connection_quality(self) -> ConnectionQuality:
        """
        Assess network connection quality.
        
        Returns:
            Current connection quality
        """
        try:
            if self.current_status == NetworkStatus.OFFLINE:
                self.current_quality = ConnectionQuality.UNAVAILABLE
                return self.current_quality
            
            # Perform quality tests
            latencies = []
            packet_loss_count = 0
            total_tests = 5
            
            for i in range(total_tests):
                try:
                    start_time = time.time()
                    
                    # Test with primary host
                    sock = socket.create_connection(
                        (self.test_hosts[0], 53),
                        timeout=self.connectivity_timeout_seconds
                    )
                    sock.close()
                    
                    end_time = time.time()
                    latency = (end_time - start_time) * 1000
                    latencies.append(latency)
                    
                except (socket.error, OSError, socket.timeout):
                    packet_loss_count += 1
                
                # Small delay between tests
                await asyncio.sleep(0.1)
            
            # Calculate metrics
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                jitter = max(latencies) - min(latencies) if len(latencies) > 1 else 0.0
            else:
                avg_latency = float('inf')
                jitter = 0.0
            
            packet_loss_percent = (packet_loss_count / total_tests) * 100
            
            # Estimate bandwidth (simplified)
            bandwidth_mbps = self._estimate_bandwidth(avg_latency, packet_loss_percent)
            
            # Create metrics
            metrics = NetworkMetrics(
                latency_ms=avg_latency,
                packet_loss_percent=packet_loss_percent,
                bandwidth_mbps=bandwidth_mbps,
                jitter_ms=jitter,
                timestamp=datetime.now()
            )
            
            # Store recent metrics
            self.recent_metrics.append(metrics)
            if len(self.recent_metrics) > self.max_recent_metrics:
                self.recent_metrics.pop(0)
            
            # Determine quality
            previous_quality = self.current_quality
            
            if packet_loss_percent > 10 or avg_latency > 1000:
                self.current_quality = ConnectionQuality.POOR
            elif packet_loss_percent > 5 or avg_latency > 500:
                self.current_quality = ConnectionQuality.FAIR
            elif packet_loss_percent > 1 or avg_latency > 200:
                self.current_quality = ConnectionQuality.GOOD
            else:
                self.current_quality = ConnectionQuality.EXCELLENT
            
            # Track quality changes
            if previous_quality != self.current_quality:
                if (previous_quality in [ConnectionQuality.EXCELLENT, ConnectionQuality.GOOD] and
                    self.current_quality in [ConnectionQuality.FAIR, ConnectionQuality.POOR]):
                    self.connection_stats['quality_degradations'] += 1
                
                # Create quality change event
                event = NetworkEvent(
                    event_type="quality_changed",
                    timestamp=datetime.now(),
                    current_status=self.current_status,
                    metrics=metrics
                )
                
                await self._handle_network_event(event)
                
                logger.info(f"Network quality changed: {previous_quality} -> {self.current_quality}")
            
            self.last_quality_check_time = datetime.now()
            return self.current_quality
            
        except Exception as e:
            logger.error(f"Error assessing connection quality: {e}")
            self.current_quality = ConnectionQuality.UNAVAILABLE
            return self.current_quality
    
    def _estimate_bandwidth(self, latency_ms: float, packet_loss_percent: float) -> float:
        """
        Estimate bandwidth based on latency and packet loss.
        
        Args:
            latency_ms: Average latency in milliseconds
            packet_loss_percent: Packet loss percentage
            
        Returns:
            Estimated bandwidth in Mbps
        """
        try:
            # Simplified bandwidth estimation
            # In production, this could use actual throughput tests
            
            base_bandwidth = 100.0  # Assume 100 Mbps base
            
            # Reduce based on latency
            if latency_ms > 500:
                base_bandwidth *= 0.3
            elif latency_ms > 200:
                base_bandwidth *= 0.6
            elif latency_ms > 100:
                base_bandwidth *= 0.8
            
            # Reduce based on packet loss
            if packet_loss_percent > 5:
                base_bandwidth *= 0.4
            elif packet_loss_percent > 1:
                base_bandwidth *= 0.7
            
            return max(0.1, base_bandwidth)  # Minimum 0.1 Mbps
            
        except Exception as e:
            logger.error(f"Error estimating bandwidth: {e}")
            return 1.0  # Default 1 Mbps
    
    async def _handle_network_event(self, event: NetworkEvent):
        """
        Handle network event by storing it and notifying callbacks.
        
        Args:
            event: Network event to handle
        """
        try:
            # Store event
            self.network_events.append(event)
            
            # Limit event history
            if len(self.network_events) > 1000:
                self.network_events = self.network_events[-500:]  # Keep last 500
            
            # Notify callbacks
            for callback in self.event_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in network event callback: {e}")
            
        except Exception as e:
            logger.error(f"Error handling network event: {e}")
    
    def add_event_callback(self, callback: Callable[[NetworkEvent], None]):
        """
        Add callback for network events.
        
        Args:
            callback: Function to call when network events occur
        """
        self.event_callbacks.append(callback)
    
    def remove_event_callback(self, callback: Callable[[NetworkEvent], None]):
        """
        Remove network event callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)
    
    def get_network_status(self) -> Dict[str, Any]:
        """
        Get current network status and statistics.
        
        Returns:
            Network status information
        """
        try:
            # Calculate uptime percentage
            total_time = (
                self.connection_stats['total_uptime_seconds'] + 
                self.connection_stats['total_downtime_seconds']
            )
            uptime_percentage = (
                (self.connection_stats['total_uptime_seconds'] / total_time * 100)
                if total_time > 0 else 0.0
            )
            
            # Get recent quality trend
            recent_quality_trend = self._analyze_quality_trend()
            
            return {
                'current_status': self.current_status.value,
                'current_quality': self.current_quality.value,
                'last_check_time': self.last_check_time.isoformat(),
                'last_quality_check_time': self.last_quality_check_time.isoformat(),
                'uptime_percentage': uptime_percentage,
                'statistics': self.connection_stats.copy(),
                'quality_trend': recent_quality_trend,
                'recent_events_count': len(self.network_events),
                'monitoring_active': self._is_monitoring
            }
            
        except Exception as e:
            logger.error(f"Error getting network status: {e}")
            return {
                'current_status': self.current_status.value,
                'current_quality': self.current_quality.value,
                'error': str(e)
            }
    
    def _analyze_quality_trend(self) -> str:
        """
        Analyze recent quality trend.
        
        Returns:
            Quality trend description
        """
        try:
            if len(self.recent_metrics) < 3:
                return "insufficient_data"
            
            # Get recent latency values
            recent_latencies = [m.latency_ms for m in self.recent_metrics[-5:]]
            
            if len(recent_latencies) < 3:
                return "stable"
            
            # Calculate trend
            first_half = sum(recent_latencies[:len(recent_latencies)//2]) / (len(recent_latencies)//2)
            second_half = sum(recent_latencies[len(recent_latencies)//2:]) / (len(recent_latencies) - len(recent_latencies)//2)
            
            change_percent = ((second_half - first_half) / first_half) * 100
            
            if change_percent > 20:
                return "degrading"
            elif change_percent < -20:
                return "improving"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error analyzing quality trend: {e}")
            return "unknown"
    
    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent network events.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of recent network events
        """
        try:
            recent_events = self.network_events[-limit:] if self.network_events else []
            
            return [
                {
                    'event_type': event.event_type,
                    'timestamp': event.timestamp.isoformat(),
                    'previous_status': event.previous_status.value if event.previous_status else None,
                    'current_status': event.current_status.value,
                    'duration_seconds': event.duration_seconds,
                    'metrics': {
                        'latency_ms': event.metrics.latency_ms,
                        'packet_loss_percent': event.metrics.packet_loss_percent,
                        'bandwidth_mbps': event.metrics.bandwidth_mbps,
                        'jitter_ms': event.metrics.jitter_ms
                    } if event.metrics else None
                }
                for event in reversed(recent_events)
            ]
            
        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []
    
    async def wait_for_connectivity(self, timeout_seconds: int = 60) -> bool:
        """
        Wait for network connectivity to be restored.
        
        Args:
            timeout_seconds: Maximum time to wait
            
        Returns:
            True if connectivity restored, False if timeout
        """
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout_seconds:
                status = await self.check_connectivity()
                
                if status in [NetworkStatus.ONLINE, NetworkStatus.LIMITED]:
                    return True
                
                await asyncio.sleep(1)  # Check every second
            
            return False
            
        except Exception as e:
            logger.error(f"Error waiting for connectivity: {e}")
            return False
    
    def is_online(self) -> bool:
        """Check if network is currently online."""
        return self.current_status in [NetworkStatus.ONLINE, NetworkStatus.LIMITED]
    
    def is_stable(self) -> bool:
        """Check if network connection is stable."""
        return (
            self.current_status == NetworkStatus.ONLINE and
            self.current_quality in [ConnectionQuality.EXCELLENT, ConnectionQuality.GOOD]
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of network monitor.
        
        Returns:
            Health check result
        """
        try:
            # Check current connectivity
            current_status = await self.check_connectivity()
            
            # Check monitoring tasks
            monitoring_healthy = self._is_monitoring and (
                self._monitoring_task and not self._monitoring_task.done()
            )
            
            quality_monitoring_healthy = (
                not self.enable_quality_monitoring or
                (self._quality_task and not self._quality_task.done())
            )
            
            # Overall health
            overall_health = "healthy" if (
                monitoring_healthy and quality_monitoring_healthy
            ) else "degraded"
            
            return {
                'status': overall_health,
                'network_status': current_status.value,
                'network_quality': self.current_quality.value,
                'monitoring_active': self._is_monitoring,
                'monitoring_task_healthy': monitoring_healthy,
                'quality_monitoring_healthy': quality_monitoring_healthy,
                'last_check_age_seconds': (datetime.now() - self.last_check_time).total_seconds(),
                'statistics': self.connection_stats.copy(),
                'recent_events_count': len(self.network_events)
            }
            
        except Exception as e:
            logger.error(f"Network monitor health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'network_status': self.current_status.value
            }


# Factory function for creating network monitor
def create_network_monitor(
    check_interval_seconds: int = 30,
    quality_check_interval_seconds: int = 300,
    connectivity_timeout_seconds: int = 5,
    enable_quality_monitoring: bool = True,
    test_hosts: List[str] = None
) -> NetworkMonitor:
    """
    Factory function to create a network monitor instance.
    
    Args:
        check_interval_seconds: Interval between connectivity checks
        quality_check_interval_seconds: Interval between quality assessments
        connectivity_timeout_seconds: Timeout for connectivity tests
        enable_quality_monitoring: Whether to monitor connection quality
        test_hosts: List of hosts to test connectivity against
        
    Returns:
        Configured NetworkMonitor instance
    """
    return NetworkMonitor(
        check_interval_seconds=check_interval_seconds,
        quality_check_interval_seconds=quality_check_interval_seconds,
        connectivity_timeout_seconds=connectivity_timeout_seconds,
        enable_quality_monitoring=enable_quality_monitoring,
        test_hosts=test_hosts
=======
"""
Network monitoring system for BharatVoice Assistant.

This module provides intelligent network connectivity monitoring,
intermittent connectivity handling, and network quality assessment.
"""

import asyncio
import logging
import socket
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

from pydantic import BaseModel


logger = logging.getLogger(__name__)


class NetworkStatus(str, Enum):
    """Network connectivity status."""
    ONLINE = "online"
    OFFLINE = "offline"
    LIMITED = "limited"
    UNSTABLE = "unstable"


class ConnectionQuality(str, Enum):
    """Network connection quality."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNAVAILABLE = "unavailable"


@dataclass
class NetworkMetrics:
    """Network performance metrics."""
    latency_ms: float
    packet_loss_percent: float
    bandwidth_mbps: float
    jitter_ms: float
    timestamp: datetime


class NetworkEvent(BaseModel):
    """Network connectivity event."""
    event_type: str  # "connected", "disconnected", "quality_changed"
    timestamp: datetime
    previous_status: Optional[NetworkStatus] = None
    current_status: NetworkStatus
    metrics: Optional[NetworkMetrics] = None
    duration_seconds: Optional[float] = None


class NetworkMonitor:
    """
    Intelligent network monitoring system that tracks connectivity,
    quality, and provides graceful handling of intermittent connections.
    """
    
    def __init__(
        self,
        check_interval_seconds: int = 30,
        quality_check_interval_seconds: int = 300,
        connectivity_timeout_seconds: int = 5,
        enable_quality_monitoring: bool = True,
        test_hosts: List[str] = None
    ):
        """
        Initialize network monitor.
        
        Args:
            check_interval_seconds: Interval between connectivity checks
            quality_check_interval_seconds: Interval between quality assessments
            connectivity_timeout_seconds: Timeout for connectivity tests
            enable_quality_monitoring: Whether to monitor connection quality
            test_hosts: List of hosts to test connectivity against
        """
        self.check_interval_seconds = check_interval_seconds
        self.quality_check_interval_seconds = quality_check_interval_seconds
        self.connectivity_timeout_seconds = connectivity_timeout_seconds
        self.enable_quality_monitoring = enable_quality_monitoring
        
        # Default test hosts (reliable public DNS servers)
        self.test_hosts = test_hosts or [
            "8.8.8.8",      # Google DNS
            "1.1.1.1",      # Cloudflare DNS
            "208.67.222.222" # OpenDNS
        ]
        
        # Current state
        self.current_status = NetworkStatus.OFFLINE
        self.current_quality = ConnectionQuality.UNAVAILABLE
        self.last_check_time = datetime.now()
        self.last_quality_check_time = datetime.now()
        
        # Event tracking
        self.network_events: List[NetworkEvent] = []
        self.event_callbacks: List[Callable[[NetworkEvent], None]] = []
        
        # Statistics
        self.connection_stats = {
            'total_checks': 0,
            'successful_checks': 0,
            'failed_checks': 0,
            'status_changes': 0,
            'total_downtime_seconds': 0.0,
            'total_uptime_seconds': 0.0,
            'average_latency_ms': 0.0,
            'quality_degradations': 0
        }
        
        # Monitoring tasks
        self._monitoring_task = None
        self._quality_task = None
        self._is_monitoring = False
        
        # Recent metrics for trend analysis
        self.recent_metrics: List[NetworkMetrics] = []
        self.max_recent_metrics = 100
        
        logger.info("NetworkMonitor initialized successfully")
    
    async def start_monitoring(self):
        """Start network monitoring tasks."""
        if self._is_monitoring:
            logger.warning("Network monitoring already started")
            return
        
        self._is_monitoring = True
        
        # Start connectivity monitoring
        self._monitoring_task = asyncio.create_task(self._connectivity_monitoring_loop())
        
        # Start quality monitoring if enabled
        if self.enable_quality_monitoring:
            self._quality_task = asyncio.create_task(self._quality_monitoring_loop())
        
        # Perform initial check
        await self.check_connectivity()
        
        logger.info("Network monitoring started")
    
    async def stop_monitoring(self):
        """Stop network monitoring tasks."""
        if not self._is_monitoring:
            return
        
        self._is_monitoring = False
        
        # Cancel monitoring tasks
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self._quality_task and not self._quality_task.done():
            self._quality_task.cancel()
            try:
                await self._quality_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Network monitoring stopped")
    
    async def _connectivity_monitoring_loop(self):
        """Background connectivity monitoring loop."""
        while self._is_monitoring:
            try:
                await self.check_connectivity()
                await asyncio.sleep(self.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connectivity monitoring loop: {e}")
                await asyncio.sleep(self.check_interval_seconds)
    
    async def _quality_monitoring_loop(self):
        """Background quality monitoring loop."""
        while self._is_monitoring:
            try:
                if self.current_status == NetworkStatus.ONLINE:
                    await self.assess_connection_quality()
                await asyncio.sleep(self.quality_check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in quality monitoring loop: {e}")
                await asyncio.sleep(self.quality_check_interval_seconds)
    
    async def check_connectivity(self) -> NetworkStatus:
        """
        Check network connectivity status.
        
        Returns:
            Current network status
        """
        try:
            previous_status = self.current_status
            check_start_time = time.time()
            
            # Test connectivity to multiple hosts
            successful_connections = 0
            total_latency = 0.0
            
            for host in self.test_hosts:
                try:
                    start_time = time.time()
                    
                    # Create socket connection
                    sock = socket.create_connection(
                        (host, 53),  # DNS port
                        timeout=self.connectivity_timeout_seconds
                    )
                    sock.close()
                    
                    end_time = time.time()
                    latency = (end_time - start_time) * 1000  # Convert to ms
                    total_latency += latency
                    successful_connections += 1
                    
                except (socket.error, OSError, socket.timeout):
                    continue
            
            # Determine network status
            if successful_connections == 0:
                self.current_status = NetworkStatus.OFFLINE
            elif successful_connections == len(self.test_hosts):
                self.current_status = NetworkStatus.ONLINE
            elif successful_connections >= len(self.test_hosts) // 2:
                self.current_status = NetworkStatus.LIMITED
            else:
                self.current_status = NetworkStatus.UNSTABLE
            
            # Update statistics
            self.connection_stats['total_checks'] += 1
            if successful_connections > 0:
                self.connection_stats['successful_checks'] += 1
                if successful_connections > 0:
                    avg_latency = total_latency / successful_connections
                    self.connection_stats['average_latency_ms'] = (
                        (self.connection_stats['average_latency_ms'] * 
                         (self.connection_stats['successful_checks'] - 1) + avg_latency) /
                        self.connection_stats['successful_checks']
                    )
            else:
                self.connection_stats['failed_checks'] += 1
            
            # Track status changes
            if previous_status != self.current_status:
                self.connection_stats['status_changes'] += 1
                
                # Calculate uptime/downtime
                check_duration = time.time() - check_start_time
                if previous_status == NetworkStatus.OFFLINE:
                    self.connection_stats['total_downtime_seconds'] += check_duration
                else:
                    self.connection_stats['total_uptime_seconds'] += check_duration
                
                # Create network event
                event = NetworkEvent(
                    event_type="status_changed",
                    timestamp=datetime.now(),
                    previous_status=previous_status,
                    current_status=self.current_status,
                    duration_seconds=check_duration
                )
                
                await self._handle_network_event(event)
                
                logger.info(f"Network status changed: {previous_status} -> {self.current_status}")
            
            self.last_check_time = datetime.now()
            return self.current_status
            
        except Exception as e:
            logger.error(f"Error checking network connectivity: {e}")
            self.current_status = NetworkStatus.OFFLINE
            return self.current_status
    
    async def assess_connection_quality(self) -> ConnectionQuality:
        """
        Assess network connection quality.
        
        Returns:
            Current connection quality
        """
        try:
            if self.current_status == NetworkStatus.OFFLINE:
                self.current_quality = ConnectionQuality.UNAVAILABLE
                return self.current_quality
            
            # Perform quality tests
            latencies = []
            packet_loss_count = 0
            total_tests = 5
            
            for i in range(total_tests):
                try:
                    start_time = time.time()
                    
                    # Test with primary host
                    sock = socket.create_connection(
                        (self.test_hosts[0], 53),
                        timeout=self.connectivity_timeout_seconds
                    )
                    sock.close()
                    
                    end_time = time.time()
                    latency = (end_time - start_time) * 1000
                    latencies.append(latency)
                    
                except (socket.error, OSError, socket.timeout):
                    packet_loss_count += 1
                
                # Small delay between tests
                await asyncio.sleep(0.1)
            
            # Calculate metrics
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                jitter = max(latencies) - min(latencies) if len(latencies) > 1 else 0.0
            else:
                avg_latency = float('inf')
                jitter = 0.0
            
            packet_loss_percent = (packet_loss_count / total_tests) * 100
            
            # Estimate bandwidth (simplified)
            bandwidth_mbps = self._estimate_bandwidth(avg_latency, packet_loss_percent)
            
            # Create metrics
            metrics = NetworkMetrics(
                latency_ms=avg_latency,
                packet_loss_percent=packet_loss_percent,
                bandwidth_mbps=bandwidth_mbps,
                jitter_ms=jitter,
                timestamp=datetime.now()
            )
            
            # Store recent metrics
            self.recent_metrics.append(metrics)
            if len(self.recent_metrics) > self.max_recent_metrics:
                self.recent_metrics.pop(0)
            
            # Determine quality
            previous_quality = self.current_quality
            
            if packet_loss_percent > 10 or avg_latency > 1000:
                self.current_quality = ConnectionQuality.POOR
            elif packet_loss_percent > 5 or avg_latency > 500:
                self.current_quality = ConnectionQuality.FAIR
            elif packet_loss_percent > 1 or avg_latency > 200:
                self.current_quality = ConnectionQuality.GOOD
            else:
                self.current_quality = ConnectionQuality.EXCELLENT
            
            # Track quality changes
            if previous_quality != self.current_quality:
                if (previous_quality in [ConnectionQuality.EXCELLENT, ConnectionQuality.GOOD] and
                    self.current_quality in [ConnectionQuality.FAIR, ConnectionQuality.POOR]):
                    self.connection_stats['quality_degradations'] += 1
                
                # Create quality change event
                event = NetworkEvent(
                    event_type="quality_changed",
                    timestamp=datetime.now(),
                    current_status=self.current_status,
                    metrics=metrics
                )
                
                await self._handle_network_event(event)
                
                logger.info(f"Network quality changed: {previous_quality} -> {self.current_quality}")
            
            self.last_quality_check_time = datetime.now()
            return self.current_quality
            
        except Exception as e:
            logger.error(f"Error assessing connection quality: {e}")
            self.current_quality = ConnectionQuality.UNAVAILABLE
            return self.current_quality
    
    def _estimate_bandwidth(self, latency_ms: float, packet_loss_percent: float) -> float:
        """
        Estimate bandwidth based on latency and packet loss.
        
        Args:
            latency_ms: Average latency in milliseconds
            packet_loss_percent: Packet loss percentage
            
        Returns:
            Estimated bandwidth in Mbps
        """
        try:
            # Simplified bandwidth estimation
            # In production, this could use actual throughput tests
            
            base_bandwidth = 100.0  # Assume 100 Mbps base
            
            # Reduce based on latency
            if latency_ms > 500:
                base_bandwidth *= 0.3
            elif latency_ms > 200:
                base_bandwidth *= 0.6
            elif latency_ms > 100:
                base_bandwidth *= 0.8
            
            # Reduce based on packet loss
            if packet_loss_percent > 5:
                base_bandwidth *= 0.4
            elif packet_loss_percent > 1:
                base_bandwidth *= 0.7
            
            return max(0.1, base_bandwidth)  # Minimum 0.1 Mbps
            
        except Exception as e:
            logger.error(f"Error estimating bandwidth: {e}")
            return 1.0  # Default 1 Mbps
    
    async def _handle_network_event(self, event: NetworkEvent):
        """
        Handle network event by storing it and notifying callbacks.
        
        Args:
            event: Network event to handle
        """
        try:
            # Store event
            self.network_events.append(event)
            
            # Limit event history
            if len(self.network_events) > 1000:
                self.network_events = self.network_events[-500:]  # Keep last 500
            
            # Notify callbacks
            for callback in self.event_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in network event callback: {e}")
            
        except Exception as e:
            logger.error(f"Error handling network event: {e}")
    
    def add_event_callback(self, callback: Callable[[NetworkEvent], None]):
        """
        Add callback for network events.
        
        Args:
            callback: Function to call when network events occur
        """
        self.event_callbacks.append(callback)
    
    def remove_event_callback(self, callback: Callable[[NetworkEvent], None]):
        """
        Remove network event callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)
    
    def get_network_status(self) -> Dict[str, Any]:
        """
        Get current network status and statistics.
        
        Returns:
            Network status information
        """
        try:
            # Calculate uptime percentage
            total_time = (
                self.connection_stats['total_uptime_seconds'] + 
                self.connection_stats['total_downtime_seconds']
            )
            uptime_percentage = (
                (self.connection_stats['total_uptime_seconds'] / total_time * 100)
                if total_time > 0 else 0.0
            )
            
            # Get recent quality trend
            recent_quality_trend = self._analyze_quality_trend()
            
            return {
                'current_status': self.current_status.value,
                'current_quality': self.current_quality.value,
                'last_check_time': self.last_check_time.isoformat(),
                'last_quality_check_time': self.last_quality_check_time.isoformat(),
                'uptime_percentage': uptime_percentage,
                'statistics': self.connection_stats.copy(),
                'quality_trend': recent_quality_trend,
                'recent_events_count': len(self.network_events),
                'monitoring_active': self._is_monitoring
            }
            
        except Exception as e:
            logger.error(f"Error getting network status: {e}")
            return {
                'current_status': self.current_status.value,
                'current_quality': self.current_quality.value,
                'error': str(e)
            }
    
    def _analyze_quality_trend(self) -> str:
        """
        Analyze recent quality trend.
        
        Returns:
            Quality trend description
        """
        try:
            if len(self.recent_metrics) < 3:
                return "insufficient_data"
            
            # Get recent latency values
            recent_latencies = [m.latency_ms for m in self.recent_metrics[-5:]]
            
            if len(recent_latencies) < 3:
                return "stable"
            
            # Calculate trend
            first_half = sum(recent_latencies[:len(recent_latencies)//2]) / (len(recent_latencies)//2)
            second_half = sum(recent_latencies[len(recent_latencies)//2:]) / (len(recent_latencies) - len(recent_latencies)//2)
            
            change_percent = ((second_half - first_half) / first_half) * 100
            
            if change_percent > 20:
                return "degrading"
            elif change_percent < -20:
                return "improving"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error analyzing quality trend: {e}")
            return "unknown"
    
    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent network events.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of recent network events
        """
        try:
            recent_events = self.network_events[-limit:] if self.network_events else []
            
            return [
                {
                    'event_type': event.event_type,
                    'timestamp': event.timestamp.isoformat(),
                    'previous_status': event.previous_status.value if event.previous_status else None,
                    'current_status': event.current_status.value,
                    'duration_seconds': event.duration_seconds,
                    'metrics': {
                        'latency_ms': event.metrics.latency_ms,
                        'packet_loss_percent': event.metrics.packet_loss_percent,
                        'bandwidth_mbps': event.metrics.bandwidth_mbps,
                        'jitter_ms': event.metrics.jitter_ms
                    } if event.metrics else None
                }
                for event in reversed(recent_events)
            ]
            
        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []
    
    async def wait_for_connectivity(self, timeout_seconds: int = 60) -> bool:
        """
        Wait for network connectivity to be restored.
        
        Args:
            timeout_seconds: Maximum time to wait
            
        Returns:
            True if connectivity restored, False if timeout
        """
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout_seconds:
                status = await self.check_connectivity()
                
                if status in [NetworkStatus.ONLINE, NetworkStatus.LIMITED]:
                    return True
                
                await asyncio.sleep(1)  # Check every second
            
            return False
            
        except Exception as e:
            logger.error(f"Error waiting for connectivity: {e}")
            return False
    
    def is_online(self) -> bool:
        """Check if network is currently online."""
        return self.current_status in [NetworkStatus.ONLINE, NetworkStatus.LIMITED]
    
    def is_stable(self) -> bool:
        """Check if network connection is stable."""
        return (
            self.current_status == NetworkStatus.ONLINE and
            self.current_quality in [ConnectionQuality.EXCELLENT, ConnectionQuality.GOOD]
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of network monitor.
        
        Returns:
            Health check result
        """
        try:
            # Check current connectivity
            current_status = await self.check_connectivity()
            
            # Check monitoring tasks
            monitoring_healthy = self._is_monitoring and (
                self._monitoring_task and not self._monitoring_task.done()
            )
            
            quality_monitoring_healthy = (
                not self.enable_quality_monitoring or
                (self._quality_task and not self._quality_task.done())
            )
            
            # Overall health
            overall_health = "healthy" if (
                monitoring_healthy and quality_monitoring_healthy
            ) else "degraded"
            
            return {
                'status': overall_health,
                'network_status': current_status.value,
                'network_quality': self.current_quality.value,
                'monitoring_active': self._is_monitoring,
                'monitoring_task_healthy': monitoring_healthy,
                'quality_monitoring_healthy': quality_monitoring_healthy,
                'last_check_age_seconds': (datetime.now() - self.last_check_time).total_seconds(),
                'statistics': self.connection_stats.copy(),
                'recent_events_count': len(self.network_events)
            }
            
        except Exception as e:
            logger.error(f"Network monitor health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'network_status': self.current_status.value
            }


# Factory function for creating network monitor
def create_network_monitor(
    check_interval_seconds: int = 30,
    quality_check_interval_seconds: int = 300,
    connectivity_timeout_seconds: int = 5,
    enable_quality_monitoring: bool = True,
    test_hosts: List[str] = None
) -> NetworkMonitor:
    """
    Factory function to create a network monitor instance.
    
    Args:
        check_interval_seconds: Interval between connectivity checks
        quality_check_interval_seconds: Interval between quality assessments
        connectivity_timeout_seconds: Timeout for connectivity tests
        enable_quality_monitoring: Whether to monitor connection quality
        test_hosts: List of hosts to test connectivity against
        
    Returns:
        Configured NetworkMonitor instance
    """
    return NetworkMonitor(
        check_interval_seconds=check_interval_seconds,
        quality_check_interval_seconds=quality_check_interval_seconds,
        connectivity_timeout_seconds=connectivity_timeout_seconds,
        enable_quality_monitoring=enable_quality_monitoring,
        test_hosts=test_hosts
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    )