"""
Network simulation utilities for testing under realistic Indian network conditions.

This module provides tools to simulate various network scenarios common in India,
including slow connections, intermittent connectivity, high latency, and bandwidth limitations.
"""

import asyncio
import random
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import structlog


logger = structlog.get_logger(__name__)


class NetworkType(Enum):
    """Network type enumeration for Indian conditions."""
    FIBER_URBAN = "fiber_urban"          # High-speed fiber in metros
    BROADBAND_URBAN = "broadband_urban"  # ADSL/Cable in cities
    MOBILE_4G = "mobile_4g"              # 4G mobile networks
    MOBILE_3G = "mobile_3g"              # 3G mobile networks
    MOBILE_2G = "mobile_2g"              # 2G/EDGE networks
    SATELLITE = "satellite"              # Satellite internet
    RURAL_BROADBAND = "rural_broadband"  # Rural broadband connections


@dataclass
class NetworkCondition:
    """Network condition parameters."""
    bandwidth_kbps: float
    latency_ms: float
    packet_loss: float
    jitter_ms: float
    stability: float  # 0.0 to 1.0, where 1.0 is perfectly stable
    connection_drops: float  # Probability of connection drops per minute


class NetworkSimulator:
    """
    Network simulator for testing under various Indian network conditions.
    """
    
    # Predefined network conditions based on Indian network scenarios
    NETWORK_CONDITIONS = {
        NetworkType.FIBER_URBAN: NetworkCondition(
            bandwidth_kbps=50000,  # 50 Mbps
            latency_ms=20,
            packet_loss=0.001,
            jitter_ms=5,
            stability=0.98,
            connection_drops=0.01
        ),
        NetworkType.BROADBAND_URBAN: NetworkCondition(
            bandwidth_kbps=8000,  # 8 Mbps
            latency_ms=50,
            packet_loss=0.01,
            jitter_ms=20,
            stability=0.95,
            connection_drops=0.02
        ),
        NetworkType.MOBILE_4G: NetworkCondition(
            bandwidth_kbps=5000,  # 5 Mbps
            latency_ms=100,
            packet_loss=0.02,
            jitter_ms=50,
            stability=0.90,
            connection_drops=0.05
        ),
        NetworkType.MOBILE_3G: NetworkCondition(
            bandwidth_kbps=1000,  # 1 Mbps
            latency_ms=200,
            packet_loss=0.05,
            jitter_ms=100,
            stability=0.85,
            connection_drops=0.10
        ),
        NetworkType.MOBILE_2G: NetworkCondition(
            bandwidth_kbps=64,  # 64 Kbps
            latency_ms=800,
            packet_loss=0.10,
            jitter_ms=200,
            stability=0.70,
            connection_drops=0.20
        ),
        NetworkType.SATELLITE: NetworkCondition(
            bandwidth_kbps=2000,  # 2 Mbps
            latency_ms=600,  # Satellite latency
            packet_loss=0.03,
            jitter_ms=100,
            stability=0.80,
            connection_drops=0.15
        ),
        NetworkType.RURAL_BROADBAND: NetworkCondition(
            bandwidth_kbps=512,  # 512 Kbps
            latency_ms=300,
            packet_loss=0.08,
            jitter_ms=150,
            stability=0.75,
            connection_drops=0.25
        )
    }
    
    def __init__(self, network_type: NetworkType = NetworkType.MOBILE_3G):
        """
        Initialize network simulator.
        
        Args:
            network_type: Type of network to simulate
        """
        self.network_type = network_type
        self.condition = self.NETWORK_CONDITIONS[network_type]
        self.is_connected = True
        self.connection_start_time = time.time()
        self.total_bytes_transferred = 0
        self.request_history: List[Dict[str, Any]] = []
    
    async def simulate_request_delay(self, payload_size_bytes: int) -> Dict[str, Any]:
        """
        Simulate network delay for a request based on current conditions.
        
        Args:
            payload_size_bytes: Size of the request payload
            
        Returns:
            Dictionary with delay information
        """
        # Calculate base transmission time
        transmission_time = (payload_size_bytes * 8) / (self.condition.bandwidth_kbps * 1000)
        
        # Add latency
        base_delay = self.condition.latency_ms / 1000.0
        
        # Add jitter (random variation)
        jitter = random.uniform(0, self.condition.jitter_ms / 1000.0)
        
        # Simulate packet loss (causes retransmissions)
        if random.random() < self.condition.packet_loss:
            retransmission_delay = transmission_time * 2  # Simplified retransmission
            logger.warning("Packet loss simulated", delay=retransmission_delay)
        else:
            retransmission_delay = 0
        
        # Total delay
        total_delay = transmission_time + base_delay + jitter + retransmission_delay
        
        # Simulate connection instability
        if random.random() > self.condition.stability:
            instability_delay = random.uniform(0.5, 2.0)
            total_delay += instability_delay
            logger.warning("Network instability simulated", delay=instability_delay)
        
        # Apply the delay
        await asyncio.sleep(total_delay)
        
        # Update statistics
        self.total_bytes_transferred += payload_size_bytes
        
        delay_info = {
            "transmission_time": transmission_time,
            "latency": base_delay,
            "jitter": jitter,
            "retransmission_delay": retransmission_delay,
            "total_delay": total_delay,
            "payload_size_bytes": payload_size_bytes,
            "effective_bandwidth_kbps": (payload_size_bytes * 8) / (total_delay * 1000) if total_delay > 0 else 0
        }
        
        self.request_history.append(delay_info)
        return delay_info
    
    async def simulate_connection_drop(self) -> bool:
        """
        Simulate connection drops based on network stability.
        
        Returns:
            True if connection dropped, False otherwise
        """
        # Calculate probability based on time since last connection
        time_connected = time.time() - self.connection_start_time
        drop_probability = self.condition.connection_drops * (time_connected / 60.0)  # Per minute
        
        if random.random() < drop_probability:
            self.is_connected = False
            drop_duration = random.uniform(1.0, 10.0)  # 1-10 seconds
            
            logger.warning(
                "Connection drop simulated",
                duration=drop_duration,
                network_type=self.network_type.value
            )
            
            await asyncio.sleep(drop_duration)
            
            # Reconnect
            self.is_connected = True
            self.connection_start_time = time.time()
            
            logger.info("Connection restored", network_type=self.network_type.value)
            return True
        
        return False
    
    def get_current_conditions(self) -> Dict[str, Any]:
        """Get current network conditions."""
        return {
            "network_type": self.network_type.value,
            "bandwidth_kbps": self.condition.bandwidth_kbps,
            "latency_ms": self.condition.latency_ms,
            "packet_loss": self.condition.packet_loss,
            "jitter_ms": self.condition.jitter_ms,
            "stability": self.condition.stability,
            "connection_drops_per_minute": self.condition.connection_drops,
            "is_connected": self.is_connected,
            "uptime_seconds": time.time() - self.connection_start_time,
            "total_bytes_transferred": self.total_bytes_transferred
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from request history."""
        if not self.request_history:
            return {"no_data": True}
        
        delays = [r["total_delay"] for r in self.request_history]
        bandwidths = [r["effective_bandwidth_kbps"] for r in self.request_history]
        
        return {
            "total_requests": len(self.request_history),
            "average_delay_ms": sum(delays) / len(delays) * 1000,
            "max_delay_ms": max(delays) * 1000,
            "min_delay_ms": min(delays) * 1000,
            "average_bandwidth_kbps": sum(bandwidths) / len(bandwidths),
            "total_bytes_transferred": self.total_bytes_transferred,
            "packet_loss_events": len([r for r in self.request_history if r["retransmission_delay"] > 0])
        }


class IndianNetworkScenarios:
    """
    Predefined network scenarios common in India.
    """
    
    @staticmethod
    def metro_city_office() -> NetworkSimulator:
        """High-speed connection in metro city office."""
        return NetworkSimulator(NetworkType.FIBER_URBAN)
    
    @staticmethod
    def urban_home() -> NetworkSimulator:
        """Typical urban home broadband."""
        return NetworkSimulator(NetworkType.BROADBAND_URBAN)
    
    @staticmethod
    def mobile_commute() -> NetworkSimulator:
        """Mobile connection during commute (4G)."""
        return NetworkSimulator(NetworkType.MOBILE_4G)
    
    @staticmethod
    def rural_area() -> NetworkSimulator:
        """Rural area connection."""
        return NetworkSimulator(NetworkType.RURAL_BROADBAND)
    
    @staticmethod
    def remote_location() -> NetworkSimulator:
        """Remote location with satellite internet."""
        return NetworkSimulator(NetworkType.SATELLITE)
    
    @staticmethod
    def low_end_mobile() -> NetworkSimulator:
        """Low-end mobile device with 2G/3G."""
        return NetworkSimulator(NetworkType.MOBILE_3G)
    
    @staticmethod
    def congested_network() -> NetworkSimulator:
        """Congested network during peak hours."""
        simulator = NetworkSimulator(NetworkType.MOBILE_3G)
        # Modify conditions for congestion
        simulator.condition.bandwidth_kbps *= 0.3  # 30% of normal bandwidth
        simulator.condition.latency_ms *= 2  # Double latency
        simulator.condition.packet_loss *= 3  # Triple packet loss
        simulator.condition.stability *= 0.8  # Reduced stability
        return simulator


class NetworkTestDecorator:
    """
    Decorator for applying network simulation to test functions.
    """
    
    def __init__(self, network_type: NetworkType, simulate_payload: bool = True):
        """
        Initialize network test decorator.
        
        Args:
            network_type: Type of network to simulate
            simulate_payload: Whether to simulate payload-based delays
        """
        self.network_type = network_type
        self.simulate_payload = simulate_payload
    
    def __call__(self, func: Callable) -> Callable:
        """Apply network simulation to test function."""
        async def wrapper(*args, **kwargs):
            simulator = NetworkSimulator(self.network_type)
            
            # Add simulator to kwargs if not present
            if 'network_simulator' not in kwargs:
                kwargs['network_simulator'] = simulator
            
            # Simulate initial connection delay
            await simulator.simulate_request_delay(1024)  # 1KB initial handshake
            
            # Check for connection drops
            await simulator.simulate_connection_drop()
            
            try:
                result = await func(*args, **kwargs)
                
                # Log performance metrics
                metrics = simulator.get_performance_metrics()
                logger.info(
                    "Network simulation completed",
                    network_type=self.network_type.value,
                    metrics=metrics
                )
                
                return result
            
            except Exception as e:
                logger.error(
                    "Test failed under network simulation",
                    network_type=self.network_type.value,
                    error=str(e)
                )
                raise
        
        return wrapper


# Convenience decorators for common scenarios
def test_with_metro_network(func: Callable) -> Callable:
    """Test with metro city network conditions."""
    return NetworkTestDecorator(NetworkType.FIBER_URBAN)(func)


def test_with_rural_network(func: Callable) -> Callable:
    """Test with rural network conditions."""
    return NetworkTestDecorator(NetworkType.RURAL_BROADBAND)(func)


def test_with_mobile_network(func: Callable) -> Callable:
    """Test with mobile network conditions."""
    return NetworkTestDecorator(NetworkType.MOBILE_3G)(func)


def test_with_slow_network(func: Callable) -> Callable:
    """Test with slow network conditions."""
    return NetworkTestDecorator(NetworkType.MOBILE_2G)(func)


async def benchmark_network_conditions():
    """
    Benchmark all network conditions for comparison.
    
    Returns:
        Dictionary with benchmark results for all network types
    """
    results = {}
    test_payload_sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB
    
    for network_type in NetworkType:
        simulator = NetworkSimulator(network_type)
        network_results = []
        
        for payload_size in test_payload_sizes:
            delay_info = await simulator.simulate_request_delay(payload_size)
            network_results.append(delay_info)
        
        results[network_type.value] = {
            "conditions": simulator.get_current_conditions(),
            "performance": simulator.get_performance_metrics(),
            "test_results": network_results
        }
    
    return results


if __name__ == "__main__":
    """Run network simulation benchmark."""
    async def main():
        print("Benchmarking Indian network conditions...")
        results = await benchmark_network_conditions()
        
        for network_type, data in results.items():
            print(f"\n{network_type.upper()}:")
            conditions = data["conditions"]
            performance = data["performance"]
            
            print(f"  Bandwidth: {conditions['bandwidth_kbps']} kbps")
            print(f"  Latency: {conditions['latency_ms']} ms")
            print(f"  Packet Loss: {conditions['packet_loss']*100:.1f}%")
            print(f"  Average Delay: {performance['average_delay_ms']:.1f} ms")
            print(f"  Effective Bandwidth: {performance['average_bandwidth_kbps']:.1f} kbps")
    
    asyncio.run(main())