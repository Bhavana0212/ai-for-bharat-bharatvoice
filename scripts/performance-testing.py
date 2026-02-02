#!/usr/bin/env python3
"""
Performance Testing Script for BharatVoice Assistant

This script conducts comprehensive performance testing including load testing,
stress testing, and performance profiling for the production deployment.
"""

import asyncio
import aiohttp
import time
import json
import statistics
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
import psutil
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result data structure."""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    payload_size: int
    response_size: int
    timestamp: datetime
    error: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    error_rate: float
    throughput_mbps: float


class PerformanceTester:
    """
    Comprehensive performance testing suite for BharatVoice Assistant.
    """
    
    def __init__(self, base_url: str, max_concurrent: int = 100):
        self.base_url = base_url.rstrip('/')
        self.max_concurrent = max_concurrent
        self.session: Optional[aiohttp.ClientSession] = None
        self.results: List[TestResult] = []
        
        # Test scenarios
        self.test_scenarios = {
            'health_check': {
                'method': 'GET',
                'endpoint': '/health/ready',
                'payload': None,
                'weight': 20
            },
            'voice_synthesis': {
                'method': 'POST',
                'endpoint': '/api/voice/synthesize',
                'payload': {
                    'text': 'नमस्ते, मुझे दिल्ली से मुंबई की ट्रेन की जानकारी चाहिए',
                    'language': 'hi-IN',
                    'voice_settings': {'speed': 1.0, 'pitch': 1.0}
                },
                'weight': 30
            },
            'train_search': {
                'method': 'GET',
                'endpoint': '/api/railways/trains',
                'params': {'from': 'NDLS', 'to': 'CSTM', 'date': '2024-12-01'},
                'payload': None,
                'weight': 25
            },
            'weather_info': {
                'method': 'GET',
                'endpoint': '/api/weather/current',
                'params': {'city': 'Delhi'},
                'payload': None,
                'weight': 15
            },
            'platform_search': {
                'method': 'GET',
                'endpoint': '/api/platforms/food/restaurants',
                'params': {'location': '28.6139,77.2090', 'cuisine': 'North Indian'},
                'payload': None,
                'weight': 10
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=self.max_concurrent,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'BharatVoice-PerformanceTester/1.0'}
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def make_request(self, scenario_name: str, scenario: Dict[str, Any]) -> TestResult:
        """Make a single HTTP request and record metrics."""
        start_time = time.time()
        timestamp = datetime.now()
        
        try:
            url = f"{self.base_url}{scenario['endpoint']}"
            method = scenario['method']
            
            kwargs = {}
            if scenario.get('payload'):
                kwargs['json'] = scenario['payload']
            if scenario.get('params'):
                kwargs['params'] = scenario['params']
            
            payload_size = len(json.dumps(scenario.get('payload', {})).encode())
            
            async with self.session.request(method, url, **kwargs) as response:
                response_data = await response.read()
                response_time = time.time() - start_time
                response_size = len(response_data)
                
                return TestResult(
                    endpoint=scenario['endpoint'],
                    method=method,
                    status_code=response.status,
                    response_time=response_time,
                    payload_size=payload_size,
                    response_size=response_size,
                    timestamp=timestamp,
                    error=None if response.status < 400 else f"HTTP {response.status}"
                )
        
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                endpoint=scenario['endpoint'],
                method=scenario['method'],
                status_code=0,
                response_time=response_time,
                payload_size=0,
                response_size=0,
                timestamp=timestamp,
                error=str(e)
            )
    
    async def run_load_test(
        self,
        duration_seconds: int = 300,
        requests_per_second: int = 10,
        ramp_up_seconds: int = 60
    ) -> List[TestResult]:
        """
        Run load test with gradual ramp-up.
        
        Args:
            duration_seconds: Total test duration
            requests_per_second: Target requests per second
            ramp_up_seconds: Time to reach target RPS
        """
        logger.info(f"Starting load test: {requests_per_second} RPS for {duration_seconds}s")
        
        results = []
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Calculate ramp-up rate
        ramp_up_rate = requests_per_second / ramp_up_seconds if ramp_up_seconds > 0 else requests_per_second
        
        request_count = 0
        
        while time.time() < end_time:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Calculate current target RPS based on ramp-up
            if elapsed < ramp_up_seconds:
                current_rps = min(ramp_up_rate * elapsed, requests_per_second)
            else:
                current_rps = requests_per_second
            
            # Calculate how many requests we should have made by now
            target_requests = int(current_rps * elapsed)
            
            # Make requests to catch up to target
            requests_to_make = max(0, target_requests - request_count)
            
            if requests_to_make > 0:
                # Create semaphore to limit concurrent requests
                semaphore = asyncio.Semaphore(min(requests_to_make, self.max_concurrent))
                
                # Create tasks for concurrent requests
                tasks = []
                for _ in range(requests_to_make):
                    scenario_name, scenario = self._select_scenario()
                    task = self._make_request_with_semaphore(semaphore, scenario_name, scenario)
                    tasks.append(task)
                
                # Execute requests concurrently
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in batch_results:
                    if isinstance(result, TestResult):
                        results.append(result)
                        request_count += 1
            
            # Sleep briefly to avoid busy waiting
            await asyncio.sleep(0.1)
        
        logger.info(f"Load test completed: {len(results)} requests made")
        return results
    
    async def _make_request_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        scenario_name: str,
        scenario: Dict[str, Any]
    ) -> TestResult:
        """Make request with semaphore for concurrency control."""
        async with semaphore:
            return await self.make_request(scenario_name, scenario)
    
    def _select_scenario(self) -> tuple:
        """Select a test scenario based on weights."""
        import random
        
        # Create weighted list
        weighted_scenarios = []
        for name, scenario in self.test_scenarios.items():
            weight = scenario.get('weight', 1)
            weighted_scenarios.extend([(name, scenario)] * weight)
        
        return random.choice(weighted_scenarios)
    
    async def run_stress_test(
        self,
        max_rps: int = 1000,
        step_duration: int = 60,
        step_size: int = 50
    ) -> List[TestResult]:
        """
        Run stress test with increasing load until failure.
        
        Args:
            max_rps: Maximum requests per second to test
            step_duration: Duration of each load step
            step_size: RPS increase per step
        """
        logger.info(f"Starting stress test: up to {max_rps} RPS")
        
        all_results = []
        current_rps = step_size
        
        while current_rps <= max_rps:
            logger.info(f"Testing {current_rps} RPS for {step_duration}s")
            
            # Run load test for this step
            step_results = await self.run_load_test(
                duration_seconds=step_duration,
                requests_per_second=current_rps,
                ramp_up_seconds=10
            )
            
            all_results.extend(step_results)
            
            # Analyze step results
            step_metrics = self.calculate_metrics(step_results)
            
            logger.info(f"Step results - RPS: {current_rps}, "
                       f"Success Rate: {(1 - step_metrics.error_rate) * 100:.1f}%, "
                       f"Avg Response Time: {step_metrics.average_response_time:.3f}s")
            
            # Check if we should stop (high error rate or response time)
            if step_metrics.error_rate > 0.1 or step_metrics.average_response_time > 5.0:
                logger.warning(f"Stopping stress test at {current_rps} RPS due to high error rate or response time")
                break
            
            current_rps += step_size
            
            # Brief pause between steps
            await asyncio.sleep(5)
        
        logger.info(f"Stress test completed: {len(all_results)} total requests")
        return all_results
    
    async def run_spike_test(
        self,
        baseline_rps: int = 10,
        spike_rps: int = 500,
        spike_duration: int = 30,
        total_duration: int = 300
    ) -> List[TestResult]:
        """
        Run spike test with sudden load increases.
        
        Args:
            baseline_rps: Normal load RPS
            spike_rps: Spike load RPS
            spike_duration: Duration of each spike
            total_duration: Total test duration
        """
        logger.info(f"Starting spike test: {baseline_rps} RPS baseline with {spike_rps} RPS spikes")
        
        all_results = []
        start_time = time.time()
        end_time = start_time + total_duration
        
        spike_interval = 120  # Spike every 2 minutes
        last_spike_time = 0
        
        while time.time() < end_time:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Determine if we should spike
            if elapsed - last_spike_time >= spike_interval:
                # Run spike
                logger.info(f"Starting spike: {spike_rps} RPS for {spike_duration}s")
                spike_results = await self.run_load_test(
                    duration_seconds=spike_duration,
                    requests_per_second=spike_rps,
                    ramp_up_seconds=5
                )
                all_results.extend(spike_results)
                last_spike_time = elapsed
                
                # Brief recovery period
                await asyncio.sleep(10)
            else:
                # Run baseline load
                baseline_duration = min(30, spike_interval - (elapsed - last_spike_time))
                baseline_results = await self.run_load_test(
                    duration_seconds=int(baseline_duration),
                    requests_per_second=baseline_rps,
                    ramp_up_seconds=5
                )
                all_results.extend(baseline_results)
        
        logger.info(f"Spike test completed: {len(all_results)} total requests")
        return all_results
    
    def calculate_metrics(self, results: List[TestResult]) -> PerformanceMetrics:
        """Calculate performance metrics from test results."""
        if not results:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Filter successful requests
        successful_results = [r for r in results if r.error is None and r.status_code < 400]
        failed_results = [r for r in results if r.error is not None or r.status_code >= 400]
        
        # Response times
        response_times = [r.response_time for r in successful_results]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = self._percentile(response_times, 95)
            p99_response_time = self._percentile(response_times, 99)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = median_response_time = p95_response_time = p99_response_time = 0
            min_response_time = max_response_time = 0
        
        # Calculate RPS
        if results:
            time_span = (max(r.timestamp for r in results) - min(r.timestamp for r in results)).total_seconds()
            requests_per_second = len(results) / max(time_span, 1)
        else:
            requests_per_second = 0
        
        # Calculate throughput
        total_bytes = sum(r.response_size for r in successful_results)
        throughput_mbps = (total_bytes * 8) / (1024 * 1024 * max(time_span, 1)) if results else 0
        
        return PerformanceMetrics(
            total_requests=len(results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            average_response_time=avg_response_time,
            median_response_time=median_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            requests_per_second=requests_per_second,
            error_rate=len(failed_results) / len(results),
            throughput_mbps=throughput_mbps
        )
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def generate_report(self, results: List[TestResult], test_type: str) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        metrics = self.calculate_metrics(results)
        
        # Group results by endpoint
        endpoint_results = {}
        for result in results:
            if result.endpoint not in endpoint_results:
                endpoint_results[result.endpoint] = []
            endpoint_results[result.endpoint].append(result)
        
        # Calculate per-endpoint metrics
        endpoint_metrics = {}
        for endpoint, endpoint_result_list in endpoint_results.items():
            endpoint_metrics[endpoint] = self.calculate_metrics(endpoint_result_list)
        
        # System resource usage (if available)
        system_metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
        }
        
        report = {
            'test_type': test_type,
            'timestamp': datetime.now().isoformat(),
            'overall_metrics': {
                'total_requests': metrics.total_requests,
                'successful_requests': metrics.successful_requests,
                'failed_requests': metrics.failed_requests,
                'success_rate': (1 - metrics.error_rate) * 100,
                'average_response_time': metrics.average_response_time,
                'median_response_time': metrics.median_response_time,
                'p95_response_time': metrics.p95_response_time,
                'p99_response_time': metrics.p99_response_time,
                'min_response_time': metrics.min_response_time,
                'max_response_time': metrics.max_response_time,
                'requests_per_second': metrics.requests_per_second,
                'throughput_mbps': metrics.throughput_mbps
            },
            'endpoint_metrics': {
                endpoint: {
                    'total_requests': em.total_requests,
                    'success_rate': (1 - em.error_rate) * 100,
                    'average_response_time': em.average_response_time,
                    'p95_response_time': em.p95_response_time,
                    'requests_per_second': em.requests_per_second
                }
                for endpoint, em in endpoint_metrics.items()
            },
            'system_metrics': system_metrics,
            'recommendations': self._generate_recommendations(metrics, endpoint_metrics)
        }
        
        return report
    
    def _generate_recommendations(
        self,
        overall_metrics: PerformanceMetrics,
        endpoint_metrics: Dict[str, PerformanceMetrics]
    ) -> List[str]:
        """Generate performance recommendations based on test results."""
        recommendations = []
        
        # Overall performance recommendations
        if overall_metrics.error_rate > 0.05:
            recommendations.append(f"High error rate ({overall_metrics.error_rate * 100:.1f}%) - investigate error causes and improve error handling")
        
        if overall_metrics.average_response_time > 2.0:
            recommendations.append(f"High average response time ({overall_metrics.average_response_time:.2f}s) - consider performance optimization")
        
        if overall_metrics.p95_response_time > 5.0:
            recommendations.append(f"High P95 response time ({overall_metrics.p95_response_time:.2f}s) - investigate slow requests")
        
        # Endpoint-specific recommendations
        for endpoint, metrics in endpoint_metrics.items():
            if metrics.error_rate > 0.1:
                recommendations.append(f"Endpoint {endpoint} has high error rate ({metrics.error_rate * 100:.1f}%)")
            
            if metrics.average_response_time > 3.0:
                recommendations.append(f"Endpoint {endpoint} has slow response time ({metrics.average_response_time:.2f}s)")
        
        # Scaling recommendations
        if overall_metrics.requests_per_second < 50:
            recommendations.append("Low throughput - consider horizontal scaling or performance optimization")
        
        if not recommendations:
            recommendations.append("Performance looks good! Consider running stress tests to find limits.")
        
        return recommendations
    
    def save_results(self, results: List[TestResult], filename: str):
        """Save test results to file."""
        data = []
        for result in results:
            data.append({
                'timestamp': result.timestamp.isoformat(),
                'endpoint': result.endpoint,
                'method': result.method,
                'status_code': result.status_code,
                'response_time': result.response_time,
                'payload_size': result.payload_size,
                'response_size': result.response_size,
                'error': result.error
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
    
    def create_visualizations(self, results: List[TestResult], output_dir: str = "performance_charts"):
        """Create performance visualization charts."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert results to DataFrame
        data = []
        for result in results:
            data.append({
                'timestamp': result.timestamp,
                'endpoint': result.endpoint,
                'response_time': result.response_time,
                'status_code': result.status_code,
                'success': result.error is None and result.status_code < 400
            })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            logger.warning("No data to visualize")
            return
        
        # Response time over time
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['response_time'])
        plt.title('Response Time Over Time')
        plt.xlabel('Time')
        plt.ylabel('Response Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/response_time_timeline.png")
        plt.close()
        
        # Response time distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['response_time'], bins=50, alpha=0.7)
        plt.title('Response Time Distribution')
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/response_time_distribution.png")
        plt.close()
        
        # Success rate by endpoint
        endpoint_success = df.groupby('endpoint')['success'].agg(['count', 'sum']).reset_index()
        endpoint_success['success_rate'] = endpoint_success['sum'] / endpoint_success['count'] * 100
        
        plt.figure(figsize=(12, 6))
        plt.bar(endpoint_success['endpoint'], endpoint_success['success_rate'])
        plt.title('Success Rate by Endpoint')
        plt.xlabel('Endpoint')
        plt.ylabel('Success Rate (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/success_rate_by_endpoint.png")
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}/")


async def main():
    """Main function for performance testing."""
    parser = argparse.ArgumentParser(description='BharatVoice Performance Testing')
    parser.add_argument('--url', default='https://api.bharatvoice.com', help='Base URL to test')
    parser.add_argument('--test-type', choices=['load', 'stress', 'spike'], default='load', help='Type of test to run')
    parser.add_argument('--duration', type=int, default=300, help='Test duration in seconds')
    parser.add_argument('--rps', type=int, default=10, help='Requests per second')
    parser.add_argument('--max-concurrent', type=int, default=100, help='Maximum concurrent requests')
    parser.add_argument('--output', default='performance_results.json', help='Output file for results')
    parser.add_argument('--visualize', action='store_true', help='Create visualization charts')
    
    args = parser.parse_args()
    
    logger.info(f"Starting {args.test_type} test against {args.url}")
    
    async with PerformanceTester(args.url, args.max_concurrent) as tester:
        if args.test_type == 'load':
            results = await tester.run_load_test(
                duration_seconds=args.duration,
                requests_per_second=args.rps
            )
        elif args.test_type == 'stress':
            results = await tester.run_stress_test(
                max_rps=args.rps * 10,
                step_duration=60,
                step_size=args.rps
            )
        elif args.test_type == 'spike':
            results = await tester.run_spike_test(
                baseline_rps=args.rps,
                spike_rps=args.rps * 10,
                total_duration=args.duration
            )
        
        # Generate and display report
        report = tester.generate_report(results, args.test_type)
        
        print("\n" + "="*60)
        print(f"PERFORMANCE TEST REPORT - {args.test_type.upper()}")
        print("="*60)
        print(f"Total Requests: {report['overall_metrics']['total_requests']}")
        print(f"Success Rate: {report['overall_metrics']['success_rate']:.1f}%")
        print(f"Average Response Time: {report['overall_metrics']['average_response_time']:.3f}s")
        print(f"P95 Response Time: {report['overall_metrics']['p95_response_time']:.3f}s")
        print(f"Requests/Second: {report['overall_metrics']['requests_per_second']:.1f}")
        print(f"Throughput: {report['overall_metrics']['throughput_mbps']:.2f} Mbps")
        
        print("\nRECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"- {rec}")
        
        # Save results
        tester.save_results(results, args.output)
        
        # Save report
        report_file = args.output.replace('.json', '_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create visualizations if requested
        if args.visualize:
            tester.create_visualizations(results)
        
        logger.info("Performance testing completed successfully")


if __name__ == "__main__":
    asyncio.run(main())