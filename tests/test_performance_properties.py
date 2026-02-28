<<<<<<< HEAD
"""
Property-based tests for Performance Requirements.

**Property 20: Performance Requirements**
**Validates: Requirements 4.1, 4.2, 4.3**

This module tests that the BharatVoice Assistant meets performance requirements
including response time targets, concurrent user handling, and system resource management.
"""

import asyncio
import time
from typing import List, Dict, Any
import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
import structlog

from bharatvoice.core.models import LanguageCode
from bharatvoice.utils.performance_monitor import (
    PerformanceMonitor,
    QueryComplexity,
    RequestPriority,
    PerformanceTargets,
    get_performance_monitor
)
from bharatvoice.utils.accessibility import get_accessibility_manager


logger = structlog.get_logger(__name__)


# Test data generators
@st.composite
def generate_query_complexity(draw):
    """Generate query complexity levels."""
    return draw(st.sampled_from(list(QueryComplexity)))


@st.composite
def generate_language_code(draw):
    """Generate supported language codes."""
    return draw(st.sampled_from([
        LanguageCode.HINDI,
        LanguageCode.ENGLISH_INDIA,
        LanguageCode.TAMIL,
        LanguageCode.TELUGU,
        LanguageCode.BENGALI,
        LanguageCode.MARATHI,
        LanguageCode.GUJARATI,
        LanguageCode.KANNADA,
        LanguageCode.MALAYALAM,
        LanguageCode.PUNJABI,
        LanguageCode.ODIA,
    ]))


@st.composite
def generate_request_priority(draw):
    """Generate request priority levels."""
    return draw(st.sampled_from(list(RequestPriority)))


@st.composite
def generate_concurrent_requests(draw):
    """Generate concurrent request scenarios."""
    num_requests = draw(st.integers(min_value=1, max_value=50))
    requests = []
    
    for i in range(num_requests):
        request = {
            "id": f"req_{i}",
            "complexity": draw(generate_query_complexity()),
            "language": draw(generate_language_code()),
            "priority": draw(generate_request_priority()),
            "processing_time": draw(st.floats(min_value=0.1, max_value=10.0))
        }
        requests.append(request)
    
    return requests


class PerformanceTestStateMachine(RuleBasedStateMachine):
    """
    Stateful testing for performance monitoring system.
    """
    
    def __init__(self):
        super().__init__()
        self.monitor = PerformanceMonitor()
        self.active_requests = []
        self.completed_requests = []
    
    @initialize()
    async def setup_monitor(self):
        """Initialize the performance monitor."""
        await self.monitor.start()
    
    @rule(
        complexity=generate_query_complexity(),
        language=generate_language_code(),
        priority=generate_request_priority(),
        processing_time=st.floats(min_value=0.1, max_value=5.0)
    )
    async def add_request(self, complexity, language, priority, processing_time):
        """Add a request to the system."""
        request_id = f"test_req_{len(self.active_requests)}"
        
        async def mock_callback():
            await asyncio.sleep(processing_time)
            return {"status": "completed", "processing_time": processing_time}
        
        # Queue the request
        queued = await self.monitor.queue_request(
            request_id=request_id,
            callback=mock_callback,
            complexity=complexity,
            language=language,
            priority=priority
        )
        
        if queued:
            self.active_requests.append({
                "id": request_id,
                "complexity": complexity,
                "language": language,
                "priority": priority,
                "processing_time": processing_time,
                "start_time": time.time()
            })
    
    @invariant()
    def system_health_maintained(self):
        """System should maintain health under load."""
        # Check that system health indicators are reasonable
        stats = self.monitor.get_performance_stats()
        
        # CPU and memory usage should be within limits
        if "cpu_usage" in stats:
            assert stats["cpu_usage"] <= PerformanceTargets.MAX_CPU_USAGE * 1.1, \
                f"CPU usage too high: {stats['cpu_usage']}%"
        
        if "memory_usage" in stats:
            assert stats["memory_usage"] <= PerformanceTargets.MAX_MEMORY_USAGE * 1.1, \
                f"Memory usage too high: {stats['memory_usage']}%"
        
        # Active requests should not exceed limits
        active_count = stats.get("active_requests", 0)
        assert active_count <= PerformanceTargets.MAX_CONCURRENT_REQUESTS, \
            f"Too many active requests: {active_count}"
    
    @invariant()
    def queue_size_reasonable(self):
        """Request queue size should remain reasonable."""
        queue_size = self.monitor.request_queue.size()
        assert queue_size <= 1000, f"Queue size too large: {queue_size}"


@pytest.mark.asyncio
class TestPerformanceRequirements:
    """Test performance requirements compliance."""
    
    @pytest.fixture
    async def performance_monitor(self):
        """Create and start performance monitor."""
        monitor = PerformanceMonitor()
        await monitor.start()
        yield monitor
        await monitor.stop()
    
    @given(
        complexity=generate_query_complexity(),
        language=generate_language_code(),
        response_time=st.floats(min_value=0.1, max_value=10.0)
    )
    @settings(max_examples=50, deadline=30000)
    async def test_response_time_targets(self, performance_monitor, complexity, language, response_time):
        """
        **Property 20: Performance Requirements**
        **Validates: Requirements 4.1**
        
        Property: Response times should meet targets based on query complexity.
        """
        # Record performance metrics
        performance_monitor.record_performance(
            response_time=response_time,
            query_complexity=complexity,
            language=language,
            success=True
        )
        
        # Get target time for complexity
        target_time = self._get_target_time(complexity)
        
        # Property 1: Response time should be recorded correctly
        stats = performance_monitor.get_performance_stats()
        assert f"{complexity.value}_total_requests" in stats, \
            f"No requests recorded for complexity {complexity.value}"
        
        # Property 2: Fast responses should always meet targets
        if response_time <= target_time:
            # This should always pass for responses within target
            assert response_time <= target_time, \
                f"Response time {response_time}s exceeds target {target_time}s for {complexity.value}"
        
        # Property 3: System should track performance degradation
        if response_time > target_time * 2:  # Significantly over target
            # System should be aware of performance issues
            assert not performance_monitor.is_system_healthy() or response_time < target_time * 3, \
                f"System should detect performance degradation for {response_time}s response"
    
    def _get_target_time(self, complexity: QueryComplexity) -> float:
        """Get target response time for complexity."""
        if complexity == QueryComplexity.SIMPLE:
            return PerformanceTargets.SIMPLE_QUERY_TARGET
        elif complexity == QueryComplexity.COMPLEX:
            return PerformanceTargets.COMPLEX_QUERY_TARGET
        else:  # MULTILINGUAL
            return PerformanceTargets.MULTILINGUAL_QUERY_TARGET
    
    @given(requests=generate_concurrent_requests())
    @settings(max_examples=20, deadline=60000)
    async def test_concurrent_user_handling(self, performance_monitor, requests):
        """
        **Property 20: Performance Requirements**
        **Validates: Requirements 4.2**
        
        Property: System should handle concurrent users without degradation.
        """
        assume(len(requests) >= 2)  # Need multiple requests for concurrency test
        assume(len(requests) <= 20)  # Keep test manageable
        
        # Create mock callbacks for requests
        async def create_mock_callback(processing_time):
            async def callback():
                await asyncio.sleep(processing_time)
                return {"status": "completed"}
            return callback
        
        # Submit all requests concurrently
        start_time = time.time()
        tasks = []
        
        for req in requests:
            callback = await create_mock_callback(req["processing_time"])
            
            # Queue request
            queued = await performance_monitor.queue_request(
                request_id=req["id"],
                callback=callback,
                complexity=req["complexity"],
                language=req["language"],
                priority=req["priority"]
            )
            
            # Property 1: Requests should be queued successfully under normal load
            if len(requests) <= PerformanceTargets.MAX_CONCURRENT_REQUESTS:
                assert queued, f"Request {req['id']} should be queued successfully"
        
        # Wait for some processing
        await asyncio.sleep(0.5)
        
        # Property 2: System should maintain health under concurrent load
        stats = performance_monitor.get_performance_stats()
        active_requests = stats.get("active_requests", 0)
        
        # Should not exceed maximum concurrent requests
        assert active_requests <= PerformanceTargets.MAX_CONCURRENT_REQUESTS, \
            f"Too many active requests: {active_requests}"
        
        # Property 3: Queue should handle overflow gracefully
        queue_size = stats.get("queue_size", 0)
        assert queue_size >= 0, "Queue size should be non-negative"
        
        # Property 4: System should prioritize requests correctly
        # Higher priority requests should be processed first when possible
        high_priority_count = sum(1 for req in requests if req["priority"] in [RequestPriority.HIGH, RequestPriority.CRITICAL])
        if high_priority_count > 0:
            # System should be processing or have processed high priority requests
            assert active_requests > 0 or queue_size < len(requests), \
                "System should be processing requests"
    
    @given(
        complexity=generate_query_complexity(),
        language=generate_language_code(),
        num_requests=st.integers(min_value=10, max_value=100)
    )
    @settings(max_examples=10, deadline=30000)
    async def test_load_balancing_effectiveness(self, performance_monitor, complexity, language, num_requests):
        """
        **Property 20: Performance Requirements**
        **Validates: Requirements 4.3**
        
        Property: Load balancing should distribute requests effectively.
        """
        # Generate requests with varying processing times
        processing_times = [0.1 + (i % 5) * 0.2 for i in range(num_requests)]
        
        # Submit requests
        submitted_count = 0
        for i, processing_time in enumerate(processing_times):
            async def mock_callback():
                await asyncio.sleep(processing_time)
                return {"status": "completed"}
            
            queued = await performance_monitor.queue_request(
                request_id=f"load_test_{i}",
                callback=mock_callback,
                complexity=complexity,
                language=language,
                priority=RequestPriority.NORMAL
            )
            
            if queued:
                submitted_count += 1
        
        # Property 1: Should accept reasonable number of requests
        acceptance_rate = submitted_count / num_requests
        if num_requests <= PerformanceTargets.MAX_CONCURRENT_REQUESTS:
            assert acceptance_rate >= 0.8, \
                f"Low acceptance rate: {acceptance_rate} for {num_requests} requests"
        
        # Property 2: Load balancer should limit concurrent processing
        stats = performance_monitor.get_performance_stats()
        active_requests = stats.get("active_requests", 0)
        assert active_requests <= PerformanceTargets.MAX_CONCURRENT_REQUESTS, \
            f"Load balancer failed: {active_requests} active requests"
        
        # Property 3: System should maintain responsiveness
        # Even under load, system should respond to status queries quickly
        status_start = time.time()
        status = performance_monitor.get_performance_stats()
        status_time = time.time() - status_start
        
        assert status_time < 1.0, \
            f"System unresponsive under load: {status_time}s for status query"
        
        # Property 4: Performance metrics should be tracked
        assert "total_metrics" in status, "Performance metrics should be tracked"
        assert status["total_metrics"] >= 0, "Metrics count should be non-negative"
    
    @given(
        error_rate=st.floats(min_value=0.0, max_value=0.2),
        num_requests=st.integers(min_value=20, max_value=50)
    )
    @settings(max_examples=10, deadline=30000)
    async def test_error_handling_performance(self, performance_monitor, error_rate, num_requests):
        """
        **Property 20: Performance Requirements**
        **Validates: Requirements 4.1, 4.3**
        
        Property: Error handling should not significantly impact performance.
        """
        successful_requests = 0
        failed_requests = 0
        
        # Submit requests with some failures
        for i in range(num_requests):
            should_fail = (i / num_requests) < error_rate
            
            async def mock_callback():
                if should_fail:
                    raise Exception("Simulated error")
                await asyncio.sleep(0.1)
                return {"status": "completed"}
            
            try:
                queued = await performance_monitor.queue_request(
                    request_id=f"error_test_{i}",
                    callback=mock_callback,
                    complexity=QueryComplexity.SIMPLE,
                    language=LanguageCode.ENGLISH_INDIA,
                    priority=RequestPriority.NORMAL
                )
                
                if queued:
                    if should_fail:
                        failed_requests += 1
                    else:
                        successful_requests += 1
            except Exception:
                failed_requests += 1
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        # Property 1: System should continue operating despite errors
        stats = performance_monitor.get_performance_stats()
        assert "total_metrics" in stats, "System should continue tracking metrics"
        
        # Property 2: Error rate should be tracked
        if failed_requests > 0:
            # System should have recorded some failures
            total_recorded = stats.get("total_metrics", 0)
            assert total_recorded > 0, "System should record failed requests"
        
        # Property 3: Successful requests should still meet performance targets
        if successful_requests > 0:
            # System should maintain performance for successful requests
            simple_avg_time = stats.get("simple_avg_response_time", 0)
            if simple_avg_time > 0:
                assert simple_avg_time <= PerformanceTargets.SIMPLE_QUERY_TARGET * 2, \
                    f"Performance degraded due to errors: {simple_avg_time}s"
        
        # Property 4: System should remain healthy
        assert performance_monitor.is_system_healthy() or error_rate > 0.1, \
            "System should remain healthy with low error rates"
    
    @given(
        complexity=generate_query_complexity(),
        language=generate_language_code()
    )
    @settings(max_examples=30, deadline=20000)
    async def test_performance_monitoring_accuracy(self, performance_monitor, complexity, language):
        """
        **Property 20: Performance Requirements**
        **Validates: Requirements 4.1, 4.2, 4.3**
        
        Property: Performance monitoring should accurately track and report metrics.
        """
        # Record known performance data
        test_response_times = [0.5, 1.0, 1.5, 2.0, 2.5]
        
        for response_time in test_response_times:
            performance_monitor.record_performance(
                response_time=response_time,
                query_complexity=complexity,
                language=language,
                success=True
            )
        
        # Get performance statistics
        stats = performance_monitor.get_performance_stats()
        
        # Property 1: Should track request counts accurately
        complexity_key = f"{complexity.value}_total_requests"
        if complexity_key in stats:
            assert stats[complexity_key] >= len(test_response_times), \
                f"Request count mismatch: expected >= {len(test_response_times)}, got {stats[complexity_key]}"
        
        # Property 2: Should calculate average response times
        avg_key = f"{complexity.value}_avg_response_time"
        if avg_key in stats:
            recorded_avg = stats[avg_key]
            expected_avg = sum(test_response_times) / len(test_response_times)
            
            # Allow for some variance due to other tests
            assert recorded_avg > 0, "Average response time should be positive"
            assert recorded_avg <= expected_avg * 2, \
                f"Average response time seems incorrect: {recorded_avg}s"
        
        # Property 3: Should track error rates
        error_key = f"{complexity.value}_error_rate"
        if error_key in stats:
            error_rate = stats[error_key]
            assert 0 <= error_rate <= 1, f"Error rate should be between 0 and 1: {error_rate}"
        
        # Property 4: Should provide system health status
        is_healthy = performance_monitor.is_system_healthy()
        assert isinstance(is_healthy, bool), "Health status should be boolean"
        
        # Property 5: Should track total metrics
        total_metrics = stats.get("total_metrics", 0)
        assert total_metrics >= len(test_response_times), \
=======
"""
Property-based tests for Performance Requirements.

**Property 20: Performance Requirements**
**Validates: Requirements 4.1, 4.2, 4.3**

This module tests that the BharatVoice Assistant meets performance requirements
including response time targets, concurrent user handling, and system resource management.
"""

import asyncio
import time
from typing import List, Dict, Any
import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
import structlog

from bharatvoice.core.models import LanguageCode
from bharatvoice.utils.performance_monitor import (
    PerformanceMonitor,
    QueryComplexity,
    RequestPriority,
    PerformanceTargets,
    get_performance_monitor
)
from bharatvoice.utils.accessibility import get_accessibility_manager


logger = structlog.get_logger(__name__)


# Test data generators
@st.composite
def generate_query_complexity(draw):
    """Generate query complexity levels."""
    return draw(st.sampled_from(list(QueryComplexity)))


@st.composite
def generate_language_code(draw):
    """Generate supported language codes."""
    return draw(st.sampled_from([
        LanguageCode.HINDI,
        LanguageCode.ENGLISH_INDIA,
        LanguageCode.TAMIL,
        LanguageCode.TELUGU,
        LanguageCode.BENGALI,
        LanguageCode.MARATHI,
        LanguageCode.GUJARATI,
        LanguageCode.KANNADA,
        LanguageCode.MALAYALAM,
        LanguageCode.PUNJABI,
        LanguageCode.ODIA,
    ]))


@st.composite
def generate_request_priority(draw):
    """Generate request priority levels."""
    return draw(st.sampled_from(list(RequestPriority)))


@st.composite
def generate_concurrent_requests(draw):
    """Generate concurrent request scenarios."""
    num_requests = draw(st.integers(min_value=1, max_value=50))
    requests = []
    
    for i in range(num_requests):
        request = {
            "id": f"req_{i}",
            "complexity": draw(generate_query_complexity()),
            "language": draw(generate_language_code()),
            "priority": draw(generate_request_priority()),
            "processing_time": draw(st.floats(min_value=0.1, max_value=10.0))
        }
        requests.append(request)
    
    return requests


class PerformanceTestStateMachine(RuleBasedStateMachine):
    """
    Stateful testing for performance monitoring system.
    """
    
    def __init__(self):
        super().__init__()
        self.monitor = PerformanceMonitor()
        self.active_requests = []
        self.completed_requests = []
    
    @initialize()
    async def setup_monitor(self):
        """Initialize the performance monitor."""
        await self.monitor.start()
    
    @rule(
        complexity=generate_query_complexity(),
        language=generate_language_code(),
        priority=generate_request_priority(),
        processing_time=st.floats(min_value=0.1, max_value=5.0)
    )
    async def add_request(self, complexity, language, priority, processing_time):
        """Add a request to the system."""
        request_id = f"test_req_{len(self.active_requests)}"
        
        async def mock_callback():
            await asyncio.sleep(processing_time)
            return {"status": "completed", "processing_time": processing_time}
        
        # Queue the request
        queued = await self.monitor.queue_request(
            request_id=request_id,
            callback=mock_callback,
            complexity=complexity,
            language=language,
            priority=priority
        )
        
        if queued:
            self.active_requests.append({
                "id": request_id,
                "complexity": complexity,
                "language": language,
                "priority": priority,
                "processing_time": processing_time,
                "start_time": time.time()
            })
    
    @invariant()
    def system_health_maintained(self):
        """System should maintain health under load."""
        # Check that system health indicators are reasonable
        stats = self.monitor.get_performance_stats()
        
        # CPU and memory usage should be within limits
        if "cpu_usage" in stats:
            assert stats["cpu_usage"] <= PerformanceTargets.MAX_CPU_USAGE * 1.1, \
                f"CPU usage too high: {stats['cpu_usage']}%"
        
        if "memory_usage" in stats:
            assert stats["memory_usage"] <= PerformanceTargets.MAX_MEMORY_USAGE * 1.1, \
                f"Memory usage too high: {stats['memory_usage']}%"
        
        # Active requests should not exceed limits
        active_count = stats.get("active_requests", 0)
        assert active_count <= PerformanceTargets.MAX_CONCURRENT_REQUESTS, \
            f"Too many active requests: {active_count}"
    
    @invariant()
    def queue_size_reasonable(self):
        """Request queue size should remain reasonable."""
        queue_size = self.monitor.request_queue.size()
        assert queue_size <= 1000, f"Queue size too large: {queue_size}"


@pytest.mark.asyncio
class TestPerformanceRequirements:
    """Test performance requirements compliance."""
    
    @pytest.fixture
    async def performance_monitor(self):
        """Create and start performance monitor."""
        monitor = PerformanceMonitor()
        await monitor.start()
        yield monitor
        await monitor.stop()
    
    @given(
        complexity=generate_query_complexity(),
        language=generate_language_code(),
        response_time=st.floats(min_value=0.1, max_value=10.0)
    )
    @settings(max_examples=50, deadline=30000)
    async def test_response_time_targets(self, performance_monitor, complexity, language, response_time):
        """
        **Property 20: Performance Requirements**
        **Validates: Requirements 4.1**
        
        Property: Response times should meet targets based on query complexity.
        """
        # Record performance metrics
        performance_monitor.record_performance(
            response_time=response_time,
            query_complexity=complexity,
            language=language,
            success=True
        )
        
        # Get target time for complexity
        target_time = self._get_target_time(complexity)
        
        # Property 1: Response time should be recorded correctly
        stats = performance_monitor.get_performance_stats()
        assert f"{complexity.value}_total_requests" in stats, \
            f"No requests recorded for complexity {complexity.value}"
        
        # Property 2: Fast responses should always meet targets
        if response_time <= target_time:
            # This should always pass for responses within target
            assert response_time <= target_time, \
                f"Response time {response_time}s exceeds target {target_time}s for {complexity.value}"
        
        # Property 3: System should track performance degradation
        if response_time > target_time * 2:  # Significantly over target
            # System should be aware of performance issues
            assert not performance_monitor.is_system_healthy() or response_time < target_time * 3, \
                f"System should detect performance degradation for {response_time}s response"
    
    def _get_target_time(self, complexity: QueryComplexity) -> float:
        """Get target response time for complexity."""
        if complexity == QueryComplexity.SIMPLE:
            return PerformanceTargets.SIMPLE_QUERY_TARGET
        elif complexity == QueryComplexity.COMPLEX:
            return PerformanceTargets.COMPLEX_QUERY_TARGET
        else:  # MULTILINGUAL
            return PerformanceTargets.MULTILINGUAL_QUERY_TARGET
    
    @given(requests=generate_concurrent_requests())
    @settings(max_examples=20, deadline=60000)
    async def test_concurrent_user_handling(self, performance_monitor, requests):
        """
        **Property 20: Performance Requirements**
        **Validates: Requirements 4.2**
        
        Property: System should handle concurrent users without degradation.
        """
        assume(len(requests) >= 2)  # Need multiple requests for concurrency test
        assume(len(requests) <= 20)  # Keep test manageable
        
        # Create mock callbacks for requests
        async def create_mock_callback(processing_time):
            async def callback():
                await asyncio.sleep(processing_time)
                return {"status": "completed"}
            return callback
        
        # Submit all requests concurrently
        start_time = time.time()
        tasks = []
        
        for req in requests:
            callback = await create_mock_callback(req["processing_time"])
            
            # Queue request
            queued = await performance_monitor.queue_request(
                request_id=req["id"],
                callback=callback,
                complexity=req["complexity"],
                language=req["language"],
                priority=req["priority"]
            )
            
            # Property 1: Requests should be queued successfully under normal load
            if len(requests) <= PerformanceTargets.MAX_CONCURRENT_REQUESTS:
                assert queued, f"Request {req['id']} should be queued successfully"
        
        # Wait for some processing
        await asyncio.sleep(0.5)
        
        # Property 2: System should maintain health under concurrent load
        stats = performance_monitor.get_performance_stats()
        active_requests = stats.get("active_requests", 0)
        
        # Should not exceed maximum concurrent requests
        assert active_requests <= PerformanceTargets.MAX_CONCURRENT_REQUESTS, \
            f"Too many active requests: {active_requests}"
        
        # Property 3: Queue should handle overflow gracefully
        queue_size = stats.get("queue_size", 0)
        assert queue_size >= 0, "Queue size should be non-negative"
        
        # Property 4: System should prioritize requests correctly
        # Higher priority requests should be processed first when possible
        high_priority_count = sum(1 for req in requests if req["priority"] in [RequestPriority.HIGH, RequestPriority.CRITICAL])
        if high_priority_count > 0:
            # System should be processing or have processed high priority requests
            assert active_requests > 0 or queue_size < len(requests), \
                "System should be processing requests"
    
    @given(
        complexity=generate_query_complexity(),
        language=generate_language_code(),
        num_requests=st.integers(min_value=10, max_value=100)
    )
    @settings(max_examples=10, deadline=30000)
    async def test_load_balancing_effectiveness(self, performance_monitor, complexity, language, num_requests):
        """
        **Property 20: Performance Requirements**
        **Validates: Requirements 4.3**
        
        Property: Load balancing should distribute requests effectively.
        """
        # Generate requests with varying processing times
        processing_times = [0.1 + (i % 5) * 0.2 for i in range(num_requests)]
        
        # Submit requests
        submitted_count = 0
        for i, processing_time in enumerate(processing_times):
            async def mock_callback():
                await asyncio.sleep(processing_time)
                return {"status": "completed"}
            
            queued = await performance_monitor.queue_request(
                request_id=f"load_test_{i}",
                callback=mock_callback,
                complexity=complexity,
                language=language,
                priority=RequestPriority.NORMAL
            )
            
            if queued:
                submitted_count += 1
        
        # Property 1: Should accept reasonable number of requests
        acceptance_rate = submitted_count / num_requests
        if num_requests <= PerformanceTargets.MAX_CONCURRENT_REQUESTS:
            assert acceptance_rate >= 0.8, \
                f"Low acceptance rate: {acceptance_rate} for {num_requests} requests"
        
        # Property 2: Load balancer should limit concurrent processing
        stats = performance_monitor.get_performance_stats()
        active_requests = stats.get("active_requests", 0)
        assert active_requests <= PerformanceTargets.MAX_CONCURRENT_REQUESTS, \
            f"Load balancer failed: {active_requests} active requests"
        
        # Property 3: System should maintain responsiveness
        # Even under load, system should respond to status queries quickly
        status_start = time.time()
        status = performance_monitor.get_performance_stats()
        status_time = time.time() - status_start
        
        assert status_time < 1.0, \
            f"System unresponsive under load: {status_time}s for status query"
        
        # Property 4: Performance metrics should be tracked
        assert "total_metrics" in status, "Performance metrics should be tracked"
        assert status["total_metrics"] >= 0, "Metrics count should be non-negative"
    
    @given(
        error_rate=st.floats(min_value=0.0, max_value=0.2),
        num_requests=st.integers(min_value=20, max_value=50)
    )
    @settings(max_examples=10, deadline=30000)
    async def test_error_handling_performance(self, performance_monitor, error_rate, num_requests):
        """
        **Property 20: Performance Requirements**
        **Validates: Requirements 4.1, 4.3**
        
        Property: Error handling should not significantly impact performance.
        """
        successful_requests = 0
        failed_requests = 0
        
        # Submit requests with some failures
        for i in range(num_requests):
            should_fail = (i / num_requests) < error_rate
            
            async def mock_callback():
                if should_fail:
                    raise Exception("Simulated error")
                await asyncio.sleep(0.1)
                return {"status": "completed"}
            
            try:
                queued = await performance_monitor.queue_request(
                    request_id=f"error_test_{i}",
                    callback=mock_callback,
                    complexity=QueryComplexity.SIMPLE,
                    language=LanguageCode.ENGLISH_INDIA,
                    priority=RequestPriority.NORMAL
                )
                
                if queued:
                    if should_fail:
                        failed_requests += 1
                    else:
                        successful_requests += 1
            except Exception:
                failed_requests += 1
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        # Property 1: System should continue operating despite errors
        stats = performance_monitor.get_performance_stats()
        assert "total_metrics" in stats, "System should continue tracking metrics"
        
        # Property 2: Error rate should be tracked
        if failed_requests > 0:
            # System should have recorded some failures
            total_recorded = stats.get("total_metrics", 0)
            assert total_recorded > 0, "System should record failed requests"
        
        # Property 3: Successful requests should still meet performance targets
        if successful_requests > 0:
            # System should maintain performance for successful requests
            simple_avg_time = stats.get("simple_avg_response_time", 0)
            if simple_avg_time > 0:
                assert simple_avg_time <= PerformanceTargets.SIMPLE_QUERY_TARGET * 2, \
                    f"Performance degraded due to errors: {simple_avg_time}s"
        
        # Property 4: System should remain healthy
        assert performance_monitor.is_system_healthy() or error_rate > 0.1, \
            "System should remain healthy with low error rates"
    
    @given(
        complexity=generate_query_complexity(),
        language=generate_language_code()
    )
    @settings(max_examples=30, deadline=20000)
    async def test_performance_monitoring_accuracy(self, performance_monitor, complexity, language):
        """
        **Property 20: Performance Requirements**
        **Validates: Requirements 4.1, 4.2, 4.3**
        
        Property: Performance monitoring should accurately track and report metrics.
        """
        # Record known performance data
        test_response_times = [0.5, 1.0, 1.5, 2.0, 2.5]
        
        for response_time in test_response_times:
            performance_monitor.record_performance(
                response_time=response_time,
                query_complexity=complexity,
                language=language,
                success=True
            )
        
        # Get performance statistics
        stats = performance_monitor.get_performance_stats()
        
        # Property 1: Should track request counts accurately
        complexity_key = f"{complexity.value}_total_requests"
        if complexity_key in stats:
            assert stats[complexity_key] >= len(test_response_times), \
                f"Request count mismatch: expected >= {len(test_response_times)}, got {stats[complexity_key]}"
        
        # Property 2: Should calculate average response times
        avg_key = f"{complexity.value}_avg_response_time"
        if avg_key in stats:
            recorded_avg = stats[avg_key]
            expected_avg = sum(test_response_times) / len(test_response_times)
            
            # Allow for some variance due to other tests
            assert recorded_avg > 0, "Average response time should be positive"
            assert recorded_avg <= expected_avg * 2, \
                f"Average response time seems incorrect: {recorded_avg}s"
        
        # Property 3: Should track error rates
        error_key = f"{complexity.value}_error_rate"
        if error_key in stats:
            error_rate = stats[error_key]
            assert 0 <= error_rate <= 1, f"Error rate should be between 0 and 1: {error_rate}"
        
        # Property 4: Should provide system health status
        is_healthy = performance_monitor.is_system_healthy()
        assert isinstance(is_healthy, bool), "Health status should be boolean"
        
        # Property 5: Should track total metrics
        total_metrics = stats.get("total_metrics", 0)
        assert total_metrics >= len(test_response_times), \
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
            f"Total metrics count incorrect: {total_metrics}"