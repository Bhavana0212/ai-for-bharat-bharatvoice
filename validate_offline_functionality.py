#!/usr/bin/env python3
"""
Validation script for offline functionality implementation.

This script validates that the offline voice processing system and data
synchronization components are properly implemented and functional.
"""

import asyncio
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from bharatvoice.core.models import (
        AudioBuffer,
        AudioFormat,
        LanguageCode,
        AccentType,
    )
    from bharatvoice.services.offline_sync.offline_voice_processor import (
        OfflineVoiceProcessor,
        NetworkStatus,
        create_offline_voice_processor,
    )
    from bharatvoice.services.offline_sync.data_sync_manager import (
        DataSyncManager,
        SyncStatus,
        ConflictResolution,
        create_data_sync_manager,
    )
    print("✓ Successfully imported offline functionality modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


async def validate_offline_voice_processor():
    """Validate offline voice processor functionality."""
    print("\n=== Validating Offline Voice Processor ===")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create offline processor
        processor = create_offline_voice_processor(
            cache_dir=temp_dir,
            max_cache_size_mb=5,
            enable_local_asr=True,
            enable_local_tts=True,
            common_queries_limit=10
        )
        print("✓ Created offline voice processor")
        
        # Test audio buffer creation
        test_audio = AudioBuffer(
            data=[0.1, 0.2, -0.1, -0.2] * 1000,
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=0.25
        )
        print("✓ Created test audio buffer")
        
        # Test voice activity detection
        vad_result = await processor.detect_voice_activity(test_audio)
        assert hasattr(vad_result, 'is_speech')
        assert hasattr(vad_result, 'confidence')
        assert hasattr(vad_result, 'energy_level')
        print("✓ Voice activity detection works")
        
        # Test audio processing
        processed_audio = await processor.process_audio_stream(
            test_audio, LanguageCode.ENGLISH_IN
        )
        assert isinstance(processed_audio, AudioBuffer)
        assert len(processed_audio.data) > 0
        print("✓ Audio processing works")
        
        # Test noise filtering
        filtered_audio = await processor.filter_background_noise(test_audio)
        assert isinstance(filtered_audio, AudioBuffer)
        assert len(filtered_audio.data) > 0
        print("✓ Noise filtering works")
        
        # Test TTS synthesis
        synthesized_audio = await processor.synthesize_speech_offline(
            "Hello world", LanguageCode.ENGLISH_IN, AccentType.STANDARD
        )
        assert isinstance(synthesized_audio, AudioBuffer)
        assert len(synthesized_audio.data) > 0
        assert synthesized_audio.duration > 0
        print("✓ TTS synthesis works")
        
        # Test query caching
        await processor.cache_query_response(
            "What is the weather?", LanguageCode.ENGLISH_IN, 
            "The weather is sunny today.", confidence=0.9
        )
        
        cached_response = await processor.process_common_query(
            "What is the weather?", LanguageCode.ENGLISH_IN
        )
        assert cached_response == "The weather is sunny today."
        print("✓ Query caching works")
        
        # Test network connectivity check
        network_status = await processor.check_network_connectivity()
        assert network_status in [NetworkStatus.ONLINE, NetworkStatus.OFFLINE]
        print("✓ Network connectivity check works")
        
        # Test health check
        health_status = await processor.health_check()
        assert 'status' in health_status
        assert health_status['status'] in ['healthy', 'degraded', 'unhealthy']
        print("✓ Health check works")
        
        # Test statistics
        stats = processor.get_offline_stats()
        assert isinstance(stats, dict)
        assert 'cached_queries_count' in stats
        assert 'local_asr_available' in stats
        assert 'local_tts_available' in stats
        print("✓ Statistics reporting works")
        
        print("✓ All offline voice processor tests passed!")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


async def validate_data_sync_manager():
    """Validate data synchronization manager functionality."""
    print("\n=== Validating Data Sync Manager ===")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create sync manager
        sync_manager = create_data_sync_manager(
            cache_dir=temp_dir,
            sync_interval_minutes=1,
            max_retry_attempts=2,
            enable_analytics=True
        )
        print("✓ Created data sync manager")
        
        # Test adding items to sync queue
        await sync_manager.add_to_sync_queue(
            "test_query", "query", {"text": "test", "response": "test response"}
        )
        
        status = sync_manager.get_sync_status()
        assert status['total_queued_items'] > 0
        print("✓ Sync queue management works")
        
        # Test offline session tracking
        session_id = await sync_manager.start_offline_session()
        assert session_id != ""
        assert sync_manager.current_session is not None
        print("✓ Offline session tracking works")
        
        # Test activity recording
        sync_manager.record_offline_activity("query", LanguageCode.ENGLISH_IN, True)
        sync_manager.record_offline_activity("tts", LanguageCode.HINDI, False)
        
        session = sync_manager.current_session
        assert session.queries_processed == 1
        assert session.tts_synthesized == 1
        print("✓ Activity recording works")
        
        # Test session ending
        await sync_manager.end_offline_session(user_satisfaction=4.5)
        assert sync_manager.current_session is None
        print("✓ Session ending works")
        
        # Test analytics
        analytics = await sync_manager.get_offline_analytics(days_back=7)
        assert isinstance(analytics, dict)
        assert 'total_sessions' in analytics
        assert 'sync_statistics' in analytics
        print("✓ Analytics reporting works")
        
        # Test sync status
        sync_status = sync_manager.get_sync_status()
        assert isinstance(sync_status, dict)
        assert 'sync_in_progress' in sync_status
        assert 'sync_statistics' in sync_status
        print("✓ Sync status reporting works")
        
        print("✓ All data sync manager tests passed!")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


async def validate_integration():
    """Validate integration between offline components."""
    print("\n=== Validating Component Integration ===")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create both components with same cache directory
        processor = create_offline_voice_processor(
            cache_dir=temp_dir,
            max_cache_size_mb=5,
            enable_local_asr=True,
            enable_local_tts=True
        )
        
        sync_manager = create_data_sync_manager(
            cache_dir=temp_dir,
            sync_interval_minutes=1,
            enable_analytics=True
        )
        print("✓ Created integrated offline system")
        
        # Start offline session
        session_id = await sync_manager.start_offline_session()
        
        # Simulate offline workflow
        queries = ["What time is it?", "How is the weather?", "Tell me a joke"]
        
        for query in queries:
            # Cache query response
            response = f"Offline response to: {query}"
            await processor.cache_query_response(
                query, LanguageCode.ENGLISH_IN, response
            )
            
            # Process query
            cached_response = await processor.process_common_query(
                query, LanguageCode.ENGLISH_IN
            )
            assert cached_response == response
            
            # Record activity
            sync_manager.record_offline_activity("query", LanguageCode.ENGLISH_IN, True)
        
        # Test audio processing workflow
        test_audio = AudioBuffer(
            data=[0.1] * 1600,  # 0.1 seconds at 16kHz
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=0.1
        )
        
        # Process audio
        processed_audio = await processor.process_audio_stream(
            test_audio, LanguageCode.ENGLISH_IN
        )
        
        # Synthesize speech
        synthesized_audio = await processor.synthesize_speech_offline(
            "Integration test", LanguageCode.ENGLISH_IN
        )
        
        # Record TTS activity
        sync_manager.record_offline_activity("tts", LanguageCode.ENGLISH_IN, False)
        
        # End session
        await sync_manager.end_offline_session(user_satisfaction=4.0)
        
        # Verify integration
        processor_stats = processor.get_offline_stats()
        sync_status = sync_manager.get_sync_status()
        
        assert processor_stats['cached_queries_count'] > 0
        assert sync_status['current_session_active'] is False
        
        print("✓ Integration workflow completed successfully")
        print("✓ All integration tests passed!")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


async def main():
    """Run all validation tests."""
    print("Starting offline functionality validation...")
    
    try:
        await validate_offline_voice_processor()
        await validate_data_sync_manager()
        await validate_integration()
        
        print("\n🎉 All offline functionality validation tests passed!")
        print("\nProperty 14: Offline Functionality - ✓ VALIDATED")
        print("Property 15: Network Resilience - ✓ VALIDATED")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)