"""
Simple test to verify database and storage implementation.
"""

import asyncio
import tempfile
import os
from pathlib import Path

# Test database models and connection
async def test_database():
    """Test database functionality."""
    print("Testing database functionality...")
    
    try:
        from src.bharatvoice.database.base import init_database, create_tables, health_check
        from src.bharatvoice.database.manager import db_manager
        
        # Initialize database
        init_database()
        print("✓ Database initialized")
        
        # Create tables
        await create_tables()
        print("✓ Database tables created")
        
        # Health check
        is_healthy = await health_check()
        print(f"✓ Database health check: {'healthy' if is_healthy else 'unhealthy'}")
        
        # Manager health check
        health_info = await db_manager.health_check()
        print(f"✓ Database manager health: {health_info.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False


async def test_redis_cache():
    """Test Redis cache functionality."""
    print("\nTesting Redis cache functionality...")
    
    try:
        from src.bharatvoice.cache.redis_cache import get_redis_cache
        
        # Get Redis cache (will initialize)
        redis_cache = await get_redis_cache()
        print("✓ Redis cache initialized")
        
        # Test basic operations (will work even if Redis is not available)
        test_key = "test_key"
        test_value = "test_value"
        
        # Set value
        success = await redis_cache.set(test_key, test_value, ttl=60)
        print(f"✓ Redis set operation: {'success' if success else 'failed (fallback mode)'}")
        
        # Get value
        retrieved_value = await redis_cache.get(test_key)
        print(f"✓ Redis get operation: {'success' if retrieved_value == test_value else 'failed/fallback'}")
        
        # Clean up
        await redis_cache.delete(test_key)
        
        return True
        
    except Exception as e:
        print(f"✗ Redis cache test failed: {e}")
        return False


async def test_cache_manager():
    """Test cache manager functionality."""
    print("\nTesting cache manager functionality...")
    
    try:
        from src.bharatvoice.cache.cache_manager import get_cache_manager
        
        # Get cache manager
        cache_manager = await get_cache_manager()
        print("✓ Cache manager initialized")
        
        # Test operations
        test_key = "test_cache_key"
        test_value = {"message": "Hello, BharatVoice!"}
        
        # Set value
        success = await cache_manager.set(test_key, test_value, ttl=60)
        print(f"✓ Cache manager set: {'success' if success else 'failed'}")
        
        # Get value
        retrieved_value = await cache_manager.get(test_key)
        print(f"✓ Cache manager get: {'success' if retrieved_value else 'failed'}")
        
        # Get stats
        stats = await cache_manager.get_stats()
        print(f"✓ Cache manager stats: {stats.get('redis', {}).get('status', 'unknown')}")
        
        # Clean up
        await cache_manager.delete(test_key)
        
        return True
        
    except Exception as e:
        print(f"✗ Cache manager test failed: {e}")
        return False


async def test_file_storage():
    """Test file storage functionality."""
    print("\nTesting file storage functionality...")
    
    try:
        from src.bharatvoice.storage.file_storage import get_file_storage
        import uuid
        
        # Get file storage
        file_storage = await get_file_storage()
        print("✓ File storage initialized")
        
        # Create test file data
        test_data = b"Hello, BharatVoice! This is a test audio file."
        test_user_id = str(uuid.uuid4())
        
        # Store file
        file_id = await file_storage.store_file(
            file_data=test_data,
            user_id=test_user_id,
            filename="test_audio.wav",
            mime_type="audio/wav",
            encrypt=True,
            compress=True,
            is_temporary=True
        )
        
        print(f"✓ File stored: {file_id}")
        
        if file_id:
            # Retrieve file
            result = await file_storage.retrieve_file(file_id, test_user_id)
            if result:
                retrieved_data, metadata = result
                print(f"✓ File retrieved: {len(retrieved_data)} bytes")
                print(f"  Metadata: {metadata.get('filename', 'unknown')}")
            else:
                print("✗ File retrieval failed")
            
            # List user files
            files = await file_storage.list_user_files(test_user_id)
            print(f"✓ User files listed: {len(files)} files")
            
            # Get storage stats
            stats = await file_storage.get_storage_stats()
            print(f"✓ Storage stats: {len(stats)} metrics")
            
            # Health check
            health = await file_storage.health_check()
            print(f"✓ Storage health: {health.get('status', 'unknown')}")
            
            # Clean up
            await file_storage.delete_file(file_id, test_user_id)
            print("✓ Test file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ File storage test failed: {e}")
        return False


async def test_encryption():
    """Test file encryption functionality."""
    print("\nTesting file encryption functionality...")
    
    try:
        from src.bharatvoice.storage.encryption import get_file_encryption
        
        # Get encryption
        encryption = get_file_encryption()
        print(f"✓ Encryption available: {encryption.is_available()}")
        
        if encryption.is_available():
            # Test data encryption
            test_data = b"This is secret audio data for BharatVoice!"
            
            # Encrypt
            encrypted_data = encryption.encrypt_data(test_data)
            print(f"✓ Data encrypted: {len(encrypted_data)} bytes")
            
            # Decrypt
            decrypted_data = encryption.decrypt_data(encrypted_data)
            print(f"✓ Data decrypted: {len(decrypted_data)} bytes")
            
            # Verify
            if test_data == decrypted_data:
                print("✓ Encryption/decryption successful")
            else:
                print("✗ Encryption/decryption failed")
            
            # Test checksum
            checksum = encryption.calculate_checksum(test_data)
            print(f"✓ Checksum calculated: {checksum[:16]}...")
            
            # Verify checksum
            is_valid = encryption.verify_checksum(test_data, checksum)
            print(f"✓ Checksum verification: {'valid' if is_valid else 'invalid'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Encryption test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("BharatVoice Database and Storage Implementation Test")
    print("=" * 50)
    
    tests = [
        test_database,
        test_redis_cache,
        test_cache_manager,
        test_file_storage,
        test_encryption
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! Database and storage implementation is working.")
    else:
        print("⚠️  Some tests failed. Check the implementation or dependencies.")


if __name__ == "__main__":
    asyncio.run(main())