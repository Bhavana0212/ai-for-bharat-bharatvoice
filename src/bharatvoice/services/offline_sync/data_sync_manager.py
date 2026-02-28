<<<<<<< HEAD
"""
Data synchronization system for BharatVoice Assistant.

This module provides intelligent data synchronization between offline cache
and online services when connectivity is restored, with conflict resolution
and offline usage analytics.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import sqlite3
import hashlib

from pydantic import BaseModel
from bharatvoice.core.models import LanguageCode, AudioBuffer


logger = logging.getLogger(__name__)


class SyncStatus(str, Enum):
    """Data synchronization status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"


class ConflictResolution(str, Enum):
    """Conflict resolution strategies."""
    LOCAL_WINS = "local_wins"
    REMOTE_WINS = "remote_wins"
    MERGE = "merge"
    USER_CHOICE = "user_choice"
    TIMESTAMP_BASED = "timestamp_based"


class SyncItem(BaseModel):
    """Represents an item to be synchronized."""
    
    item_id: str
    item_type: str  # "query", "model", "preference", "analytics"
    local_data: Dict[str, Any]
    remote_data: Optional[Dict[str, Any]] = None
    local_timestamp: datetime
    remote_timestamp: Optional[datetime] = None
    sync_status: SyncStatus = SyncStatus.PENDING
    conflict_resolution: Optional[ConflictResolution] = None
    retry_count: int = 0
    last_sync_attempt: Optional[datetime] = None


class OfflineUsageRecord(BaseModel):
    """Records offline usage for analytics."""
    
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    queries_processed: int = 0
    tts_synthesized: int = 0
    languages_used: List[LanguageCode] = []
    network_interruptions: int = 0
    cache_hit_rate: float = 0.0
    user_satisfaction: Optional[float] = None


class DataSyncManager:
    """
    Manages data synchronization between offline cache and online services.
    
    Features:
    - Intelligent sync scheduling based on network conditions
    - Conflict resolution with multiple strategies
    - Offline usage analytics and reporting
    - Graceful handling of intermittent connectivity
    - Data integrity validation
    """
    
    def __init__(
        self,
        cache_dir: str = ".bharatvoice_offline",
        sync_interval_minutes: int = 15,
        max_retry_attempts: int = 3,
        conflict_resolution_strategy: ConflictResolution = ConflictResolution.TIMESTAMP_BASED,
        enable_analytics: bool = True
    ):
        """
        Initialize data synchronization manager.
        
        Args:
            cache_dir: Directory for offline cache storage
            sync_interval_minutes: Interval between sync attempts
            max_retry_attempts: Maximum retry attempts for failed syncs
            conflict_resolution_strategy: Default conflict resolution strategy
            enable_analytics: Whether to enable offline usage analytics
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.sync_interval_minutes = sync_interval_minutes
        self.max_retry_attempts = max_retry_attempts
        self.conflict_resolution_strategy = conflict_resolution_strategy
        self.enable_analytics = enable_analytics
        
        # Database paths
        self.sync_db_path = self.cache_dir / "sync_queue.db"
        self.analytics_db_path = self.cache_dir / "offline_analytics.db"
        
        # Sync state
        self.sync_queue: List[SyncItem] = []
        self.sync_in_progress = False
        self.last_sync_time: Optional[datetime] = None
        self.current_session: Optional[OfflineUsageRecord] = None
        
        # Statistics
        self.sync_stats = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'conflicts_resolved': 0,
            'data_uploaded_mb': 0.0,
            'data_downloaded_mb': 0.0
        }
        
        # Initialize databases
        self._init_sync_databases()
        
        # Load pending sync items
        self._load_sync_queue()
        
        # Start background sync task
        self._sync_task = None
        
        logger.info("DataSyncManager initialized successfully")
    
    def _init_sync_databases(self):
        """Initialize SQLite databases for sync management."""
        try:
            # Initialize sync queue database
            with sqlite3.connect(self.sync_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sync_queue (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        item_id TEXT NOT NULL,
                        item_type TEXT NOT NULL,
                        local_data TEXT NOT NULL,
                        remote_data TEXT,
                        local_timestamp TEXT NOT NULL,
                        remote_timestamp TEXT,
                        sync_status TEXT NOT NULL,
                        conflict_resolution TEXT,
                        retry_count INTEGER DEFAULT 0,
                        last_sync_attempt TEXT,
                        created_at TEXT NOT NULL,
                        UNIQUE(item_id, item_type)
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sync_status 
                    ON sync_queue(sync_status)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_item_type 
                    ON sync_queue(item_type)
                """)
            
            # Initialize analytics database
            if self.enable_analytics:
                with sqlite3.connect(self.analytics_db_path) as conn:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS offline_sessions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT UNIQUE NOT NULL,
                            start_time TEXT NOT NULL,
                            end_time TEXT,
                            queries_processed INTEGER DEFAULT 0,
                            tts_synthesized INTEGER DEFAULT 0,
                            languages_used TEXT,
                            network_interruptions INTEGER DEFAULT 0,
                            cache_hit_rate REAL DEFAULT 0.0,
                            user_satisfaction REAL,
                            created_at TEXT NOT NULL
                        )
                    """)
                    
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS sync_events (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            event_type TEXT NOT NULL,
                            timestamp TEXT NOT NULL,
                            details TEXT,
                            success BOOLEAN NOT NULL,
                            duration_seconds REAL
                        )
                    """)
            
            logger.info("Sync databases initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing sync databases: {e}")
    
    def _load_sync_queue(self):
        """Load pending sync items from database."""
        try:
            with sqlite3.connect(self.sync_db_path) as conn:
                cursor = conn.execute("""
                    SELECT item_id, item_type, local_data, remote_data,
                           local_timestamp, remote_timestamp, sync_status,
                           conflict_resolution, retry_count, last_sync_attempt
                    FROM sync_queue
                    WHERE sync_status IN ('pending', 'failed', 'conflict')
                    ORDER BY local_timestamp ASC
                """)
                
                for row in cursor.fetchall():
                    sync_item = SyncItem(
                        item_id=row[0],
                        item_type=row[1],
                        local_data=json.loads(row[2]),
                        remote_data=json.loads(row[3]) if row[3] else None,
                        local_timestamp=datetime.fromisoformat(row[4]),
                        remote_timestamp=datetime.fromisoformat(row[5]) if row[5] else None,
                        sync_status=SyncStatus(row[6]),
                        conflict_resolution=ConflictResolution(row[7]) if row[7] else None,
                        retry_count=row[8],
                        last_sync_attempt=datetime.fromisoformat(row[9]) if row[9] else None
                    )
                    
                    self.sync_queue.append(sync_item)
            
            logger.info(f"Loaded {len(self.sync_queue)} pending sync items")
            
        except Exception as e:
            logger.error(f"Error loading sync queue: {e}")
    
    async def start_sync_service(self):
        """Start the background synchronization service."""
        if self._sync_task is None or self._sync_task.done():
            self._sync_task = asyncio.create_task(self._sync_loop())
            logger.info("Data sync service started")
    
    async def stop_sync_service(self):
        """Stop the background synchronization service."""
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            logger.info("Data sync service stopped")
    
    async def _sync_loop(self):
        """Background sync loop."""
        while True:
            try:
                await asyncio.sleep(self.sync_interval_minutes * 60)
                
                if not self.sync_in_progress and self.sync_queue:
                    await self.perform_sync()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def add_to_sync_queue(
        self,
        item_id: str,
        item_type: str,
        local_data: Dict[str, Any],
        priority: bool = False
    ):
        """
        Add item to synchronization queue.
        
        Args:
            item_id: Unique identifier for the item
            item_type: Type of item (query, model, preference, analytics)
            local_data: Local data to sync
            priority: Whether to prioritize this item
        """
        try:
            sync_item = SyncItem(
                item_id=item_id,
                item_type=item_type,
                local_data=local_data,
                local_timestamp=datetime.now(),
                sync_status=SyncStatus.PENDING
            )
            
            # Add to queue
            if priority:
                self.sync_queue.insert(0, sync_item)
            else:
                self.sync_queue.append(sync_item)
            
            # Store in database
            with sqlite3.connect(self.sync_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO sync_queue 
                    (item_id, item_type, local_data, local_timestamp, 
                     sync_status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    item_id, item_type, json.dumps(local_data),
                    sync_item.local_timestamp.isoformat(),
                    sync_item.sync_status.value,
                    datetime.now().isoformat()
                ))
            
            logger.debug(f"Added {item_type} item to sync queue: {item_id}")
            
        except Exception as e:
            logger.error(f"Error adding item to sync queue: {e}")
    
    async def perform_sync(self) -> Dict[str, Any]:
        """
        Perform synchronization of queued items.
        
        Returns:
            Sync results summary
        """
        if self.sync_in_progress:
            return {"status": "already_in_progress"}
        
        self.sync_in_progress = True
        sync_start_time = datetime.now()
        
        try:
            # Check network connectivity
            if not await self._check_network_connectivity():
                logger.info("No network connectivity - skipping sync")
                return {"status": "no_network", "queued_items": len(self.sync_queue)}
            
            sync_results = {
                "status": "completed",
                "total_items": len(self.sync_queue),
                "successful": 0,
                "failed": 0,
                "conflicts": 0,
                "skipped": 0
            }
            
            # Process sync queue
            items_to_remove = []
            
            for sync_item in self.sync_queue:
                try:
                    # Skip items that have exceeded retry limit
                    if sync_item.retry_count >= self.max_retry_attempts:
                        sync_results["skipped"] += 1
                        continue
                    
                    # Update sync status
                    sync_item.sync_status = SyncStatus.IN_PROGRESS
                    sync_item.last_sync_attempt = datetime.now()
                    
                    # Perform sync based on item type
                    result = await self._sync_item(sync_item)
                    
                    if result["success"]:
                        sync_item.sync_status = SyncStatus.COMPLETED
                        sync_results["successful"] += 1
                        items_to_remove.append(sync_item)
                        
                    elif result.get("conflict"):
                        sync_item.sync_status = SyncStatus.CONFLICT
                        sync_item.remote_data = result.get("remote_data")
                        sync_item.remote_timestamp = result.get("remote_timestamp")
                        sync_results["conflicts"] += 1
                        
                    else:
                        sync_item.sync_status = SyncStatus.FAILED
                        sync_item.retry_count += 1
                        sync_results["failed"] += 1
                    
                    # Update database
                    await self._update_sync_item_in_db(sync_item)
                    
                except Exception as e:
                    logger.error(f"Error syncing item {sync_item.item_id}: {e}")
                    sync_item.sync_status = SyncStatus.FAILED
                    sync_item.retry_count += 1
                    sync_results["failed"] += 1
                    await self._update_sync_item_in_db(sync_item)
            
            # Remove completed items from queue
            for item in items_to_remove:
                self.sync_queue.remove(item)
                await self._remove_sync_item_from_db(item.item_id, item.item_type)
            
            # Update statistics
            self.sync_stats['total_syncs'] += 1
            self.sync_stats['successful_syncs'] += sync_results["successful"]
            self.sync_stats['failed_syncs'] += sync_results["failed"]
            self.sync_stats['conflicts_resolved'] += sync_results["conflicts"]
            
            self.last_sync_time = datetime.now()
            
            # Log sync event
            if self.enable_analytics:
                await self._log_sync_event(
                    "sync_completed",
                    sync_results,
                    True,
                    (datetime.now() - sync_start_time).total_seconds()
                )
            
            logger.info(f"Sync completed: {sync_results}")
            return sync_results
            
        except Exception as e:
            logger.error(f"Error during sync: {e}")
            
            if self.enable_analytics:
                await self._log_sync_event(
                    "sync_failed",
                    {"error": str(e)},
                    False,
                    (datetime.now() - sync_start_time).total_seconds()
                )
            
            return {"status": "error", "error": str(e)}
            
        finally:
            self.sync_in_progress = False
    
    async def _sync_item(self, sync_item: SyncItem) -> Dict[str, Any]:
        """
        Sync individual item with remote service.
        
        Args:
            sync_item: Item to synchronize
            
        Returns:
            Sync result
        """
        try:
            if sync_item.item_type == "query":
                return await self._sync_query(sync_item)
            elif sync_item.item_type == "model":
                return await self._sync_model(sync_item)
            elif sync_item.item_type == "preference":
                return await self._sync_preference(sync_item)
            elif sync_item.item_type == "analytics":
                return await self._sync_analytics(sync_item)
            else:
                logger.warning(f"Unknown sync item type: {sync_item.item_type}")
                return {"success": False, "error": "unknown_item_type"}
                
        except Exception as e:
            logger.error(f"Error syncing {sync_item.item_type} item: {e}")
            return {"success": False, "error": str(e)}
    
    async def _sync_query(self, sync_item: SyncItem) -> Dict[str, Any]:
        """Sync cached query with remote service."""
        try:
            # Simulate remote API call
            # In production, this would call the actual remote service
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Check if remote has newer version
            remote_data = await self._fetch_remote_query(sync_item.item_id)
            
            if remote_data:
                remote_timestamp = datetime.fromisoformat(remote_data.get("timestamp"))
                
                # Check for conflicts
                if remote_timestamp > sync_item.local_timestamp:
                    # Remote is newer - potential conflict
                    return {
                        "success": False,
                        "conflict": True,
                        "remote_data": remote_data,
                        "remote_timestamp": remote_timestamp
                    }
            
            # Upload local data to remote
            upload_success = await self._upload_query_to_remote(sync_item)
            
            return {"success": upload_success}
            
        except Exception as e:
            logger.error(f"Error syncing query: {e}")
            return {"success": False, "error": str(e)}
    
    async def _sync_model(self, sync_item: SyncItem) -> Dict[str, Any]:
        """Sync cached model with remote service."""
        try:
            # For models, we typically only upload usage statistics
            # The actual model data is usually too large for frequent sync
            
            model_stats = {
                "model_id": sync_item.item_id,
                "usage_count": sync_item.local_data.get("usage_count", 0),
                "last_used": sync_item.local_data.get("last_used"),
                "performance_metrics": sync_item.local_data.get("performance_metrics", {})
            }
            
            # Upload model statistics
            upload_success = await self._upload_model_stats_to_remote(model_stats)
            
            return {"success": upload_success}
            
        except Exception as e:
            logger.error(f"Error syncing model: {e}")
            return {"success": False, "error": str(e)}
    
    async def _sync_preference(self, sync_item: SyncItem) -> Dict[str, Any]:
        """Sync user preferences with remote service."""
        try:
            # Fetch remote preferences
            remote_prefs = await self._fetch_remote_preferences(sync_item.item_id)
            
            if remote_prefs:
                # Merge preferences intelligently
                merged_prefs = self._merge_preferences(
                    sync_item.local_data,
                    remote_prefs
                )
                
                # Upload merged preferences
                upload_success = await self._upload_preferences_to_remote(
                    sync_item.item_id, merged_prefs
                )
                
                return {"success": upload_success}
            else:
                # No remote preferences, upload local ones
                upload_success = await self._upload_preferences_to_remote(
                    sync_item.item_id, sync_item.local_data
                )
                
                return {"success": upload_success}
            
        except Exception as e:
            logger.error(f"Error syncing preferences: {e}")
            return {"success": False, "error": str(e)}
    
    async def _sync_analytics(self, sync_item: SyncItem) -> Dict[str, Any]:
        """Sync offline usage analytics with remote service."""
        try:
            # Upload analytics data to remote service
            upload_success = await self._upload_analytics_to_remote(sync_item.local_data)
            
            return {"success": upload_success}
            
        except Exception as e:
            logger.error(f"Error syncing analytics: {e}")
            return {"success": False, "error": str(e)}
    
    async def resolve_conflict(
        self,
        sync_item: SyncItem,
        resolution_strategy: Optional[ConflictResolution] = None
    ) -> Dict[str, Any]:
        """
        Resolve synchronization conflict.
        
        Args:
            sync_item: Conflicted sync item
            resolution_strategy: Strategy to use for resolution
            
        Returns:
            Resolution result
        """
        try:
            strategy = resolution_strategy or self.conflict_resolution_strategy
            
            if strategy == ConflictResolution.LOCAL_WINS:
                # Keep local data, overwrite remote
                result = await self._upload_data_to_remote(sync_item, force=True)
                
            elif strategy == ConflictResolution.REMOTE_WINS:
                # Keep remote data, overwrite local
                result = await self._download_data_from_remote(sync_item, force=True)
                
            elif strategy == ConflictResolution.TIMESTAMP_BASED:
                # Use newer timestamp
                if (sync_item.remote_timestamp and 
                    sync_item.remote_timestamp > sync_item.local_timestamp):
                    result = await self._download_data_from_remote(sync_item)
                else:
                    result = await self._upload_data_to_remote(sync_item)
                    
            elif strategy == ConflictResolution.MERGE:
                # Attempt to merge data
                merged_data = self._merge_data(
                    sync_item.local_data,
                    sync_item.remote_data
                )
                sync_item.local_data = merged_data
                result = await self._upload_data_to_remote(sync_item)
                
            else:
                # USER_CHOICE - requires manual intervention
                return {
                    "success": False,
                    "requires_user_input": True,
                    "local_data": sync_item.local_data,
                    "remote_data": sync_item.remote_data
                }
            
            if result.get("success"):
                sync_item.sync_status = SyncStatus.COMPLETED
                sync_item.conflict_resolution = strategy
                await self._update_sync_item_in_db(sync_item)
                
                self.sync_stats['conflicts_resolved'] += 1
                
                logger.info(f"Conflict resolved using {strategy.value} strategy")
            
            return result
            
        except Exception as e:
            logger.error(f"Error resolving conflict: {e}")
            return {"success": False, "error": str(e)}
    
    def _merge_data(self, local_data: Dict[str, Any], remote_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge local and remote data intelligently.
        
        Args:
            local_data: Local data
            remote_data: Remote data
            
        Returns:
            Merged data
        """
        try:
            merged = local_data.copy()
            
            for key, remote_value in remote_data.items():
                if key not in merged:
                    # New key from remote
                    merged[key] = remote_value
                elif isinstance(remote_value, dict) and isinstance(merged[key], dict):
                    # Recursively merge dictionaries
                    merged[key] = self._merge_data(merged[key], remote_value)
                elif isinstance(remote_value, list) and isinstance(merged[key], list):
                    # Merge lists (remove duplicates)
                    merged[key] = list(set(merged[key] + remote_value))
                elif key.endswith('_count') or key.endswith('_total'):
                    # Sum numeric counters
                    merged[key] = merged[key] + remote_value
                # For other types, keep local value (local wins for conflicts)
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging data: {e}")
            return local_data
    
    def _merge_preferences(self, local_prefs: Dict[str, Any], remote_prefs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user preferences with conflict resolution."""
        try:
            merged = remote_prefs.copy()  # Start with remote as base
            
            # Override with local preferences for user-specific settings
            user_specific_keys = [
                'language_preference', 'accent_preference', 'voice_speed',
                'volume_level', 'notification_settings', 'privacy_settings'
            ]
            
            for key in user_specific_keys:
                if key in local_prefs:
                    merged[key] = local_prefs[key]
            
            # Merge usage statistics
            if 'usage_stats' in local_prefs and 'usage_stats' in remote_prefs:
                merged['usage_stats'] = self._merge_data(
                    local_prefs['usage_stats'],
                    remote_prefs['usage_stats']
                )
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging preferences: {e}")
            return local_prefs
    
    async def start_offline_session(self, session_id: Optional[str] = None) -> str:
        """
        Start tracking an offline usage session.
        
        Args:
            session_id: Optional session ID, generated if not provided
            
        Returns:
            Session ID
        """
        try:
            if not self.enable_analytics:
                return ""
            
            session_id = session_id or self._generate_session_id()
            
            self.current_session = OfflineUsageRecord(
                session_id=session_id,
                start_time=datetime.now()
            )
            
            # Store in database
            with sqlite3.connect(self.analytics_db_path) as conn:
                conn.execute("""
                    INSERT INTO offline_sessions 
                    (session_id, start_time, created_at)
                    VALUES (?, ?, ?)
                """, (
                    session_id,
                    self.current_session.start_time.isoformat(),
                    datetime.now().isoformat()
                ))
            
            logger.info(f"Started offline session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error starting offline session: {e}")
            return ""
    
    async def end_offline_session(self, user_satisfaction: Optional[float] = None):
        """
        End the current offline usage session.
        
        Args:
            user_satisfaction: Optional user satisfaction rating (0.0-5.0)
        """
        try:
            if not self.enable_analytics or not self.current_session:
                return
            
            self.current_session.end_time = datetime.now()
            self.current_session.user_satisfaction = user_satisfaction
            
            # Update database
            with sqlite3.connect(self.analytics_db_path) as conn:
                conn.execute("""
                    UPDATE offline_sessions 
                    SET end_time = ?, queries_processed = ?, tts_synthesized = ?,
                        languages_used = ?, network_interruptions = ?,
                        cache_hit_rate = ?, user_satisfaction = ?
                    WHERE session_id = ?
                """, (
                    self.current_session.end_time.isoformat(),
                    self.current_session.queries_processed,
                    self.current_session.tts_synthesized,
                    json.dumps([lang.value for lang in self.current_session.languages_used]),
                    self.current_session.network_interruptions,
                    self.current_session.cache_hit_rate,
                    self.current_session.user_satisfaction,
                    self.current_session.session_id
                ))
            
            # Add to sync queue for upload
            await self.add_to_sync_queue(
                f"session_{self.current_session.session_id}",
                "analytics",
                self.current_session.dict()
            )
            
            logger.info(f"Ended offline session: {self.current_session.session_id}")
            self.current_session = None
            
        except Exception as e:
            logger.error(f"Error ending offline session: {e}")
    
    def record_offline_activity(
        self,
        activity_type: str,
        language: Optional[LanguageCode] = None,
        cache_hit: bool = False
    ):
        """
        Record offline activity for current session.
        
        Args:
            activity_type: Type of activity ("query", "tts", "network_interruption")
            language: Language used (if applicable)
            cache_hit: Whether this was a cache hit
        """
        try:
            if not self.enable_analytics or not self.current_session:
                return
            
            if activity_type == "query":
                self.current_session.queries_processed += 1
            elif activity_type == "tts":
                self.current_session.tts_synthesized += 1
            elif activity_type == "network_interruption":
                self.current_session.network_interruptions += 1
            
            if language and language not in self.current_session.languages_used:
                self.current_session.languages_used.append(language)
            
            # Update cache hit rate
            if activity_type in ["query", "tts"]:
                total_activities = (
                    self.current_session.queries_processed + 
                    self.current_session.tts_synthesized
                )
                if total_activities > 0:
                    # This is a simplified calculation - in production, track hits/misses separately
                    if cache_hit:
                        self.current_session.cache_hit_rate = min(
                            1.0, self.current_session.cache_hit_rate + 0.1
                        )
                    else:
                        self.current_session.cache_hit_rate = max(
                            0.0, self.current_session.cache_hit_rate - 0.05
                        )
            
        except Exception as e:
            logger.error(f"Error recording offline activity: {e}")
    
    async def get_offline_analytics(
        self,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get offline usage analytics.
        
        Args:
            days_back: Number of days to include in analytics
            
        Returns:
            Analytics data
        """
        try:
            if not self.enable_analytics:
                return {"analytics_disabled": True}
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            analytics = {
                "period_days": days_back,
                "total_sessions": 0,
                "total_offline_time_hours": 0.0,
                "total_queries": 0,
                "total_tts_synthesized": 0,
                "average_session_duration_minutes": 0.0,
                "languages_used": {},
                "average_cache_hit_rate": 0.0,
                "network_interruptions": 0,
                "user_satisfaction_average": 0.0,
                "sync_statistics": self.sync_stats.copy()
            }
            
            with sqlite3.connect(self.analytics_db_path) as conn:
                # Get session statistics
                cursor = conn.execute("""
                    SELECT COUNT(*), 
                           AVG(CASE WHEN end_time IS NOT NULL 
                               THEN (julianday(end_time) - julianday(start_time)) * 24 * 60 
                               ELSE 0 END),
                           SUM(queries_processed),
                           SUM(tts_synthesized),
                           AVG(cache_hit_rate),
                           SUM(network_interruptions),
                           AVG(user_satisfaction)
                    FROM offline_sessions 
                    WHERE start_time >= ?
                """, (cutoff_date.isoformat(),))
                
                row = cursor.fetchone()
                if row:
                    analytics["total_sessions"] = row[0] or 0
                    analytics["average_session_duration_minutes"] = row[1] or 0.0
                    analytics["total_queries"] = row[2] or 0
                    analytics["total_tts_synthesized"] = row[3] or 0
                    analytics["average_cache_hit_rate"] = row[4] or 0.0
                    analytics["network_interruptions"] = row[5] or 0
                    analytics["user_satisfaction_average"] = row[6] or 0.0
                
                # Calculate total offline time
                cursor = conn.execute("""
                    SELECT SUM(CASE WHEN end_time IS NOT NULL 
                               THEN (julianday(end_time) - julianday(start_time)) * 24 
                               ELSE 0 END)
                    FROM offline_sessions 
                    WHERE start_time >= ?
                """, (cutoff_date.isoformat(),))
                
                result = cursor.fetchone()
                if result and result[0]:
                    analytics["total_offline_time_hours"] = result[0]
                
                # Get language usage statistics
                cursor = conn.execute("""
                    SELECT languages_used FROM offline_sessions 
                    WHERE start_time >= ? AND languages_used IS NOT NULL
                """, (cutoff_date.isoformat(),))
                
                language_counts = {}
                for row in cursor.fetchall():
                    try:
                        languages = json.loads(row[0])
                        for lang in languages:
                            language_counts[lang] = language_counts.get(lang, 0) + 1
                    except (json.JSONDecodeError, TypeError):
                        continue
                
                analytics["languages_used"] = language_counts
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting offline analytics: {e}")
            return {"error": str(e)}
    
    def get_sync_status(self) -> Dict[str, Any]:
        """
        Get current synchronization status.
        
        Returns:
            Sync status information
        """
        try:
            pending_by_type = {}
            failed_by_type = {}
            conflicts_by_type = {}
            
            for item in self.sync_queue:
                if item.sync_status == SyncStatus.PENDING:
                    pending_by_type[item.item_type] = pending_by_type.get(item.item_type, 0) + 1
                elif item.sync_status == SyncStatus.FAILED:
                    failed_by_type[item.item_type] = failed_by_type.get(item.item_type, 0) + 1
                elif item.sync_status == SyncStatus.CONFLICT:
                    conflicts_by_type[item.item_type] = conflicts_by_type.get(item.item_type, 0) + 1
            
            return {
                "sync_in_progress": self.sync_in_progress,
                "last_sync_time": self.last_sync_time.isoformat() if self.last_sync_time else None,
                "total_queued_items": len(self.sync_queue),
                "pending_by_type": pending_by_type,
                "failed_by_type": failed_by_type,
                "conflicts_by_type": conflicts_by_type,
                "sync_statistics": self.sync_stats.copy(),
                "current_session_active": self.current_session is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting sync status: {e}")
            return {"error": str(e)}
    
    # Helper methods for remote API calls (these would be implemented with actual API calls)
    
    async def _check_network_connectivity(self) -> bool:
        """Check if network connectivity is available."""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except (socket.error, OSError):
            return False
    
    async def _fetch_remote_query(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Fetch query data from remote service."""
        # Simulate API call
        await asyncio.sleep(0.1)
        return None  # No remote data found
    
    async def _upload_query_to_remote(self, sync_item: SyncItem) -> bool:
        """Upload query to remote service."""
        # Simulate API call
        await asyncio.sleep(0.1)
        return True  # Success
    
    async def _upload_model_stats_to_remote(self, model_stats: Dict[str, Any]) -> bool:
        """Upload model statistics to remote service."""
        # Simulate API call
        await asyncio.sleep(0.1)
        return True  # Success
    
    async def _fetch_remote_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetch user preferences from remote service."""
        # Simulate API call
        await asyncio.sleep(0.1)
        return None  # No remote preferences found
    
    async def _upload_preferences_to_remote(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Upload user preferences to remote service."""
        # Simulate API call
        await asyncio.sleep(0.1)
        return True  # Success
    
    async def _upload_analytics_to_remote(self, analytics_data: Dict[str, Any]) -> bool:
        """Upload analytics data to remote service."""
        # Simulate API call
        await asyncio.sleep(0.1)
        return True  # Success
    
    async def _upload_data_to_remote(self, sync_item: SyncItem, force: bool = False) -> Dict[str, Any]:
        """Upload data to remote service."""
        # Simulate API call
        await asyncio.sleep(0.1)
        return {"success": True}
    
    async def _download_data_from_remote(self, sync_item: SyncItem, force: bool = False) -> Dict[str, Any]:
        """Download data from remote service."""
        # Simulate API call
        await asyncio.sleep(0.1)
        return {"success": True}
    
    async def _update_sync_item_in_db(self, sync_item: SyncItem):
        """Update sync item in database."""
        try:
            with sqlite3.connect(self.sync_db_path) as conn:
                conn.execute("""
                    UPDATE sync_queue 
                    SET remote_data = ?, remote_timestamp = ?, sync_status = ?,
                        conflict_resolution = ?, retry_count = ?, last_sync_attempt = ?
                    WHERE item_id = ? AND item_type = ?
                """, (
                    json.dumps(sync_item.remote_data) if sync_item.remote_data else None,
                    sync_item.remote_timestamp.isoformat() if sync_item.remote_timestamp else None,
                    sync_item.sync_status.value,
                    sync_item.conflict_resolution.value if sync_item.conflict_resolution else None,
                    sync_item.retry_count,
                    sync_item.last_sync_attempt.isoformat() if sync_item.last_sync_attempt else None,
                    sync_item.item_id,
                    sync_item.item_type
                ))
        except Exception as e:
            logger.error(f"Error updating sync item in database: {e}")
    
    async def _remove_sync_item_from_db(self, item_id: str, item_type: str):
        """Remove sync item from database."""
        try:
            with sqlite3.connect(self.sync_db_path) as conn:
                conn.execute("""
                    DELETE FROM sync_queue 
                    WHERE item_id = ? AND item_type = ?
                """, (item_id, item_type))
        except Exception as e:
            logger.error(f"Error removing sync item from database: {e}")
    
    async def _log_sync_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        success: bool,
        duration_seconds: float
    ):
        """Log sync event for analytics."""
        try:
            with sqlite3.connect(self.analytics_db_path) as conn:
                conn.execute("""
                    INSERT INTO sync_events 
                    (event_type, timestamp, details, success, duration_seconds)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    event_type,
                    datetime.now().isoformat(),
                    json.dumps(details),
                    success,
                    duration_seconds
                ))
        except Exception as e:
            logger.error(f"Error logging sync event: {e}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        import uuid
        return f"offline_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"


# Factory function for creating data sync manager
def create_data_sync_manager(
    cache_dir: str = ".bharatvoice_offline",
    sync_interval_minutes: int = 15,
    max_retry_attempts: int = 3,
    conflict_resolution_strategy: ConflictResolution = ConflictResolution.TIMESTAMP_BASED,
    enable_analytics: bool = True
) -> DataSyncManager:
    """
    Factory function to create a data sync manager instance.
    
    Args:
        cache_dir: Directory for offline cache storage
        sync_interval_minutes: Interval between sync attempts
        max_retry_attempts: Maximum retry attempts for failed syncs
        conflict_resolution_strategy: Default conflict resolution strategy
        enable_analytics: Whether to enable offline usage analytics
        
    Returns:
        Configured DataSyncManager instance
    """
    return DataSyncManager(
        cache_dir=cache_dir,
        sync_interval_minutes=sync_interval_minutes,
        max_retry_attempts=max_retry_attempts,
        conflict_resolution_strategy=conflict_resolution_strategy,
        enable_analytics=enable_analytics
=======
"""
Data synchronization system for BharatVoice Assistant.

This module provides intelligent data synchronization between offline cache
and online services when connectivity is restored, with conflict resolution
and offline usage analytics.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import sqlite3
import hashlib

from pydantic import BaseModel
from bharatvoice.core.models import LanguageCode, AudioBuffer


logger = logging.getLogger(__name__)


class SyncStatus(str, Enum):
    """Data synchronization status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"


class ConflictResolution(str, Enum):
    """Conflict resolution strategies."""
    LOCAL_WINS = "local_wins"
    REMOTE_WINS = "remote_wins"
    MERGE = "merge"
    USER_CHOICE = "user_choice"
    TIMESTAMP_BASED = "timestamp_based"


class SyncItem(BaseModel):
    """Represents an item to be synchronized."""
    
    item_id: str
    item_type: str  # "query", "model", "preference", "analytics"
    local_data: Dict[str, Any]
    remote_data: Optional[Dict[str, Any]] = None
    local_timestamp: datetime
    remote_timestamp: Optional[datetime] = None
    sync_status: SyncStatus = SyncStatus.PENDING
    conflict_resolution: Optional[ConflictResolution] = None
    retry_count: int = 0
    last_sync_attempt: Optional[datetime] = None


class OfflineUsageRecord(BaseModel):
    """Records offline usage for analytics."""
    
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    queries_processed: int = 0
    tts_synthesized: int = 0
    languages_used: List[LanguageCode] = []
    network_interruptions: int = 0
    cache_hit_rate: float = 0.0
    user_satisfaction: Optional[float] = None


class DataSyncManager:
    """
    Manages data synchronization between offline cache and online services.
    
    Features:
    - Intelligent sync scheduling based on network conditions
    - Conflict resolution with multiple strategies
    - Offline usage analytics and reporting
    - Graceful handling of intermittent connectivity
    - Data integrity validation
    """
    
    def __init__(
        self,
        cache_dir: str = ".bharatvoice_offline",
        sync_interval_minutes: int = 15,
        max_retry_attempts: int = 3,
        conflict_resolution_strategy: ConflictResolution = ConflictResolution.TIMESTAMP_BASED,
        enable_analytics: bool = True
    ):
        """
        Initialize data synchronization manager.
        
        Args:
            cache_dir: Directory for offline cache storage
            sync_interval_minutes: Interval between sync attempts
            max_retry_attempts: Maximum retry attempts for failed syncs
            conflict_resolution_strategy: Default conflict resolution strategy
            enable_analytics: Whether to enable offline usage analytics
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.sync_interval_minutes = sync_interval_minutes
        self.max_retry_attempts = max_retry_attempts
        self.conflict_resolution_strategy = conflict_resolution_strategy
        self.enable_analytics = enable_analytics
        
        # Database paths
        self.sync_db_path = self.cache_dir / "sync_queue.db"
        self.analytics_db_path = self.cache_dir / "offline_analytics.db"
        
        # Sync state
        self.sync_queue: List[SyncItem] = []
        self.sync_in_progress = False
        self.last_sync_time: Optional[datetime] = None
        self.current_session: Optional[OfflineUsageRecord] = None
        
        # Statistics
        self.sync_stats = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'conflicts_resolved': 0,
            'data_uploaded_mb': 0.0,
            'data_downloaded_mb': 0.0
        }
        
        # Initialize databases
        self._init_sync_databases()
        
        # Load pending sync items
        self._load_sync_queue()
        
        # Start background sync task
        self._sync_task = None
        
        logger.info("DataSyncManager initialized successfully")
    
    def _init_sync_databases(self):
        """Initialize SQLite databases for sync management."""
        try:
            # Initialize sync queue database
            with sqlite3.connect(self.sync_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sync_queue (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        item_id TEXT NOT NULL,
                        item_type TEXT NOT NULL,
                        local_data TEXT NOT NULL,
                        remote_data TEXT,
                        local_timestamp TEXT NOT NULL,
                        remote_timestamp TEXT,
                        sync_status TEXT NOT NULL,
                        conflict_resolution TEXT,
                        retry_count INTEGER DEFAULT 0,
                        last_sync_attempt TEXT,
                        created_at TEXT NOT NULL,
                        UNIQUE(item_id, item_type)
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sync_status 
                    ON sync_queue(sync_status)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_item_type 
                    ON sync_queue(item_type)
                """)
            
            # Initialize analytics database
            if self.enable_analytics:
                with sqlite3.connect(self.analytics_db_path) as conn:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS offline_sessions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT UNIQUE NOT NULL,
                            start_time TEXT NOT NULL,
                            end_time TEXT,
                            queries_processed INTEGER DEFAULT 0,
                            tts_synthesized INTEGER DEFAULT 0,
                            languages_used TEXT,
                            network_interruptions INTEGER DEFAULT 0,
                            cache_hit_rate REAL DEFAULT 0.0,
                            user_satisfaction REAL,
                            created_at TEXT NOT NULL
                        )
                    """)
                    
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS sync_events (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            event_type TEXT NOT NULL,
                            timestamp TEXT NOT NULL,
                            details TEXT,
                            success BOOLEAN NOT NULL,
                            duration_seconds REAL
                        )
                    """)
            
            logger.info("Sync databases initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing sync databases: {e}")
    
    def _load_sync_queue(self):
        """Load pending sync items from database."""
        try:
            with sqlite3.connect(self.sync_db_path) as conn:
                cursor = conn.execute("""
                    SELECT item_id, item_type, local_data, remote_data,
                           local_timestamp, remote_timestamp, sync_status,
                           conflict_resolution, retry_count, last_sync_attempt
                    FROM sync_queue
                    WHERE sync_status IN ('pending', 'failed', 'conflict')
                    ORDER BY local_timestamp ASC
                """)
                
                for row in cursor.fetchall():
                    sync_item = SyncItem(
                        item_id=row[0],
                        item_type=row[1],
                        local_data=json.loads(row[2]),
                        remote_data=json.loads(row[3]) if row[3] else None,
                        local_timestamp=datetime.fromisoformat(row[4]),
                        remote_timestamp=datetime.fromisoformat(row[5]) if row[5] else None,
                        sync_status=SyncStatus(row[6]),
                        conflict_resolution=ConflictResolution(row[7]) if row[7] else None,
                        retry_count=row[8],
                        last_sync_attempt=datetime.fromisoformat(row[9]) if row[9] else None
                    )
                    
                    self.sync_queue.append(sync_item)
            
            logger.info(f"Loaded {len(self.sync_queue)} pending sync items")
            
        except Exception as e:
            logger.error(f"Error loading sync queue: {e}")
    
    async def start_sync_service(self):
        """Start the background synchronization service."""
        if self._sync_task is None or self._sync_task.done():
            self._sync_task = asyncio.create_task(self._sync_loop())
            logger.info("Data sync service started")
    
    async def stop_sync_service(self):
        """Stop the background synchronization service."""
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            logger.info("Data sync service stopped")
    
    async def _sync_loop(self):
        """Background sync loop."""
        while True:
            try:
                await asyncio.sleep(self.sync_interval_minutes * 60)
                
                if not self.sync_in_progress and self.sync_queue:
                    await self.perform_sync()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def add_to_sync_queue(
        self,
        item_id: str,
        item_type: str,
        local_data: Dict[str, Any],
        priority: bool = False
    ):
        """
        Add item to synchronization queue.
        
        Args:
            item_id: Unique identifier for the item
            item_type: Type of item (query, model, preference, analytics)
            local_data: Local data to sync
            priority: Whether to prioritize this item
        """
        try:
            sync_item = SyncItem(
                item_id=item_id,
                item_type=item_type,
                local_data=local_data,
                local_timestamp=datetime.now(),
                sync_status=SyncStatus.PENDING
            )
            
            # Add to queue
            if priority:
                self.sync_queue.insert(0, sync_item)
            else:
                self.sync_queue.append(sync_item)
            
            # Store in database
            with sqlite3.connect(self.sync_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO sync_queue 
                    (item_id, item_type, local_data, local_timestamp, 
                     sync_status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    item_id, item_type, json.dumps(local_data),
                    sync_item.local_timestamp.isoformat(),
                    sync_item.sync_status.value,
                    datetime.now().isoformat()
                ))
            
            logger.debug(f"Added {item_type} item to sync queue: {item_id}")
            
        except Exception as e:
            logger.error(f"Error adding item to sync queue: {e}")
    
    async def perform_sync(self) -> Dict[str, Any]:
        """
        Perform synchronization of queued items.
        
        Returns:
            Sync results summary
        """
        if self.sync_in_progress:
            return {"status": "already_in_progress"}
        
        self.sync_in_progress = True
        sync_start_time = datetime.now()
        
        try:
            # Check network connectivity
            if not await self._check_network_connectivity():
                logger.info("No network connectivity - skipping sync")
                return {"status": "no_network", "queued_items": len(self.sync_queue)}
            
            sync_results = {
                "status": "completed",
                "total_items": len(self.sync_queue),
                "successful": 0,
                "failed": 0,
                "conflicts": 0,
                "skipped": 0
            }
            
            # Process sync queue
            items_to_remove = []
            
            for sync_item in self.sync_queue:
                try:
                    # Skip items that have exceeded retry limit
                    if sync_item.retry_count >= self.max_retry_attempts:
                        sync_results["skipped"] += 1
                        continue
                    
                    # Update sync status
                    sync_item.sync_status = SyncStatus.IN_PROGRESS
                    sync_item.last_sync_attempt = datetime.now()
                    
                    # Perform sync based on item type
                    result = await self._sync_item(sync_item)
                    
                    if result["success"]:
                        sync_item.sync_status = SyncStatus.COMPLETED
                        sync_results["successful"] += 1
                        items_to_remove.append(sync_item)
                        
                    elif result.get("conflict"):
                        sync_item.sync_status = SyncStatus.CONFLICT
                        sync_item.remote_data = result.get("remote_data")
                        sync_item.remote_timestamp = result.get("remote_timestamp")
                        sync_results["conflicts"] += 1
                        
                    else:
                        sync_item.sync_status = SyncStatus.FAILED
                        sync_item.retry_count += 1
                        sync_results["failed"] += 1
                    
                    # Update database
                    await self._update_sync_item_in_db(sync_item)
                    
                except Exception as e:
                    logger.error(f"Error syncing item {sync_item.item_id}: {e}")
                    sync_item.sync_status = SyncStatus.FAILED
                    sync_item.retry_count += 1
                    sync_results["failed"] += 1
                    await self._update_sync_item_in_db(sync_item)
            
            # Remove completed items from queue
            for item in items_to_remove:
                self.sync_queue.remove(item)
                await self._remove_sync_item_from_db(item.item_id, item.item_type)
            
            # Update statistics
            self.sync_stats['total_syncs'] += 1
            self.sync_stats['successful_syncs'] += sync_results["successful"]
            self.sync_stats['failed_syncs'] += sync_results["failed"]
            self.sync_stats['conflicts_resolved'] += sync_results["conflicts"]
            
            self.last_sync_time = datetime.now()
            
            # Log sync event
            if self.enable_analytics:
                await self._log_sync_event(
                    "sync_completed",
                    sync_results,
                    True,
                    (datetime.now() - sync_start_time).total_seconds()
                )
            
            logger.info(f"Sync completed: {sync_results}")
            return sync_results
            
        except Exception as e:
            logger.error(f"Error during sync: {e}")
            
            if self.enable_analytics:
                await self._log_sync_event(
                    "sync_failed",
                    {"error": str(e)},
                    False,
                    (datetime.now() - sync_start_time).total_seconds()
                )
            
            return {"status": "error", "error": str(e)}
            
        finally:
            self.sync_in_progress = False
    
    async def _sync_item(self, sync_item: SyncItem) -> Dict[str, Any]:
        """
        Sync individual item with remote service.
        
        Args:
            sync_item: Item to synchronize
            
        Returns:
            Sync result
        """
        try:
            if sync_item.item_type == "query":
                return await self._sync_query(sync_item)
            elif sync_item.item_type == "model":
                return await self._sync_model(sync_item)
            elif sync_item.item_type == "preference":
                return await self._sync_preference(sync_item)
            elif sync_item.item_type == "analytics":
                return await self._sync_analytics(sync_item)
            else:
                logger.warning(f"Unknown sync item type: {sync_item.item_type}")
                return {"success": False, "error": "unknown_item_type"}
                
        except Exception as e:
            logger.error(f"Error syncing {sync_item.item_type} item: {e}")
            return {"success": False, "error": str(e)}
    
    async def _sync_query(self, sync_item: SyncItem) -> Dict[str, Any]:
        """Sync cached query with remote service."""
        try:
            # Simulate remote API call
            # In production, this would call the actual remote service
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Check if remote has newer version
            remote_data = await self._fetch_remote_query(sync_item.item_id)
            
            if remote_data:
                remote_timestamp = datetime.fromisoformat(remote_data.get("timestamp"))
                
                # Check for conflicts
                if remote_timestamp > sync_item.local_timestamp:
                    # Remote is newer - potential conflict
                    return {
                        "success": False,
                        "conflict": True,
                        "remote_data": remote_data,
                        "remote_timestamp": remote_timestamp
                    }
            
            # Upload local data to remote
            upload_success = await self._upload_query_to_remote(sync_item)
            
            return {"success": upload_success}
            
        except Exception as e:
            logger.error(f"Error syncing query: {e}")
            return {"success": False, "error": str(e)}
    
    async def _sync_model(self, sync_item: SyncItem) -> Dict[str, Any]:
        """Sync cached model with remote service."""
        try:
            # For models, we typically only upload usage statistics
            # The actual model data is usually too large for frequent sync
            
            model_stats = {
                "model_id": sync_item.item_id,
                "usage_count": sync_item.local_data.get("usage_count", 0),
                "last_used": sync_item.local_data.get("last_used"),
                "performance_metrics": sync_item.local_data.get("performance_metrics", {})
            }
            
            # Upload model statistics
            upload_success = await self._upload_model_stats_to_remote(model_stats)
            
            return {"success": upload_success}
            
        except Exception as e:
            logger.error(f"Error syncing model: {e}")
            return {"success": False, "error": str(e)}
    
    async def _sync_preference(self, sync_item: SyncItem) -> Dict[str, Any]:
        """Sync user preferences with remote service."""
        try:
            # Fetch remote preferences
            remote_prefs = await self._fetch_remote_preferences(sync_item.item_id)
            
            if remote_prefs:
                # Merge preferences intelligently
                merged_prefs = self._merge_preferences(
                    sync_item.local_data,
                    remote_prefs
                )
                
                # Upload merged preferences
                upload_success = await self._upload_preferences_to_remote(
                    sync_item.item_id, merged_prefs
                )
                
                return {"success": upload_success}
            else:
                # No remote preferences, upload local ones
                upload_success = await self._upload_preferences_to_remote(
                    sync_item.item_id, sync_item.local_data
                )
                
                return {"success": upload_success}
            
        except Exception as e:
            logger.error(f"Error syncing preferences: {e}")
            return {"success": False, "error": str(e)}
    
    async def _sync_analytics(self, sync_item: SyncItem) -> Dict[str, Any]:
        """Sync offline usage analytics with remote service."""
        try:
            # Upload analytics data to remote service
            upload_success = await self._upload_analytics_to_remote(sync_item.local_data)
            
            return {"success": upload_success}
            
        except Exception as e:
            logger.error(f"Error syncing analytics: {e}")
            return {"success": False, "error": str(e)}
    
    async def resolve_conflict(
        self,
        sync_item: SyncItem,
        resolution_strategy: Optional[ConflictResolution] = None
    ) -> Dict[str, Any]:
        """
        Resolve synchronization conflict.
        
        Args:
            sync_item: Conflicted sync item
            resolution_strategy: Strategy to use for resolution
            
        Returns:
            Resolution result
        """
        try:
            strategy = resolution_strategy or self.conflict_resolution_strategy
            
            if strategy == ConflictResolution.LOCAL_WINS:
                # Keep local data, overwrite remote
                result = await self._upload_data_to_remote(sync_item, force=True)
                
            elif strategy == ConflictResolution.REMOTE_WINS:
                # Keep remote data, overwrite local
                result = await self._download_data_from_remote(sync_item, force=True)
                
            elif strategy == ConflictResolution.TIMESTAMP_BASED:
                # Use newer timestamp
                if (sync_item.remote_timestamp and 
                    sync_item.remote_timestamp > sync_item.local_timestamp):
                    result = await self._download_data_from_remote(sync_item)
                else:
                    result = await self._upload_data_to_remote(sync_item)
                    
            elif strategy == ConflictResolution.MERGE:
                # Attempt to merge data
                merged_data = self._merge_data(
                    sync_item.local_data,
                    sync_item.remote_data
                )
                sync_item.local_data = merged_data
                result = await self._upload_data_to_remote(sync_item)
                
            else:
                # USER_CHOICE - requires manual intervention
                return {
                    "success": False,
                    "requires_user_input": True,
                    "local_data": sync_item.local_data,
                    "remote_data": sync_item.remote_data
                }
            
            if result.get("success"):
                sync_item.sync_status = SyncStatus.COMPLETED
                sync_item.conflict_resolution = strategy
                await self._update_sync_item_in_db(sync_item)
                
                self.sync_stats['conflicts_resolved'] += 1
                
                logger.info(f"Conflict resolved using {strategy.value} strategy")
            
            return result
            
        except Exception as e:
            logger.error(f"Error resolving conflict: {e}")
            return {"success": False, "error": str(e)}
    
    def _merge_data(self, local_data: Dict[str, Any], remote_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge local and remote data intelligently.
        
        Args:
            local_data: Local data
            remote_data: Remote data
            
        Returns:
            Merged data
        """
        try:
            merged = local_data.copy()
            
            for key, remote_value in remote_data.items():
                if key not in merged:
                    # New key from remote
                    merged[key] = remote_value
                elif isinstance(remote_value, dict) and isinstance(merged[key], dict):
                    # Recursively merge dictionaries
                    merged[key] = self._merge_data(merged[key], remote_value)
                elif isinstance(remote_value, list) and isinstance(merged[key], list):
                    # Merge lists (remove duplicates)
                    merged[key] = list(set(merged[key] + remote_value))
                elif key.endswith('_count') or key.endswith('_total'):
                    # Sum numeric counters
                    merged[key] = merged[key] + remote_value
                # For other types, keep local value (local wins for conflicts)
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging data: {e}")
            return local_data
    
    def _merge_preferences(self, local_prefs: Dict[str, Any], remote_prefs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user preferences with conflict resolution."""
        try:
            merged = remote_prefs.copy()  # Start with remote as base
            
            # Override with local preferences for user-specific settings
            user_specific_keys = [
                'language_preference', 'accent_preference', 'voice_speed',
                'volume_level', 'notification_settings', 'privacy_settings'
            ]
            
            for key in user_specific_keys:
                if key in local_prefs:
                    merged[key] = local_prefs[key]
            
            # Merge usage statistics
            if 'usage_stats' in local_prefs and 'usage_stats' in remote_prefs:
                merged['usage_stats'] = self._merge_data(
                    local_prefs['usage_stats'],
                    remote_prefs['usage_stats']
                )
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging preferences: {e}")
            return local_prefs
    
    async def start_offline_session(self, session_id: Optional[str] = None) -> str:
        """
        Start tracking an offline usage session.
        
        Args:
            session_id: Optional session ID, generated if not provided
            
        Returns:
            Session ID
        """
        try:
            if not self.enable_analytics:
                return ""
            
            session_id = session_id or self._generate_session_id()
            
            self.current_session = OfflineUsageRecord(
                session_id=session_id,
                start_time=datetime.now()
            )
            
            # Store in database
            with sqlite3.connect(self.analytics_db_path) as conn:
                conn.execute("""
                    INSERT INTO offline_sessions 
                    (session_id, start_time, created_at)
                    VALUES (?, ?, ?)
                """, (
                    session_id,
                    self.current_session.start_time.isoformat(),
                    datetime.now().isoformat()
                ))
            
            logger.info(f"Started offline session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error starting offline session: {e}")
            return ""
    
    async def end_offline_session(self, user_satisfaction: Optional[float] = None):
        """
        End the current offline usage session.
        
        Args:
            user_satisfaction: Optional user satisfaction rating (0.0-5.0)
        """
        try:
            if not self.enable_analytics or not self.current_session:
                return
            
            self.current_session.end_time = datetime.now()
            self.current_session.user_satisfaction = user_satisfaction
            
            # Update database
            with sqlite3.connect(self.analytics_db_path) as conn:
                conn.execute("""
                    UPDATE offline_sessions 
                    SET end_time = ?, queries_processed = ?, tts_synthesized = ?,
                        languages_used = ?, network_interruptions = ?,
                        cache_hit_rate = ?, user_satisfaction = ?
                    WHERE session_id = ?
                """, (
                    self.current_session.end_time.isoformat(),
                    self.current_session.queries_processed,
                    self.current_session.tts_synthesized,
                    json.dumps([lang.value for lang in self.current_session.languages_used]),
                    self.current_session.network_interruptions,
                    self.current_session.cache_hit_rate,
                    self.current_session.user_satisfaction,
                    self.current_session.session_id
                ))
            
            # Add to sync queue for upload
            await self.add_to_sync_queue(
                f"session_{self.current_session.session_id}",
                "analytics",
                self.current_session.dict()
            )
            
            logger.info(f"Ended offline session: {self.current_session.session_id}")
            self.current_session = None
            
        except Exception as e:
            logger.error(f"Error ending offline session: {e}")
    
    def record_offline_activity(
        self,
        activity_type: str,
        language: Optional[LanguageCode] = None,
        cache_hit: bool = False
    ):
        """
        Record offline activity for current session.
        
        Args:
            activity_type: Type of activity ("query", "tts", "network_interruption")
            language: Language used (if applicable)
            cache_hit: Whether this was a cache hit
        """
        try:
            if not self.enable_analytics or not self.current_session:
                return
            
            if activity_type == "query":
                self.current_session.queries_processed += 1
            elif activity_type == "tts":
                self.current_session.tts_synthesized += 1
            elif activity_type == "network_interruption":
                self.current_session.network_interruptions += 1
            
            if language and language not in self.current_session.languages_used:
                self.current_session.languages_used.append(language)
            
            # Update cache hit rate
            if activity_type in ["query", "tts"]:
                total_activities = (
                    self.current_session.queries_processed + 
                    self.current_session.tts_synthesized
                )
                if total_activities > 0:
                    # This is a simplified calculation - in production, track hits/misses separately
                    if cache_hit:
                        self.current_session.cache_hit_rate = min(
                            1.0, self.current_session.cache_hit_rate + 0.1
                        )
                    else:
                        self.current_session.cache_hit_rate = max(
                            0.0, self.current_session.cache_hit_rate - 0.05
                        )
            
        except Exception as e:
            logger.error(f"Error recording offline activity: {e}")
    
    async def get_offline_analytics(
        self,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get offline usage analytics.
        
        Args:
            days_back: Number of days to include in analytics
            
        Returns:
            Analytics data
        """
        try:
            if not self.enable_analytics:
                return {"analytics_disabled": True}
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            analytics = {
                "period_days": days_back,
                "total_sessions": 0,
                "total_offline_time_hours": 0.0,
                "total_queries": 0,
                "total_tts_synthesized": 0,
                "average_session_duration_minutes": 0.0,
                "languages_used": {},
                "average_cache_hit_rate": 0.0,
                "network_interruptions": 0,
                "user_satisfaction_average": 0.0,
                "sync_statistics": self.sync_stats.copy()
            }
            
            with sqlite3.connect(self.analytics_db_path) as conn:
                # Get session statistics
                cursor = conn.execute("""
                    SELECT COUNT(*), 
                           AVG(CASE WHEN end_time IS NOT NULL 
                               THEN (julianday(end_time) - julianday(start_time)) * 24 * 60 
                               ELSE 0 END),
                           SUM(queries_processed),
                           SUM(tts_synthesized),
                           AVG(cache_hit_rate),
                           SUM(network_interruptions),
                           AVG(user_satisfaction)
                    FROM offline_sessions 
                    WHERE start_time >= ?
                """, (cutoff_date.isoformat(),))
                
                row = cursor.fetchone()
                if row:
                    analytics["total_sessions"] = row[0] or 0
                    analytics["average_session_duration_minutes"] = row[1] or 0.0
                    analytics["total_queries"] = row[2] or 0
                    analytics["total_tts_synthesized"] = row[3] or 0
                    analytics["average_cache_hit_rate"] = row[4] or 0.0
                    analytics["network_interruptions"] = row[5] or 0
                    analytics["user_satisfaction_average"] = row[6] or 0.0
                
                # Calculate total offline time
                cursor = conn.execute("""
                    SELECT SUM(CASE WHEN end_time IS NOT NULL 
                               THEN (julianday(end_time) - julianday(start_time)) * 24 
                               ELSE 0 END)
                    FROM offline_sessions 
                    WHERE start_time >= ?
                """, (cutoff_date.isoformat(),))
                
                result = cursor.fetchone()
                if result and result[0]:
                    analytics["total_offline_time_hours"] = result[0]
                
                # Get language usage statistics
                cursor = conn.execute("""
                    SELECT languages_used FROM offline_sessions 
                    WHERE start_time >= ? AND languages_used IS NOT NULL
                """, (cutoff_date.isoformat(),))
                
                language_counts = {}
                for row in cursor.fetchall():
                    try:
                        languages = json.loads(row[0])
                        for lang in languages:
                            language_counts[lang] = language_counts.get(lang, 0) + 1
                    except (json.JSONDecodeError, TypeError):
                        continue
                
                analytics["languages_used"] = language_counts
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting offline analytics: {e}")
            return {"error": str(e)}
    
    def get_sync_status(self) -> Dict[str, Any]:
        """
        Get current synchronization status.
        
        Returns:
            Sync status information
        """
        try:
            pending_by_type = {}
            failed_by_type = {}
            conflicts_by_type = {}
            
            for item in self.sync_queue:
                if item.sync_status == SyncStatus.PENDING:
                    pending_by_type[item.item_type] = pending_by_type.get(item.item_type, 0) + 1
                elif item.sync_status == SyncStatus.FAILED:
                    failed_by_type[item.item_type] = failed_by_type.get(item.item_type, 0) + 1
                elif item.sync_status == SyncStatus.CONFLICT:
                    conflicts_by_type[item.item_type] = conflicts_by_type.get(item.item_type, 0) + 1
            
            return {
                "sync_in_progress": self.sync_in_progress,
                "last_sync_time": self.last_sync_time.isoformat() if self.last_sync_time else None,
                "total_queued_items": len(self.sync_queue),
                "pending_by_type": pending_by_type,
                "failed_by_type": failed_by_type,
                "conflicts_by_type": conflicts_by_type,
                "sync_statistics": self.sync_stats.copy(),
                "current_session_active": self.current_session is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting sync status: {e}")
            return {"error": str(e)}
    
    # Helper methods for remote API calls (these would be implemented with actual API calls)
    
    async def _check_network_connectivity(self) -> bool:
        """Check if network connectivity is available."""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except (socket.error, OSError):
            return False
    
    async def _fetch_remote_query(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Fetch query data from remote service."""
        # Simulate API call
        await asyncio.sleep(0.1)
        return None  # No remote data found
    
    async def _upload_query_to_remote(self, sync_item: SyncItem) -> bool:
        """Upload query to remote service."""
        # Simulate API call
        await asyncio.sleep(0.1)
        return True  # Success
    
    async def _upload_model_stats_to_remote(self, model_stats: Dict[str, Any]) -> bool:
        """Upload model statistics to remote service."""
        # Simulate API call
        await asyncio.sleep(0.1)
        return True  # Success
    
    async def _fetch_remote_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetch user preferences from remote service."""
        # Simulate API call
        await asyncio.sleep(0.1)
        return None  # No remote preferences found
    
    async def _upload_preferences_to_remote(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Upload user preferences to remote service."""
        # Simulate API call
        await asyncio.sleep(0.1)
        return True  # Success
    
    async def _upload_analytics_to_remote(self, analytics_data: Dict[str, Any]) -> bool:
        """Upload analytics data to remote service."""
        # Simulate API call
        await asyncio.sleep(0.1)
        return True  # Success
    
    async def _upload_data_to_remote(self, sync_item: SyncItem, force: bool = False) -> Dict[str, Any]:
        """Upload data to remote service."""
        # Simulate API call
        await asyncio.sleep(0.1)
        return {"success": True}
    
    async def _download_data_from_remote(self, sync_item: SyncItem, force: bool = False) -> Dict[str, Any]:
        """Download data from remote service."""
        # Simulate API call
        await asyncio.sleep(0.1)
        return {"success": True}
    
    async def _update_sync_item_in_db(self, sync_item: SyncItem):
        """Update sync item in database."""
        try:
            with sqlite3.connect(self.sync_db_path) as conn:
                conn.execute("""
                    UPDATE sync_queue 
                    SET remote_data = ?, remote_timestamp = ?, sync_status = ?,
                        conflict_resolution = ?, retry_count = ?, last_sync_attempt = ?
                    WHERE item_id = ? AND item_type = ?
                """, (
                    json.dumps(sync_item.remote_data) if sync_item.remote_data else None,
                    sync_item.remote_timestamp.isoformat() if sync_item.remote_timestamp else None,
                    sync_item.sync_status.value,
                    sync_item.conflict_resolution.value if sync_item.conflict_resolution else None,
                    sync_item.retry_count,
                    sync_item.last_sync_attempt.isoformat() if sync_item.last_sync_attempt else None,
                    sync_item.item_id,
                    sync_item.item_type
                ))
        except Exception as e:
            logger.error(f"Error updating sync item in database: {e}")
    
    async def _remove_sync_item_from_db(self, item_id: str, item_type: str):
        """Remove sync item from database."""
        try:
            with sqlite3.connect(self.sync_db_path) as conn:
                conn.execute("""
                    DELETE FROM sync_queue 
                    WHERE item_id = ? AND item_type = ?
                """, (item_id, item_type))
        except Exception as e:
            logger.error(f"Error removing sync item from database: {e}")
    
    async def _log_sync_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        success: bool,
        duration_seconds: float
    ):
        """Log sync event for analytics."""
        try:
            with sqlite3.connect(self.analytics_db_path) as conn:
                conn.execute("""
                    INSERT INTO sync_events 
                    (event_type, timestamp, details, success, duration_seconds)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    event_type,
                    datetime.now().isoformat(),
                    json.dumps(details),
                    success,
                    duration_seconds
                ))
        except Exception as e:
            logger.error(f"Error logging sync event: {e}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        import uuid
        return f"offline_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"


# Factory function for creating data sync manager
def create_data_sync_manager(
    cache_dir: str = ".bharatvoice_offline",
    sync_interval_minutes: int = 15,
    max_retry_attempts: int = 3,
    conflict_resolution_strategy: ConflictResolution = ConflictResolution.TIMESTAMP_BASED,
    enable_analytics: bool = True
) -> DataSyncManager:
    """
    Factory function to create a data sync manager instance.
    
    Args:
        cache_dir: Directory for offline cache storage
        sync_interval_minutes: Interval between sync attempts
        max_retry_attempts: Maximum retry attempts for failed syncs
        conflict_resolution_strategy: Default conflict resolution strategy
        enable_analytics: Whether to enable offline usage analytics
        
    Returns:
        Configured DataSyncManager instance
    """
    return DataSyncManager(
        cache_dir=cache_dir,
        sync_interval_minutes=sync_interval_minutes,
        max_retry_attempts=max_retry_attempts,
        conflict_resolution_strategy=conflict_resolution_strategy,
        enable_analytics=enable_analytics
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    )