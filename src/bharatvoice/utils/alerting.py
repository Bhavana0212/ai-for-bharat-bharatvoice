<<<<<<< HEAD
"""
Alerting and notification system for BharatVoice Assistant.

This module provides comprehensive alerting capabilities for system health,
performance degradation, and critical errors with multiple notification channels.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, asdict

import structlog
from bharatvoice.core.models import LanguageCode


logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(str, Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(str, Enum):
    """Notification channels."""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    service: str
    metric: str
    threshold: float
    current_value: float
    created_at: datetime
    updated_at: datetime
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data


class AlertRule:
    """Alert rule definition."""
    
    def __init__(
        self,
        name: str,
        description: str,
        service: str,
        metric: str,
        condition: str,  # "greater_than", "less_than", "equals", "not_equals"
        threshold: float,
        severity: AlertSeverity,
        duration: int = 60,  # seconds
        cooldown: int = 300,  # seconds
        enabled: bool = True
    ):
        self.name = name
        self.description = description
        self.service = service
        self.metric = metric
        self.condition = condition
        self.threshold = threshold
        self.severity = severity
        self.duration = duration
        self.cooldown = cooldown
        self.enabled = enabled
        self.last_triggered = None
        self.breach_start_time = None
    
    def evaluate(self, current_value: float) -> bool:
        """Evaluate if alert condition is met."""
        if not self.enabled:
            return False
        
        condition_met = False
        
        if self.condition == "greater_than":
            condition_met = current_value > self.threshold
        elif self.condition == "less_than":
            condition_met = current_value < self.threshold
        elif self.condition == "equals":
            condition_met = current_value == self.threshold
        elif self.condition == "not_equals":
            condition_met = current_value != self.threshold
        
        current_time = time.time()
        
        if condition_met:
            if self.breach_start_time is None:
                self.breach_start_time = current_time
            
            # Check if breach duration exceeded
            if current_time - self.breach_start_time >= self.duration:
                # Check cooldown period
                if (self.last_triggered is None or 
                    current_time - self.last_triggered >= self.cooldown):
                    self.last_triggered = current_time
                    return True
        else:
            # Reset breach start time if condition no longer met
            self.breach_start_time = None
        
        return False


class NotificationHandler:
    """Base notification handler."""
    
    async def send_notification(self, alert: Alert, channel_config: Dict[str, Any]) -> bool:
        """Send notification for alert."""
        raise NotImplementedError


class LogNotificationHandler(NotificationHandler):
    """Log-based notification handler."""
    
    async def send_notification(self, alert: Alert, channel_config: Dict[str, Any]) -> bool:
        """Log alert notification."""
        try:
            logger.warning(
                "ALERT",
                alert_id=alert.id,
                title=alert.title,
                description=alert.description,
                severity=alert.severity.value,
                service=alert.service,
                metric=alert.metric,
                threshold=alert.threshold,
                current_value=alert.current_value,
                created_at=alert.created_at.isoformat()
            )
            return True
        except Exception as e:
            logger.error("Failed to send log notification", exc_info=e)
            return False


class WebhookNotificationHandler(NotificationHandler):
    """Webhook-based notification handler."""
    
    async def send_notification(self, alert: Alert, channel_config: Dict[str, Any]) -> bool:
        """Send webhook notification."""
        try:
            import aiohttp
            
            webhook_url = channel_config.get("url")
            if not webhook_url:
                logger.error("Webhook URL not configured")
                return False
            
            payload = {
                "alert": alert.to_dict(),
                "timestamp": datetime.utcnow().isoformat(),
                "source": "bharatvoice-assistant"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info("Webhook notification sent", alert_id=alert.id)
                        return True
                    else:
                        logger.error(
                            "Webhook notification failed",
                            alert_id=alert.id,
                            status_code=response.status
                        )
                        return False
        
        except Exception as e:
            logger.error("Failed to send webhook notification", exc_info=e)
            return False


class AlertManager:
    """
    Comprehensive alert management system.
    """
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers = {
            NotificationChannel.LOG: LogNotificationHandler(),
            NotificationChannel.WEBHOOK: WebhookNotificationHandler()
        }
        self.notification_channels: Dict[AlertSeverity, List[Dict[str, Any]]] = {}
        self.running = False
        self.evaluation_task = None
        self.evaluation_interval = 30  # seconds
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.name] = rule
        logger.info("Alert rule added", rule_name=rule.name, service=rule.service)
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info("Alert rule removed", rule_name=rule_name)
    
    def configure_notification_channel(
        self,
        severity: AlertSeverity,
        channel: NotificationChannel,
        config: Dict[str, Any]
    ):
        """Configure notification channel for alert severity."""
        if severity not in self.notification_channels:
            self.notification_channels[severity] = []
        
        self.notification_channels[severity].append({
            "channel": channel,
            "config": config
        })
        
        logger.info(
            "Notification channel configured",
            severity=severity.value,
            channel=channel.value
        )
    
    async def start(self):
        """Start alert monitoring."""
        if not self.running:
            self.running = True
            self.evaluation_task = asyncio.create_task(self._evaluation_loop())
            logger.info("Alert manager started")
    
    async def stop(self):
        """Stop alert monitoring."""
        self.running = False
        if self.evaluation_task:
            self.evaluation_task.cancel()
            try:
                await self.evaluation_task
            except asyncio.CancelledError:
                pass
        logger.info("Alert manager stopped")
    
    async def _evaluation_loop(self):
        """Main alert evaluation loop."""
        while self.running:
            try:
                await self._evaluate_alert_rules()
                await asyncio.sleep(self.evaluation_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Alert evaluation error", exc_info=e)
                await asyncio.sleep(self.evaluation_interval)
    
    async def _evaluate_alert_rules(self):
        """Evaluate all alert rules."""
        # TODO: Get actual metrics from monitoring system
        # For now, simulate some metrics
        mock_metrics = {
            "cpu_usage_percent": 75.0,
            "memory_usage_percent": 85.0,
            "response_time_ms": 2500.0,
            "error_rate_percent": 5.0,
            "active_connections": 150
        }
        
        for rule_name, rule in self.alert_rules.items():
            if rule.metric in mock_metrics:
                current_value = mock_metrics[rule.metric]
                
                if rule.evaluate(current_value):
                    await self._trigger_alert(rule, current_value)
    
    async def _trigger_alert(self, rule: AlertRule, current_value: float):
        """Trigger an alert."""
        alert_id = f"{rule.service}_{rule.metric}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            title=f"{rule.service.title()} {rule.metric} Alert",
            description=rule.description,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            service=rule.service,
            metric=rule.metric,
            threshold=rule.threshold,
            current_value=current_value,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={
                "rule_name": rule.name,
                "condition": rule.condition,
                "duration": rule.duration
            }
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        await self._send_alert_notifications(alert)
        
        logger.warning(
            "Alert triggered",
            alert_id=alert_id,
            rule_name=rule.name,
            service=rule.service,
            metric=rule.metric,
            current_value=current_value,
            threshold=rule.threshold
        )
    
    async def _send_alert_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        channels = self.notification_channels.get(alert.severity, [])
        
        for channel_config in channels:
            channel = channel_config["channel"]
            config = channel_config["config"]
            
            if channel in self.notification_handlers:
                handler = self.notification_handlers[channel]
                try:
                    success = await handler.send_notification(alert, config)
                    if success:
                        logger.info(
                            "Alert notification sent",
                            alert_id=alert.id,
                            channel=channel.value
                        )
                    else:
                        logger.error(
                            "Alert notification failed",
                            alert_id=alert.id,
                            channel=channel.value
                        )
                except Exception as e:
                    logger.error(
                        "Alert notification error",
                        alert_id=alert.id,
                        channel=channel.value,
                        exc_info=e
                    )
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            
            logger.info(
                "Alert acknowledged",
                alert_id=alert_id,
                acknowledged_by=acknowledged_by
            )
            return True
        
        return False
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            
            # Move to history and remove from active
            del self.active_alerts[alert_id]
            
            logger.info("Alert resolved", alert_id=alert_id)
            return True
        
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        
        severity_counts = {}
        for alert in self.alert_history:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        service_counts = {}
        for alert in self.alert_history:
            service = alert.service
            service_counts[service] = service_counts.get(service, 0) + 1
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "resolved_alerts": total_alerts - active_alerts,
            "severity_distribution": severity_counts,
            "service_distribution": service_counts,
            "alert_rules": len(self.alert_rules),
            "notification_channels": sum(len(channels) for channels in self.notification_channels.values())
        }


# Global alert manager instance
alert_manager = AlertManager()


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    return alert_manager


async def initialize_default_alert_rules():
    """Initialize default alert rules for system monitoring."""
    default_rules = [
        AlertRule(
            name="high_cpu_usage",
            description="CPU usage is above 80%",
            service="system",
            metric="cpu_usage_percent",
            condition="greater_than",
            threshold=80.0,
            severity=AlertSeverity.HIGH,
            duration=120,
            cooldown=300
        ),
        AlertRule(
            name="high_memory_usage",
            description="Memory usage is above 90%",
            service="system",
            metric="memory_usage_percent",
            condition="greater_than",
            threshold=90.0,
            severity=AlertSeverity.CRITICAL,
            duration=60,
            cooldown=300
        ),
        AlertRule(
            name="slow_response_time",
            description="Average response time is above 2 seconds",
            service="api",
            metric="response_time_ms",
            condition="greater_than",
            threshold=2000.0,
            severity=AlertSeverity.MEDIUM,
            duration=180,
            cooldown=600
        ),
        AlertRule(
            name="high_error_rate",
            description="Error rate is above 5%",
            service="api",
            metric="error_rate_percent",
            condition="greater_than",
            threshold=5.0,
            severity=AlertSeverity.HIGH,
            duration=120,
            cooldown=300
        )
    ]
    
    for rule in default_rules:
        alert_manager.add_alert_rule(rule)
    
    # Configure default notification channels
    alert_manager.configure_notification_channel(
        AlertSeverity.CRITICAL,
        NotificationChannel.LOG,
        {}
    )
    
    alert_manager.configure_notification_channel(
        AlertSeverity.HIGH,
        NotificationChannel.LOG,
        {}
    )
    
=======
"""
Alerting and notification system for BharatVoice Assistant.

This module provides comprehensive alerting capabilities for system health,
performance degradation, and critical errors with multiple notification channels.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, asdict

import structlog
from bharatvoice.core.models import LanguageCode


logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(str, Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(str, Enum):
    """Notification channels."""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    service: str
    metric: str
    threshold: float
    current_value: float
    created_at: datetime
    updated_at: datetime
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data


class AlertRule:
    """Alert rule definition."""
    
    def __init__(
        self,
        name: str,
        description: str,
        service: str,
        metric: str,
        condition: str,  # "greater_than", "less_than", "equals", "not_equals"
        threshold: float,
        severity: AlertSeverity,
        duration: int = 60,  # seconds
        cooldown: int = 300,  # seconds
        enabled: bool = True
    ):
        self.name = name
        self.description = description
        self.service = service
        self.metric = metric
        self.condition = condition
        self.threshold = threshold
        self.severity = severity
        self.duration = duration
        self.cooldown = cooldown
        self.enabled = enabled
        self.last_triggered = None
        self.breach_start_time = None
    
    def evaluate(self, current_value: float) -> bool:
        """Evaluate if alert condition is met."""
        if not self.enabled:
            return False
        
        condition_met = False
        
        if self.condition == "greater_than":
            condition_met = current_value > self.threshold
        elif self.condition == "less_than":
            condition_met = current_value < self.threshold
        elif self.condition == "equals":
            condition_met = current_value == self.threshold
        elif self.condition == "not_equals":
            condition_met = current_value != self.threshold
        
        current_time = time.time()
        
        if condition_met:
            if self.breach_start_time is None:
                self.breach_start_time = current_time
            
            # Check if breach duration exceeded
            if current_time - self.breach_start_time >= self.duration:
                # Check cooldown period
                if (self.last_triggered is None or 
                    current_time - self.last_triggered >= self.cooldown):
                    self.last_triggered = current_time
                    return True
        else:
            # Reset breach start time if condition no longer met
            self.breach_start_time = None
        
        return False


class NotificationHandler:
    """Base notification handler."""
    
    async def send_notification(self, alert: Alert, channel_config: Dict[str, Any]) -> bool:
        """Send notification for alert."""
        raise NotImplementedError


class LogNotificationHandler(NotificationHandler):
    """Log-based notification handler."""
    
    async def send_notification(self, alert: Alert, channel_config: Dict[str, Any]) -> bool:
        """Log alert notification."""
        try:
            logger.warning(
                "ALERT",
                alert_id=alert.id,
                title=alert.title,
                description=alert.description,
                severity=alert.severity.value,
                service=alert.service,
                metric=alert.metric,
                threshold=alert.threshold,
                current_value=alert.current_value,
                created_at=alert.created_at.isoformat()
            )
            return True
        except Exception as e:
            logger.error("Failed to send log notification", exc_info=e)
            return False


class WebhookNotificationHandler(NotificationHandler):
    """Webhook-based notification handler."""
    
    async def send_notification(self, alert: Alert, channel_config: Dict[str, Any]) -> bool:
        """Send webhook notification."""
        try:
            import aiohttp
            
            webhook_url = channel_config.get("url")
            if not webhook_url:
                logger.error("Webhook URL not configured")
                return False
            
            payload = {
                "alert": alert.to_dict(),
                "timestamp": datetime.utcnow().isoformat(),
                "source": "bharatvoice-assistant"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info("Webhook notification sent", alert_id=alert.id)
                        return True
                    else:
                        logger.error(
                            "Webhook notification failed",
                            alert_id=alert.id,
                            status_code=response.status
                        )
                        return False
        
        except Exception as e:
            logger.error("Failed to send webhook notification", exc_info=e)
            return False


class AlertManager:
    """
    Comprehensive alert management system.
    """
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers = {
            NotificationChannel.LOG: LogNotificationHandler(),
            NotificationChannel.WEBHOOK: WebhookNotificationHandler()
        }
        self.notification_channels: Dict[AlertSeverity, List[Dict[str, Any]]] = {}
        self.running = False
        self.evaluation_task = None
        self.evaluation_interval = 30  # seconds
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.name] = rule
        logger.info("Alert rule added", rule_name=rule.name, service=rule.service)
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info("Alert rule removed", rule_name=rule_name)
    
    def configure_notification_channel(
        self,
        severity: AlertSeverity,
        channel: NotificationChannel,
        config: Dict[str, Any]
    ):
        """Configure notification channel for alert severity."""
        if severity not in self.notification_channels:
            self.notification_channels[severity] = []
        
        self.notification_channels[severity].append({
            "channel": channel,
            "config": config
        })
        
        logger.info(
            "Notification channel configured",
            severity=severity.value,
            channel=channel.value
        )
    
    async def start(self):
        """Start alert monitoring."""
        if not self.running:
            self.running = True
            self.evaluation_task = asyncio.create_task(self._evaluation_loop())
            logger.info("Alert manager started")
    
    async def stop(self):
        """Stop alert monitoring."""
        self.running = False
        if self.evaluation_task:
            self.evaluation_task.cancel()
            try:
                await self.evaluation_task
            except asyncio.CancelledError:
                pass
        logger.info("Alert manager stopped")
    
    async def _evaluation_loop(self):
        """Main alert evaluation loop."""
        while self.running:
            try:
                await self._evaluate_alert_rules()
                await asyncio.sleep(self.evaluation_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Alert evaluation error", exc_info=e)
                await asyncio.sleep(self.evaluation_interval)
    
    async def _evaluate_alert_rules(self):
        """Evaluate all alert rules."""
        # TODO: Get actual metrics from monitoring system
        # For now, simulate some metrics
        mock_metrics = {
            "cpu_usage_percent": 75.0,
            "memory_usage_percent": 85.0,
            "response_time_ms": 2500.0,
            "error_rate_percent": 5.0,
            "active_connections": 150
        }
        
        for rule_name, rule in self.alert_rules.items():
            if rule.metric in mock_metrics:
                current_value = mock_metrics[rule.metric]
                
                if rule.evaluate(current_value):
                    await self._trigger_alert(rule, current_value)
    
    async def _trigger_alert(self, rule: AlertRule, current_value: float):
        """Trigger an alert."""
        alert_id = f"{rule.service}_{rule.metric}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            title=f"{rule.service.title()} {rule.metric} Alert",
            description=rule.description,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            service=rule.service,
            metric=rule.metric,
            threshold=rule.threshold,
            current_value=current_value,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={
                "rule_name": rule.name,
                "condition": rule.condition,
                "duration": rule.duration
            }
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        await self._send_alert_notifications(alert)
        
        logger.warning(
            "Alert triggered",
            alert_id=alert_id,
            rule_name=rule.name,
            service=rule.service,
            metric=rule.metric,
            current_value=current_value,
            threshold=rule.threshold
        )
    
    async def _send_alert_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        channels = self.notification_channels.get(alert.severity, [])
        
        for channel_config in channels:
            channel = channel_config["channel"]
            config = channel_config["config"]
            
            if channel in self.notification_handlers:
                handler = self.notification_handlers[channel]
                try:
                    success = await handler.send_notification(alert, config)
                    if success:
                        logger.info(
                            "Alert notification sent",
                            alert_id=alert.id,
                            channel=channel.value
                        )
                    else:
                        logger.error(
                            "Alert notification failed",
                            alert_id=alert.id,
                            channel=channel.value
                        )
                except Exception as e:
                    logger.error(
                        "Alert notification error",
                        alert_id=alert.id,
                        channel=channel.value,
                        exc_info=e
                    )
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            
            logger.info(
                "Alert acknowledged",
                alert_id=alert_id,
                acknowledged_by=acknowledged_by
            )
            return True
        
        return False
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            
            # Move to history and remove from active
            del self.active_alerts[alert_id]
            
            logger.info("Alert resolved", alert_id=alert_id)
            return True
        
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        
        severity_counts = {}
        for alert in self.alert_history:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        service_counts = {}
        for alert in self.alert_history:
            service = alert.service
            service_counts[service] = service_counts.get(service, 0) + 1
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "resolved_alerts": total_alerts - active_alerts,
            "severity_distribution": severity_counts,
            "service_distribution": service_counts,
            "alert_rules": len(self.alert_rules),
            "notification_channels": sum(len(channels) for channels in self.notification_channels.values())
        }


# Global alert manager instance
alert_manager = AlertManager()


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    return alert_manager


async def initialize_default_alert_rules():
    """Initialize default alert rules for system monitoring."""
    default_rules = [
        AlertRule(
            name="high_cpu_usage",
            description="CPU usage is above 80%",
            service="system",
            metric="cpu_usage_percent",
            condition="greater_than",
            threshold=80.0,
            severity=AlertSeverity.HIGH,
            duration=120,
            cooldown=300
        ),
        AlertRule(
            name="high_memory_usage",
            description="Memory usage is above 90%",
            service="system",
            metric="memory_usage_percent",
            condition="greater_than",
            threshold=90.0,
            severity=AlertSeverity.CRITICAL,
            duration=60,
            cooldown=300
        ),
        AlertRule(
            name="slow_response_time",
            description="Average response time is above 2 seconds",
            service="api",
            metric="response_time_ms",
            condition="greater_than",
            threshold=2000.0,
            severity=AlertSeverity.MEDIUM,
            duration=180,
            cooldown=600
        ),
        AlertRule(
            name="high_error_rate",
            description="Error rate is above 5%",
            service="api",
            metric="error_rate_percent",
            condition="greater_than",
            threshold=5.0,
            severity=AlertSeverity.HIGH,
            duration=120,
            cooldown=300
        )
    ]
    
    for rule in default_rules:
        alert_manager.add_alert_rule(rule)
    
    # Configure default notification channels
    alert_manager.configure_notification_channel(
        AlertSeverity.CRITICAL,
        NotificationChannel.LOG,
        {}
    )
    
    alert_manager.configure_notification_channel(
        AlertSeverity.HIGH,
        NotificationChannel.LOG,
        {}
    )
    
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    logger.info("Default alert rules initialized")