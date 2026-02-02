"""
Alerting API endpoints for BharatVoice Assistant.

This module provides REST API endpoints for managing alerts, alert rules,
and notification channels.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
import structlog

from bharatvoice.utils.alerting import (
    get_alert_manager,
    AlertManager,
    Alert,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    NotificationChannel
)


logger = structlog.get_logger(__name__)
router = APIRouter()


class AlertRuleCreate(BaseModel):
    """Alert rule creation model."""
    name: str
    description: str
    service: str
    metric: str
    condition: str
    threshold: float
    severity: AlertSeverity
    duration: int = 60
    cooldown: int = 300
    enabled: bool = True


class AlertRuleUpdate(BaseModel):
    """Alert rule update model."""
    description: Optional[str] = None
    condition: Optional[str] = None
    threshold: Optional[float] = None
    severity: Optional[AlertSeverity] = None
    duration: Optional[int] = None
    cooldown: Optional[int] = None
    enabled: Optional[bool] = None


class NotificationChannelConfig(BaseModel):
    """Notification channel configuration model."""
    severity: AlertSeverity
    channel: NotificationChannel
    config: Dict[str, Any]


class AlertAcknowledge(BaseModel):
    """Alert acknowledgment model."""
    acknowledged_by: str
    notes: Optional[str] = None


@router.get("/alerts")
async def list_alerts(
    status: Optional[AlertStatus] = None,
    severity: Optional[AlertSeverity] = None,
    service: Optional[str] = None,
    limit: int = 100,
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """
    List alerts with optional filtering.
    
    Args:
        status: Filter by alert status
        severity: Filter by alert severity
        service: Filter by service name
        limit: Maximum number of alerts to return
        alert_manager: Alert manager instance
        
    Returns:
        List of alerts matching the criteria
    """
    try:
        # Get active alerts and history
        active_alerts = alert_manager.get_active_alerts()
        alert_history = alert_manager.get_alert_history(limit)
        
        # Combine and filter alerts
        all_alerts = active_alerts + [a for a in alert_history if a not in active_alerts]
        
        filtered_alerts = []
        for alert in all_alerts:
            # Apply filters
            if status and alert.status != status:
                continue
            if severity and alert.severity != severity:
                continue
            if service and alert.service != service:
                continue
            
            filtered_alerts.append(alert.to_dict())
        
        # Sort by creation time (newest first) and limit
        filtered_alerts.sort(key=lambda x: x['created_at'], reverse=True)
        filtered_alerts = filtered_alerts[:limit]
        
        return {
            "alerts": filtered_alerts,
            "total": len(filtered_alerts),
            "filters": {
                "status": status.value if status else None,
                "severity": severity.value if severity else None,
                "service": service
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error("Failed to list alerts", exc_info=e)
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")


@router.get("/alerts/{alert_id}")
async def get_alert(
    alert_id: str,
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """
    Get a specific alert by ID.
    
    Args:
        alert_id: Alert identifier
        alert_manager: Alert manager instance
        
    Returns:
        Alert details
    """
    try:
        # Check active alerts first
        if alert_id in alert_manager.active_alerts:
            alert = alert_manager.active_alerts[alert_id]
            return alert.to_dict()
        
        # Check alert history
        for alert in alert_manager.alert_history:
            if alert.id == alert_id:
                return alert.to_dict()
        
        raise HTTPException(status_code=404, detail="Alert not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get alert", alert_id=alert_id, exc_info=e)
        raise HTTPException(status_code=500, detail="Failed to retrieve alert")


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    acknowledge_data: AlertAcknowledge,
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """
    Acknowledge an alert.
    
    Args:
        alert_id: Alert identifier
        acknowledge_data: Acknowledgment data
        alert_manager: Alert manager instance
        
    Returns:
        Acknowledgment confirmation
    """
    try:
        success = await alert_manager.acknowledge_alert(
            alert_id, 
            acknowledge_data.acknowledged_by
        )
        
        if success:
            return {
                "message": "Alert acknowledged successfully",
                "alert_id": alert_id,
                "acknowledged_by": acknowledge_data.acknowledged_by,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Alert not found or already acknowledged")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to acknowledge alert", alert_id=alert_id, exc_info=e)
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """
    Resolve an alert.
    
    Args:
        alert_id: Alert identifier
        alert_manager: Alert manager instance
        
    Returns:
        Resolution confirmation
    """
    try:
        success = await alert_manager.resolve_alert(alert_id)
        
        if success:
            return {
                "message": "Alert resolved successfully",
                "alert_id": alert_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resolve alert", alert_id=alert_id, exc_info=e)
        raise HTTPException(status_code=500, detail="Failed to resolve alert")


@router.get("/alert-rules")
async def list_alert_rules(
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """
    List all alert rules.
    
    Args:
        alert_manager: Alert manager instance
        
    Returns:
        List of alert rules
    """
    try:
        rules = []
        for rule_name, rule in alert_manager.alert_rules.items():
            rules.append({
                "name": rule.name,
                "description": rule.description,
                "service": rule.service,
                "metric": rule.metric,
                "condition": rule.condition,
                "threshold": rule.threshold,
                "severity": rule.severity.value,
                "duration": rule.duration,
                "cooldown": rule.cooldown,
                "enabled": rule.enabled,
                "last_triggered": rule.last_triggered
            })
        
        return {
            "alert_rules": rules,
            "total": len(rules),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error("Failed to list alert rules", exc_info=e)
        raise HTTPException(status_code=500, detail="Failed to retrieve alert rules")


@router.post("/alert-rules")
async def create_alert_rule(
    rule_data: AlertRuleCreate,
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """
    Create a new alert rule.
    
    Args:
        rule_data: Alert rule data
        alert_manager: Alert manager instance
        
    Returns:
        Created alert rule confirmation
    """
    try:
        # Check if rule already exists
        if rule_data.name in alert_manager.alert_rules:
            raise HTTPException(status_code=400, detail="Alert rule already exists")
        
        # Create alert rule
        rule = AlertRule(
            name=rule_data.name,
            description=rule_data.description,
            service=rule_data.service,
            metric=rule_data.metric,
            condition=rule_data.condition,
            threshold=rule_data.threshold,
            severity=rule_data.severity,
            duration=rule_data.duration,
            cooldown=rule_data.cooldown,
            enabled=rule_data.enabled
        )
        
        alert_manager.add_alert_rule(rule)
        
        return {
            "message": "Alert rule created successfully",
            "rule_name": rule_data.name,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create alert rule", exc_info=e)
        raise HTTPException(status_code=500, detail="Failed to create alert rule")


@router.put("/alert-rules/{rule_name}")
async def update_alert_rule(
    rule_name: str,
    rule_update: AlertRuleUpdate,
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """
    Update an existing alert rule.
    
    Args:
        rule_name: Alert rule name
        rule_update: Alert rule update data
        alert_manager: Alert manager instance
        
    Returns:
        Update confirmation
    """
    try:
        if rule_name not in alert_manager.alert_rules:
            raise HTTPException(status_code=404, detail="Alert rule not found")
        
        rule = alert_manager.alert_rules[rule_name]
        
        # Update rule properties
        if rule_update.description is not None:
            rule.description = rule_update.description
        if rule_update.condition is not None:
            rule.condition = rule_update.condition
        if rule_update.threshold is not None:
            rule.threshold = rule_update.threshold
        if rule_update.severity is not None:
            rule.severity = rule_update.severity
        if rule_update.duration is not None:
            rule.duration = rule_update.duration
        if rule_update.cooldown is not None:
            rule.cooldown = rule_update.cooldown
        if rule_update.enabled is not None:
            rule.enabled = rule_update.enabled
        
        return {
            "message": "Alert rule updated successfully",
            "rule_name": rule_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update alert rule", rule_name=rule_name, exc_info=e)
        raise HTTPException(status_code=500, detail="Failed to update alert rule")


@router.delete("/alert-rules/{rule_name}")
async def delete_alert_rule(
    rule_name: str,
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """
    Delete an alert rule.
    
    Args:
        rule_name: Alert rule name
        alert_manager: Alert manager instance
        
    Returns:
        Deletion confirmation
    """
    try:
        if rule_name not in alert_manager.alert_rules:
            raise HTTPException(status_code=404, detail="Alert rule not found")
        
        alert_manager.remove_alert_rule(rule_name)
        
        return {
            "message": "Alert rule deleted successfully",
            "rule_name": rule_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete alert rule", rule_name=rule_name, exc_info=e)
        raise HTTPException(status_code=500, detail="Failed to delete alert rule")


@router.post("/notification-channels")
async def configure_notification_channel(
    channel_config: NotificationChannelConfig,
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """
    Configure a notification channel.
    
    Args:
        channel_config: Notification channel configuration
        alert_manager: Alert manager instance
        
    Returns:
        Configuration confirmation
    """
    try:
        alert_manager.configure_notification_channel(
            channel_config.severity,
            channel_config.channel,
            channel_config.config
        )
        
        return {
            "message": "Notification channel configured successfully",
            "severity": channel_config.severity.value,
            "channel": channel_config.channel.value,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error("Failed to configure notification channel", exc_info=e)
        raise HTTPException(status_code=500, detail="Failed to configure notification channel")


@router.get("/statistics")
async def get_alert_statistics(
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """
    Get alert statistics and metrics.
    
    Args:
        alert_manager: Alert manager instance
        
    Returns:
        Alert statistics
    """
    try:
        stats = alert_manager.get_alert_statistics()
        stats["timestamp"] = datetime.utcnow().isoformat()
        return stats
    
    except Exception as e:
        logger.error("Failed to get alert statistics", exc_info=e)
        raise HTTPException(status_code=500, detail="Failed to retrieve alert statistics")


@router.post("/test-alert")
async def trigger_test_alert(
    background_tasks: BackgroundTasks,
    alert_manager: AlertManager = Depends(get_alert_manager)
):
    """
    Trigger a test alert for testing notification channels.
    
    Args:
        background_tasks: Background tasks
        alert_manager: Alert manager instance
        
    Returns:
        Test alert confirmation
    """
    try:
        # Create a test alert rule
        test_rule = AlertRule(
            name="test_alert",
            description="Test alert for notification testing",
            service="test",
            metric="test_metric",
            condition="greater_than",
            threshold=0.0,
            severity=AlertSeverity.INFO,
            duration=0,
            cooldown=0,
            enabled=True
        )
        
        # Trigger the test alert
        background_tasks.add_task(
            alert_manager._trigger_alert,
            test_rule,
            1.0
        )
        
        return {
            "message": "Test alert triggered",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error("Failed to trigger test alert", exc_info=e)
        raise HTTPException(status_code=500, detail="Failed to trigger test alert")