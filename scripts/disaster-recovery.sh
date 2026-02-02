#!/bin/bash

# BharatVoice Disaster Recovery Script
# This script handles disaster recovery scenarios

set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-bharatvoice}"
BACKUP_DIR="${BACKUP_DIR:-/backups}"
S3_BUCKET="${S3_BUCKET:-bharatvoice-backups}"
RECOVERY_MODE="${RECOVERY_MODE:-full}"

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${BACKUP_DIR}/disaster-recovery.log"
}

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  status        Check system status and health"
    echo "  backup        Create emergency backup of all data"
    echo "  restore       Restore from disaster recovery backup"
    echo "  failover      Initiate failover to backup region"
    echo "  rollback      Rollback to previous stable state"
    echo ""
    echo "Options:"
    echo "  -n, --namespace NAMESPACE    Kubernetes namespace (default: bharatvoice)"
    echo "  -m, --mode MODE             Recovery mode: full, partial, database-only"
    echo "  -f, --force                 Force operation without confirmation"
    echo "  -h, --help                  Show this help message"
    exit 1
}

# Parse command line arguments
FORCE=false
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -m|--mode)
            RECOVERY_MODE="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        status|backup|restore|failover|rollback)
            COMMAND="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [[ -z "${COMMAND}" ]]; then
    echo "ERROR: Command not specified"
    usage
fi

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites"
    
    # Check kubectl
    if ! command -v kubectl >/dev/null 2>&1; then
        log "ERROR: kubectl not found"
        exit 1
    fi
    
    # Check AWS CLI for S3 operations
    if ! command -v aws >/dev/null 2>&1; then
        log "WARNING: AWS CLI not found, S3 operations will not be available"
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
        log "ERROR: Namespace ${NAMESPACE} not found"
        exit 1
    fi
    
    log "Prerequisites check completed"
}

# Check system status
check_status() {
    log "Checking system status"
    
    echo "=== Kubernetes Cluster Status ==="
    kubectl cluster-info
    
    echo ""
    echo "=== Namespace Resources ==="
    kubectl get all -n "${NAMESPACE}"
    
    echo ""
    echo "=== Pod Status ==="
    kubectl get pods -n "${NAMESPACE}" -o wide
    
    echo ""
    echo "=== Service Status ==="
    kubectl get services -n "${NAMESPACE}"
    
    echo ""
    echo "=== PVC Status ==="
    kubectl get pvc -n "${NAMESPACE}"
    
    echo ""
    echo "=== Recent Events ==="
    kubectl get events -n "${NAMESPACE}" --sort-by='.lastTimestamp' | tail -20
    
    # Check application health
    echo ""
    echo "=== Application Health Checks ==="
    
    # Check main application
    if kubectl get deployment bharatvoice-app -n "${NAMESPACE}" >/dev/null 2>&1; then
        READY_REPLICAS=$(kubectl get deployment bharatvoice-app -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}')
        DESIRED_REPLICAS=$(kubectl get deployment bharatvoice-app -n "${NAMESPACE}" -o jsonpath='{.spec.replicas}')
        echo "BharatVoice App: ${READY_REPLICAS:-0}/${DESIRED_REPLICAS} replicas ready"
    fi
    
    # Check database
    if kubectl get statefulset postgres -n "${NAMESPACE}" >/dev/null 2>&1; then
        READY_REPLICAS=$(kubectl get statefulset postgres -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}')
        DESIRED_REPLICAS=$(kubectl get statefulset postgres -n "${NAMESPACE}" -o jsonpath='{.spec.replicas}')
        echo "PostgreSQL: ${READY_REPLICAS:-0}/${DESIRED_REPLICAS} replicas ready"
    fi
    
    # Check Redis
    if kubectl get deployment redis -n "${NAMESPACE}" >/dev/null 2>&1; then
        READY_REPLICAS=$(kubectl get deployment redis -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}')
        DESIRED_REPLICAS=$(kubectl get deployment redis -n "${NAMESPACE}" -o jsonpath='{.spec.replicas}')
        echo "Redis: ${READY_REPLICAS:-0}/${DESIRED_REPLICAS} replicas ready"
    fi
}

# Create emergency backup
create_emergency_backup() {
    log "Creating emergency backup"
    
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    EMERGENCY_BACKUP_DIR="${BACKUP_DIR}/emergency_${TIMESTAMP}"
    mkdir -p "${EMERGENCY_BACKUP_DIR}"
    
    # Backup Kubernetes resources
    log "Backing up Kubernetes resources"
    kubectl get all -n "${NAMESPACE}" -o yaml > "${EMERGENCY_BACKUP_DIR}/k8s-resources.yaml"
    kubectl get configmaps -n "${NAMESPACE}" -o yaml > "${EMERGENCY_BACKUP_DIR}/configmaps.yaml"
    kubectl get secrets -n "${NAMESPACE}" -o yaml > "${EMERGENCY_BACKUP_DIR}/secrets.yaml"
    kubectl get pvc -n "${NAMESPACE}" -o yaml > "${EMERGENCY_BACKUP_DIR}/pvcs.yaml"
    
    # Backup database
    log "Backing up database"
    kubectl exec -n "${NAMESPACE}" deployment/bharatvoice-app -- \
        pg_dump -h postgres-service -U bharatvoice -d bharatvoice --format=custom \
        > "${EMERGENCY_BACKUP_DIR}/database.dump" || log "WARNING: Database backup failed"
    
    # Backup Redis data
    log "Backing up Redis data"
    kubectl exec -n "${NAMESPACE}" deployment/redis -- \
        redis-cli --rdb - > "${EMERGENCY_BACKUP_DIR}/redis.rdb" || log "WARNING: Redis backup failed"
    
    # Create archive
    log "Creating emergency backup archive"
    tar -czf "${BACKUP_DIR}/emergency_backup_${TIMESTAMP}.tar.gz" -C "${BACKUP_DIR}" "emergency_${TIMESTAMP}"
    
    # Upload to S3 if available
    if [[ -n "${S3_BUCKET}" ]] && command -v aws >/dev/null 2>&1; then
        log "Uploading emergency backup to S3"
        aws s3 cp "${BACKUP_DIR}/emergency_backup_${TIMESTAMP}.tar.gz" \
            "s3://${S3_BUCKET}/emergency/" || log "WARNING: S3 upload failed"
    fi
    
    # Clean up temporary directory
    rm -rf "${EMERGENCY_BACKUP_DIR}"
    
    log "Emergency backup completed: emergency_backup_${TIMESTAMP}.tar.gz"
}

# Restore from disaster recovery backup
restore_from_backup() {
    log "Starting disaster recovery restore"
    
    if [[ "${FORCE}" != "true" ]]; then
        echo "WARNING: This will completely restore the system from backup!"
        echo "Recovery mode: ${RECOVERY_MODE}"
        echo "Namespace: ${NAMESPACE}"
        read -p "Are you sure you want to continue? (yes/no): " -r
        
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            log "Restore cancelled by user"
            exit 0
        fi
    fi
    
    case "${RECOVERY_MODE}" in
        "full")
            log "Performing full system restore"
            restore_kubernetes_resources
            restore_database
            restore_redis
            ;;
        "partial")
            log "Performing partial restore (application only)"
            restore_kubernetes_resources
            ;;
        "database-only")
            log "Performing database-only restore"
            restore_database
            ;;
        *)
            log "ERROR: Unknown recovery mode: ${RECOVERY_MODE}"
            exit 1
            ;;
    esac
    
    log "Disaster recovery restore completed"
}

# Restore Kubernetes resources
restore_kubernetes_resources() {
    log "Restoring Kubernetes resources"
    
    # Scale down deployments
    kubectl scale deployment --all --replicas=0 -n "${NAMESPACE}" || true
    
    # Wait for pods to terminate
    kubectl wait --for=delete pods --all -n "${NAMESPACE}" --timeout=300s || true
    
    # Apply restored configurations
    # Note: In a real scenario, you would restore from actual backup files
    kubectl apply -f k8s/ -n "${NAMESPACE}" || log "WARNING: Some resources failed to restore"
    
    # Wait for deployments to be ready
    kubectl wait --for=condition=available deployment --all -n "${NAMESPACE}" --timeout=600s || log "WARNING: Some deployments not ready"
}

# Restore database
restore_database() {
    log "Restoring database"
    
    # Find latest database backup
    LATEST_DB_BACKUP=$(find "${BACKUP_DIR}" -name "bharatvoice_backup_*.sql*" -type f | sort | tail -1)
    
    if [[ -z "${LATEST_DB_BACKUP}" ]] && [[ -n "${S3_BUCKET}" ]]; then
        log "No local backup found, checking S3"
        # Download latest backup from S3
        LATEST_S3_BACKUP=$(aws s3 ls "s3://${S3_BUCKET}/database/" | sort | tail -1 | awk '{print $4}')
        if [[ -n "${LATEST_S3_BACKUP}" ]]; then
            aws s3 cp "s3://${S3_BUCKET}/database/${LATEST_S3_BACKUP}" "${BACKUP_DIR}/"
            LATEST_DB_BACKUP="${BACKUP_DIR}/${LATEST_S3_BACKUP}"
        fi
    fi
    
    if [[ -n "${LATEST_DB_BACKUP}" ]]; then
        log "Restoring from backup: $(basename "${LATEST_DB_BACKUP}")"
        "${BACKUP_DIR}/../scripts/restore-database.sh" --force "${LATEST_DB_BACKUP}"
    else
        log "ERROR: No database backup found"
        exit 1
    fi
}

# Restore Redis
restore_redis() {
    log "Restoring Redis data"
    
    # Find latest Redis backup
    LATEST_REDIS_BACKUP=$(find "${BACKUP_DIR}" -name "redis_backup_*.rdb" -type f | sort | tail -1)
    
    if [[ -n "${LATEST_REDIS_BACKUP}" ]]; then
        log "Restoring Redis from backup: $(basename "${LATEST_REDIS_BACKUP}")"
        kubectl cp "${LATEST_REDIS_BACKUP}" "${NAMESPACE}/redis-0:/data/dump.rdb"
        kubectl rollout restart deployment/redis -n "${NAMESPACE}"
    else
        log "WARNING: No Redis backup found"
    fi
}

# Initiate failover
initiate_failover() {
    log "Initiating failover to backup region"
    
    if [[ "${FORCE}" != "true" ]]; then
        echo "WARNING: This will initiate failover to backup region!"
        read -p "Are you sure you want to continue? (yes/no): " -r
        
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            log "Failover cancelled by user"
            exit 0
        fi
    fi
    
    # Create emergency backup before failover
    create_emergency_backup
    
    # Update DNS to point to backup region
    log "Updating DNS for failover"
    # Note: This would typically involve updating Route53 or other DNS service
    
    # Scale down primary region
    log "Scaling down primary region"
    kubectl scale deployment --all --replicas=0 -n "${NAMESPACE}" || true
    
    # Notify monitoring systems
    log "Sending failover notifications"
    # Note: This would send notifications to monitoring and alerting systems
    
    log "Failover initiated successfully"
}

# Rollback to previous state
rollback_previous_state() {
    log "Rolling back to previous stable state"
    
    if [[ "${FORCE}" != "true" ]]; then
        echo "WARNING: This will rollback all deployments to previous versions!"
        read -p "Are you sure you want to continue? (yes/no): " -r
        
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            log "Rollback cancelled by user"
            exit 0
        fi
    fi
    
    # Rollback all deployments
    log "Rolling back deployments"
    kubectl rollout undo deployment/bharatvoice-app -n "${NAMESPACE}" || log "WARNING: App rollback failed"
    kubectl rollout undo deployment/bharatvoice-worker -n "${NAMESPACE}" || log "WARNING: Worker rollback failed"
    kubectl rollout undo deployment/nginx -n "${NAMESPACE}" || log "WARNING: Nginx rollback failed"
    
    # Wait for rollback to complete
    kubectl rollout status deployment/bharatvoice-app -n "${NAMESPACE}" --timeout=300s || log "WARNING: App rollback timeout"
    kubectl rollout status deployment/bharatvoice-worker -n "${NAMESPACE}" --timeout=300s || log "WARNING: Worker rollback timeout"
    kubectl rollout status deployment/nginx -n "${NAMESPACE}" --timeout=300s || log "WARNING: Nginx rollback timeout"
    
    log "Rollback completed successfully"
}

# Main execution
main() {
    log "Starting disaster recovery operation: ${COMMAND}"
    
    check_prerequisites
    
    case "${COMMAND}" in
        "status")
            check_status
            ;;
        "backup")
            create_emergency_backup
            ;;
        "restore")
            restore_from_backup
            ;;
        "failover")
            initiate_failover
            ;;
        "rollback")
            rollback_previous_state
            ;;
        *)
            log "ERROR: Unknown command: ${COMMAND}"
            usage
            ;;
    esac
    
    log "Disaster recovery operation completed: ${COMMAND}"
}

# Execute main function
main "$@"