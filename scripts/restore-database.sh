#!/bin/bash

# BharatVoice Database Restore Script
# This script restores the PostgreSQL database from backup

set -euo pipefail

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/backups/database}"
POSTGRES_HOST="${POSTGRES_HOST:-postgres-service}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-bharatvoice}"
POSTGRES_USER="${POSTGRES_USER:-bharatvoice}"
S3_BUCKET="${S3_BUCKET:-bharatvoice-backups}"
ENCRYPTION_KEY_FILE="${ENCRYPTION_KEY_FILE:-/secrets/backup-encryption-key}"

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${BACKUP_DIR}/restore.log"
}

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS] BACKUP_FILE"
    echo "Options:"
    echo "  -s, --from-s3     Download backup from S3"
    echo "  -l, --list        List available backups"
    echo "  -v, --verify      Verify backup integrity before restore"
    echo "  -f, --force       Force restore without confirmation"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 bharatvoice_backup_20240201_120000.sql.gpg"
    echo "  $0 --from-s3 bharatvoice_backup_20240201_120000.sql.gpg"
    echo "  $0 --list"
    exit 1
}

# Parse command line arguments
FROM_S3=false
LIST_BACKUPS=false
VERIFY_BACKUP=false
FORCE_RESTORE=false
BACKUP_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--from-s3)
            FROM_S3=true
            shift
            ;;
        -l|--list)
            LIST_BACKUPS=true
            shift
            ;;
        -v|--verify)
            VERIFY_BACKUP=true
            shift
            ;;
        -f|--force)
            FORCE_RESTORE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            BACKUP_FILE="$1"
            shift
            ;;
    esac
done

# List available backups
if [[ "${LIST_BACKUPS}" == "true" ]]; then
    echo "Local backups:"
    ls -la "${BACKUP_DIR}"/bharatvoice_backup_*.sql* 2>/dev/null || echo "No local backups found"
    
    if [[ -n "${S3_BUCKET}" ]] && command -v aws >/dev/null 2>&1; then
        echo ""
        echo "S3 backups:"
        aws s3 ls "s3://${S3_BUCKET}/database/" --human-readable --summarize
    fi
    exit 0
fi

# Validate backup file argument
if [[ -z "${BACKUP_FILE}" ]]; then
    echo "ERROR: Backup file not specified"
    usage
fi

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Download from S3 if requested
if [[ "${FROM_S3}" == "true" ]]; then
    if [[ -z "${S3_BUCKET}" ]] || ! command -v aws >/dev/null 2>&1; then
        log "ERROR: S3 not configured or AWS CLI not available"
        exit 1
    fi
    
    log "Downloading backup from S3: s3://${S3_BUCKET}/database/${BACKUP_FILE}"
    
    if aws s3 cp "s3://${S3_BUCKET}/database/${BACKUP_FILE}" "${BACKUP_DIR}/${BACKUP_FILE}"; then
        log "Backup downloaded from S3 successfully"
    else
        log "ERROR: Failed to download backup from S3"
        exit 1
    fi
fi

# Check if backup file exists
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_FILE}"
if [[ ! -f "${BACKUP_PATH}" ]]; then
    log "ERROR: Backup file not found: ${BACKUP_PATH}"
    exit 1
fi

# Decrypt backup if encrypted
RESTORE_FILE="${BACKUP_PATH}"
if [[ "${BACKUP_FILE}" == *.gpg ]]; then
    if [[ ! -f "${ENCRYPTION_KEY_FILE}" ]]; then
        log "ERROR: Encryption key file not found: ${ENCRYPTION_KEY_FILE}"
        exit 1
    fi
    
    log "Decrypting backup file"
    DECRYPTED_FILE="${BACKUP_PATH%.gpg}"
    
    if gpg --quiet --batch --yes --passphrase-file "${ENCRYPTION_KEY_FILE}" \
        --decrypt --output "${DECRYPTED_FILE}" "${BACKUP_PATH}"; then
        log "Backup decrypted successfully"
        RESTORE_FILE="${DECRYPTED_FILE}"
    else
        log "ERROR: Failed to decrypt backup file"
        exit 1
    fi
fi

# Verify backup integrity
if [[ "${VERIFY_BACKUP}" == "true" ]]; then
    log "Verifying backup integrity"
    
    if pg_restore --list "${RESTORE_FILE}" >/dev/null 2>&1; then
        log "Backup integrity verified successfully"
    else
        log "ERROR: Backup integrity check failed"
        exit 1
    fi
fi

# Get backup information
BACKUP_SIZE=$(du -h "${RESTORE_FILE}" | cut -f1)
log "Backup file size: ${BACKUP_SIZE}"

# Confirmation prompt
if [[ "${FORCE_RESTORE}" != "true" ]]; then
    echo ""
    echo "WARNING: This will completely replace the current database!"
    echo "Database: ${POSTGRES_DB}"
    echo "Host: ${POSTGRES_HOST}"
    echo "Backup file: ${BACKUP_FILE}"
    echo "Backup size: ${BACKUP_SIZE}"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " -r
    
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log "Restore cancelled by user"
        exit 0
    fi
fi

log "Starting database restore from: ${BACKUP_FILE}"

# Create a pre-restore backup
PRE_RESTORE_BACKUP="${BACKUP_DIR}/pre_restore_$(date '+%Y%m%d_%H%M%S').sql"
log "Creating pre-restore backup: $(basename "${PRE_RESTORE_BACKUP}")"

if pg_dump -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
    --format=custom --compress=9 --file="${PRE_RESTORE_BACKUP}"; then
    log "Pre-restore backup created successfully"
else
    log "WARNING: Failed to create pre-restore backup"
fi

# Terminate active connections to the database
log "Terminating active database connections"
psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d postgres -c \
    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${POSTGRES_DB}' AND pid <> pg_backend_pid();" \
    || log "WARNING: Failed to terminate some connections"

# Restore database
log "Restoring database from backup"

if pg_restore -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" \
    --dbname="${POSTGRES_DB}" --verbose --clean --if-exists --create \
    --single-transaction "${RESTORE_FILE}"; then
    log "Database restore completed successfully"
else
    log "ERROR: Database restore failed"
    
    # Attempt to restore from pre-restore backup
    if [[ -f "${PRE_RESTORE_BACKUP}" ]]; then
        log "Attempting to restore from pre-restore backup"
        pg_restore -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" \
            --dbname="${POSTGRES_DB}" --clean --if-exists --create \
            --single-transaction "${PRE_RESTORE_BACKUP}" \
            && log "Rollback to pre-restore state completed" \
            || log "ERROR: Rollback failed"
    fi
    
    exit 1
fi

# Clean up decrypted file if it was created
if [[ "${RESTORE_FILE}" != "${BACKUP_PATH}" ]]; then
    rm -f "${RESTORE_FILE}"
    log "Cleaned up decrypted backup file"
fi

# Verify restore
log "Verifying database restore"
if psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
    -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" >/dev/null 2>&1; then
    log "Database restore verification successful"
else
    log "WARNING: Database restore verification failed"
fi

# Update database statistics
log "Updating database statistics"
psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
    -c "ANALYZE;" || log "WARNING: Failed to update database statistics"

log "Database restore completed: ${BACKUP_FILE}"

# Send notification if webhook is configured
if [[ -n "${RESTORE_WEBHOOK_URL:-}" ]]; then
    curl -X POST "${RESTORE_WEBHOOK_URL}" \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"Database restore completed: ${BACKUP_FILE}\", \"status\": \"success\"}" \
        || log "WARNING: Failed to send restore notification"
fi