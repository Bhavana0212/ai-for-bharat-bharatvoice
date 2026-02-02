#!/bin/bash

# BharatVoice Database Backup Script
# This script creates automated backups of the PostgreSQL database

set -euo pipefail

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/backups/database}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
POSTGRES_HOST="${POSTGRES_HOST:-postgres-service}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-bharatvoice}"
POSTGRES_USER="${POSTGRES_USER:-bharatvoice}"
S3_BUCKET="${S3_BUCKET:-bharatvoice-backups}"
ENCRYPTION_KEY_FILE="${ENCRYPTION_KEY_FILE:-/secrets/backup-encryption-key}"

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${BACKUP_DIR}/backup.log"
}

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Generate backup filename with timestamp
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
BACKUP_FILE="bharatvoice_backup_${TIMESTAMP}.sql"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_FILE}"

log "Starting database backup: ${BACKUP_FILE}"

# Create database dump
if pg_dump -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
    --verbose --clean --if-exists --create --format=custom --compress=9 \
    --file="${BACKUP_PATH}"; then
    log "Database dump completed successfully"
else
    log "ERROR: Database dump failed"
    exit 1
fi

# Encrypt backup if encryption key is available
if [[ -f "${ENCRYPTION_KEY_FILE}" ]]; then
    log "Encrypting backup file"
    if gpg --cipher-algo AES256 --compress-algo 1 --s2k-mode 3 \
        --s2k-digest-algo SHA512 --s2k-count 65536 --force-mdc \
        --quiet --batch --yes --passphrase-file "${ENCRYPTION_KEY_FILE}" \
        --symmetric --output "${BACKUP_PATH}.gpg" "${BACKUP_PATH}"; then
        
        # Remove unencrypted backup
        rm "${BACKUP_PATH}"
        BACKUP_PATH="${BACKUP_PATH}.gpg"
        log "Backup encrypted successfully"
    else
        log "WARNING: Backup encryption failed, keeping unencrypted backup"
    fi
fi

# Calculate backup size and checksum
BACKUP_SIZE=$(du -h "${BACKUP_PATH}" | cut -f1)
BACKUP_CHECKSUM=$(sha256sum "${BACKUP_PATH}" | cut -d' ' -f1)

log "Backup size: ${BACKUP_SIZE}"
log "Backup checksum: ${BACKUP_CHECKSUM}"

# Upload to S3 if configured
if [[ -n "${S3_BUCKET}" ]] && command -v aws >/dev/null 2>&1; then
    log "Uploading backup to S3: s3://${S3_BUCKET}/database/"
    
    if aws s3 cp "${BACKUP_PATH}" "s3://${S3_BUCKET}/database/" \
        --storage-class STANDARD_IA \
        --metadata "checksum=${BACKUP_CHECKSUM},size=${BACKUP_SIZE}"; then
        log "Backup uploaded to S3 successfully"
        
        # Remove local backup after successful upload
        rm "${BACKUP_PATH}"
        log "Local backup removed after S3 upload"
    else
        log "WARNING: S3 upload failed, keeping local backup"
    fi
fi

# Clean up old backups
log "Cleaning up backups older than ${RETENTION_DAYS} days"
find "${BACKUP_DIR}" -name "bharatvoice_backup_*.sql*" -type f -mtime +${RETENTION_DAYS} -delete

# Clean up old S3 backups if configured
if [[ -n "${S3_BUCKET}" ]] && command -v aws >/dev/null 2>&1; then
    CUTOFF_DATE=$(date -d "${RETENTION_DAYS} days ago" '+%Y-%m-%d')
    aws s3 ls "s3://${S3_BUCKET}/database/" | while read -r line; do
        BACKUP_DATE=$(echo "$line" | awk '{print $1}')
        BACKUP_NAME=$(echo "$line" | awk '{print $4}')
        
        if [[ "${BACKUP_DATE}" < "${CUTOFF_DATE}" ]]; then
            log "Deleting old S3 backup: ${BACKUP_NAME}"
            aws s3 rm "s3://${S3_BUCKET}/database/${BACKUP_NAME}"
        fi
    done
fi

# Create backup metadata
cat > "${BACKUP_DIR}/backup_${TIMESTAMP}.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "filename": "$(basename "${BACKUP_PATH}")",
    "size": "${BACKUP_SIZE}",
    "checksum": "${BACKUP_CHECKSUM}",
    "database": "${POSTGRES_DB}",
    "host": "${POSTGRES_HOST}",
    "encrypted": $([ -f "${ENCRYPTION_KEY_FILE}" ] && echo "true" || echo "false"),
    "s3_uploaded": $([ -n "${S3_BUCKET}" ] && echo "true" || echo "false")
}
EOF

log "Database backup completed successfully: ${BACKUP_FILE}"

# Send notification if webhook is configured
if [[ -n "${BACKUP_WEBHOOK_URL:-}" ]]; then
    curl -X POST "${BACKUP_WEBHOOK_URL}" \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"Database backup completed: ${BACKUP_FILE}\", \"status\": \"success\"}" \
        || log "WARNING: Failed to send backup notification"
fi