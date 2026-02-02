# BharatVoice Assistant - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the BharatVoice AI-powered multilingual voice assistant. The system is designed for production deployment with high availability, scalability, and security.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Pre-deployment Setup](#pre-deployment-setup)
3. [Environment Configuration](#environment-configuration)
4. [Database Setup](#database-setup)
5. [Service Deployment](#service-deployment)
6. [Container Deployment](#container-deployment)
7. [Kubernetes Deployment](#kubernetes-deployment)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Security Configuration](#security-configuration)
10. [Performance Tuning](#performance-tuning)
11. [Backup and Recovery](#backup-and-recovery)
12. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **CPU**: 4 cores (8 recommended)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 50GB SSD (100GB recommended)
- **Network**: 100Mbps (1Gbps recommended)
- **OS**: Ubuntu 20.04 LTS or CentOS 8

### Recommended Production Requirements

- **CPU**: 8+ cores with AVX2 support
- **RAM**: 32GB+ 
- **Storage**: 200GB+ NVMe SSD
- **Network**: 1Gbps+ with low latency
- **OS**: Ubuntu 22.04 LTS

### External Dependencies

- **PostgreSQL**: 13+ (for user data and conversation history)
- **Redis**: 6+ (for caching and session management)
- **Python**: 3.9+ (with pip and virtualenv)
- **Docker**: 20.10+ (for containerized deployment)
- **Kubernetes**: 1.21+ (for orchestrated deployment)

## Pre-deployment Setup

### 1. System Preparation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y curl wget git build-essential python3-dev python3-pip python3-venv

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2. Clone Repository

```bash
git clone https://github.com/bharatvoice/assistant.git
cd assistant
```

### 3. Create Application User

```bash
sudo useradd -m -s /bin/bash bharatvoice
sudo usermod -aG docker bharatvoice
sudo chown -R bharatvoice:bharatvoice /opt/bharatvoice
```

## Environment Configuration

### 1. Environment Variables

Create `.env` file from template:

```bash
cp .env.example .env
```

Configure the following variables:

```bash
# Application Settings
APP_NAME=bharatvoice-assistant
APP_VERSION=1.0.0
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_TIMEOUT=300

# Database Configuration
DATABASE_URL=postgresql://bharatvoice:password@localhost:5432/bharatvoice
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your_redis_password
REDIS_MAX_CONNECTIONS=100

# Security Configuration
SECRET_KEY=your-super-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Encryption Keys
ENCRYPTION_KEY=your-32-byte-encryption-key
VOICE_DATA_ENCRYPTION_KEY=your-voice-encryption-key

# External API Keys
OPENAI_API_KEY=your-openai-api-key
GOOGLE_CLOUD_API_KEY=your-google-cloud-key
INDIAN_RAILWAYS_API_KEY=your-railways-api-key
WEATHER_API_KEY=your-weather-api-key

# Performance Settings
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=30
CACHE_TTL=3600

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
LOG_FILE_PATH=/var/log/bharatvoice/app.log
```

### 2. SSL/TLS Configuration

```bash
# Generate SSL certificates (for production, use proper CA certificates)
sudo mkdir -p /etc/ssl/bharatvoice
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/bharatvoice/private.key \
    -out /etc/ssl/bharatvoice/certificate.crt

# Set proper permissions
sudo chmod 600 /etc/ssl/bharatvoice/private.key
sudo chmod 644 /etc/ssl/bharatvoice/certificate.crt
```

## Database Setup

### 1. PostgreSQL Installation and Configuration

```bash
# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# Start and enable PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql << EOF
CREATE USER bharatvoice WITH PASSWORD 'your_secure_password';
CREATE DATABASE bharatvoice OWNER bharatvoice;
GRANT ALL PRIVILEGES ON DATABASE bharatvoice TO bharatvoice;
ALTER USER bharatvoice CREATEDB;
\q
EOF
```

### 2. Redis Installation and Configuration

```bash
# Install Redis
sudo apt install -y redis-server

# Configure Redis
sudo tee /etc/redis/redis.conf << EOF
bind 127.0.0.1
port 6379
requirepass your_redis_password
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
EOF

# Start and enable Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

### 3. Database Migration

```bash
# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Run database migrations
alembic upgrade head

# Verify database setup
python -c "from bharatvoice.database.connection import get_database_connection; print('Database connection successful')"
```

## Service Deployment

### 1. Python Virtual Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv /opt/bharatvoice/venv
source /opt/bharatvoice/venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -e .
pip install -e '.[dev]'  # For development dependencies

# Install additional ML dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers
```

### 2. Systemd Service Configuration

Create systemd service file:

```bash
sudo tee /etc/systemd/system/bharatvoice.service << EOF
[Unit]
Description=BharatVoice Assistant API Server
After=network.target postgresql.service redis-server.service
Requires=postgresql.service redis-server.service

[Service]
Type=exec
User=bharatvoice
Group=bharatvoice
WorkingDirectory=/opt/bharatvoice
Environment=PATH=/opt/bharatvoice/venv/bin
EnvironmentFile=/opt/bharatvoice/.env
ExecStart=/opt/bharatvoice/venv/bin/uvicorn bharatvoice.main:app --host 0.0.0.0 --port 8000 --workers 4
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=bharatvoice

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/bharatvoice /var/log/bharatvoice /tmp

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable bharatvoice
sudo systemctl start bharatvoice
```

### 3. Nginx Reverse Proxy

```bash
# Install Nginx
sudo apt install -y nginx

# Configure Nginx
sudo tee /etc/nginx/sites-available/bharatvoice << EOF
upstream bharatvoice_backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/ssl/bharatvoice/certificate.crt;
    ssl_certificate_key /etc/ssl/bharatvoice/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    client_max_body_size 50M;
    client_body_timeout 60s;
    client_header_timeout 60s;

    location / {
        proxy_pass http://bharatvoice_backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /health {
        proxy_pass http://bharatvoice_backend/health;
        access_log off;
    }

    location /metrics {
        proxy_pass http://bharatvoice_backend/metrics;
        allow 127.0.0.1;
        deny all;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/bharatvoice /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Container Deployment

### 1. Docker Build

```bash
# Build Docker image
docker build -t bharatvoice-assistant:latest .

# Tag for registry
docker tag bharatvoice-assistant:latest your-registry.com/bharatvoice-assistant:v1.0.0
```

### 2. Docker Compose Deployment

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  bharatvoice-app:
    image: bharatvoice-assistant:latest
    container_name: bharatvoice-app
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://bharatvoice:${DB_PASSWORD}@postgres:5432/bharatvoice
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads
    networks:
      - bharatvoice-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15
    container_name: bharatvoice-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_DB=bharatvoice
      - POSTGRES_USER=bharatvoice
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    networks:
      - bharatvoice-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U bharatvoice"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: bharatvoice-redis
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - bharatvoice-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: bharatvoice-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/bharatvoice
    depends_on:
      - bharatvoice-app
    networks:
      - bharatvoice-network

volumes:
  postgres_data:
  redis_data:

networks:
  bharatvoice-network:
    driver: bridge
```

Deploy with Docker Compose:

```bash
# Set environment variables
export DB_PASSWORD=your_secure_db_password
export REDIS_PASSWORD=your_secure_redis_password

# Deploy
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps
```

## Kubernetes Deployment

### 1. Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: bharatvoice

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: bharatvoice-config
  namespace: bharatvoice
data:
  APP_NAME: "bharatvoice-assistant"
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  API_WORKERS: "4"
```

### 2. Secrets

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: bharatvoice-secrets
  namespace: bharatvoice
type: Opaque
data:
  DATABASE_URL: <base64-encoded-database-url>
  REDIS_URL: <base64-encoded-redis-url>
  SECRET_KEY: <base64-encoded-secret-key>
  JWT_SECRET_KEY: <base64-encoded-jwt-secret>
  ENCRYPTION_KEY: <base64-encoded-encryption-key>
```

### 3. Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bharatvoice-app
  namespace: bharatvoice
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bharatvoice-app
  template:
    metadata:
      labels:
        app: bharatvoice-app
    spec:
      containers:
      - name: bharatvoice-app
        image: your-registry.com/bharatvoice-assistant:v1.0.0
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: bharatvoice-config
        - secretRef:
            name: bharatvoice-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 4. Service and Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: bharatvoice-service
  namespace: bharatvoice
spec:
  selector:
    app: bharatvoice-app
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: bharatvoice-ingress
  namespace: bharatvoice
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: bharatvoice-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: bharatvoice-service
            port:
              number: 80
```

Deploy to Kubernetes:

```bash
# Apply configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Check deployment status
kubectl get pods -n bharatvoice
kubectl get services -n bharatvoice
kubectl get ingress -n bharatvoice
```

## Monitoring and Logging

### 1. Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'bharatvoice'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### 2. Grafana Dashboard

Import the provided Grafana dashboard configuration for monitoring:
- Response times
- Request rates
- Error rates
- System resources
- User activity

### 3. Log Aggregation

Configure log forwarding to centralized logging system:

```bash
# Install Filebeat for log shipping
curl -L -O https://artifacts.elastic.co/downloads/beats/filebeat/filebeat-8.8.0-linux-x86_64.tar.gz
tar xzvf filebeat-8.8.0-linux-x86_64.tar.gz

# Configure Filebeat
sudo tee /etc/filebeat/filebeat.yml << EOF
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/bharatvoice/*.log
  fields:
    service: bharatvoice
    environment: production

output.elasticsearch:
  hosts: ["your-elasticsearch-host:9200"]

setup.kibana:
  host: "your-kibana-host:5601"
EOF
```

## Security Configuration

### 1. Firewall Setup

```bash
# Configure UFW firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 2. Security Headers

Configure security headers in Nginx:

```nginx
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header X-Content-Type-Options "nosniff" always;
add_header Referrer-Policy "no-referrer-when-downgrade" always;
add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
```

### 3. Rate Limiting

```nginx
# Rate limiting configuration
http {
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;
    
    server {
        location /api/ {
            limit_req zone=api burst=20 nodelay;
        }
        
        location /auth/login {
            limit_req zone=login burst=5 nodelay;
        }
    }
}
```

## Performance Tuning

### 1. Application Tuning

```bash
# Optimize Python settings
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1

# Configure uvicorn workers
uvicorn bharatvoice.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --max-requests 1000 \
    --max-requests-jitter 100
```

### 2. Database Optimization

```sql
-- PostgreSQL performance tuning
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
SELECT pg_reload_conf();
```

### 3. Redis Optimization

```bash
# Redis performance tuning
echo 'vm.overcommit_memory = 1' >> /etc/sysctl.conf
echo 'net.core.somaxconn = 65535' >> /etc/sysctl.conf
sysctl -p
```

## Backup and Recovery

### 1. Database Backup

```bash
#!/bin/bash
# backup-database.sh

BACKUP_DIR="/opt/bharatvoice/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="bharatvoice"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
pg_dump -h localhost -U bharatvoice -d $DB_NAME | gzip > $BACKUP_DIR/bharatvoice_$DATE.sql.gz

# Keep only last 7 days of backups
find $BACKUP_DIR -name "bharatvoice_*.sql.gz" -mtime +7 -delete

echo "Database backup completed: bharatvoice_$DATE.sql.gz"
```

### 2. Application Data Backup

```bash
#!/bin/bash
# backup-app-data.sh

BACKUP_DIR="/opt/bharatvoice/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup application data
tar -czf $BACKUP_DIR/app_data_$DATE.tar.gz \
    /opt/bharatvoice/uploads \
    /opt/bharatvoice/logs \
    /opt/bharatvoice/.env

echo "Application data backup completed: app_data_$DATE.tar.gz"
```

### 3. Automated Backup Schedule

```bash
# Add to crontab
crontab -e

# Daily database backup at 2 AM
0 2 * * * /opt/bharatvoice/scripts/backup-database.sh

# Weekly application data backup on Sunday at 3 AM
0 3 * * 0 /opt/bharatvoice/scripts/backup-app-data.sh
```

## Health Checks and Monitoring

### 1. Health Check Endpoints

The application provides several health check endpoints:

- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed system status
- `GET /health/database` - Database connectivity
- `GET /health/redis` - Redis connectivity
- `GET /metrics` - Prometheus metrics

### 2. Monitoring Alerts

Configure alerts for:
- High response times (>2s for simple queries)
- High error rates (>5%)
- Database connection issues
- High memory/CPU usage (>80%)
- Disk space usage (>85%)

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed troubleshooting guide.

## Support

For deployment support:
- Email: support@bharatvoice.ai
- Documentation: https://docs.bharatvoice.ai
- GitHub Issues: https://github.com/bharatvoice/assistant/issues

---

**Note**: This deployment guide assumes a production environment. For development deployment, see [DEVELOPMENT.md](DEVELOPMENT.md).