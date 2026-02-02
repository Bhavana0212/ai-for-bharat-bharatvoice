# BharatVoice Containerization and Deployment Guide

This guide covers the complete containerization and deployment setup for the BharatVoice AI-powered multilingual voice assistant.

## Overview

The deployment architecture includes:
- **Production-optimized Docker containers** with multi-stage builds
- **Kubernetes deployment configurations** with auto-scaling
- **Comprehensive monitoring and logging** with Prometheus, Grafana, and Loki
- **Automated backup and disaster recovery** procedures
- **Load balancing and ingress** with Nginx

## Architecture Components

### Container Images
- `bharatvoice/assistant:latest` - Main application container
- `bharatvoice/worker:latest` - Background worker container
- `postgres:15-alpine` - PostgreSQL database
- `redis:7-alpine` - Redis cache and message broker
- `nginx:alpine` - Reverse proxy and load balancer

### Kubernetes Resources
- **Namespace**: `bharatvoice` with resource quotas and limits
- **Deployments**: Application, worker, and supporting services
- **StatefulSets**: PostgreSQL for persistent data
- **Services**: Internal communication and load balancing
- **ConfigMaps**: Application configuration
- **Secrets**: Sensitive data (passwords, keys, certificates)
- **PersistentVolumes**: Data persistence for database and logs

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Kubernetes cluster (1.20+)
- kubectl configured
- Helm (optional, for monitoring stack)
- AWS CLI (for S3 backups)

### 1. Build and Test Locally

```bash
# Build production containers
docker build -f Dockerfile.production -t bharatvoice/assistant:latest .
docker build -f Dockerfile.worker -t bharatvoice/worker:latest .

# Test with Docker Compose
docker-compose -f docker-compose.production.yml up -d

# Verify services
curl http://localhost:8000/health/live
```

### 2. Deploy to Kubernetes

```bash
# Make deployment script executable
chmod +x scripts/deploy.sh

# Deploy complete stack
./scripts/deploy.sh deploy

# Check deployment status
./scripts/deploy.sh status
```

### 3. Configure Monitoring

```bash
# Access Grafana (default: admin/admin)
kubectl port-forward -n bharatvoice service/grafana-service 3000:3000

# Access Prometheus
kubectl port-forward -n bharatvoice service/prometheus-service 9090:9090
```

## Detailed Deployment

### Container Optimization

#### Production Dockerfile Features
- **Multi-stage builds** for smaller final images
- **Non-root user** for security
- **Health checks** for container orchestration
- **Optimized Python dependencies** with virtual environments
- **Production WSGI server** (Gunicorn) with proper worker management

#### Worker Container
- **Specialized Celery worker** for background tasks
- **Separate resource limits** for task processing
- **Independent scaling** from main application

### Kubernetes Configuration

#### Namespace and Resource Management
```yaml
# Resource quotas
requests.cpu: "8"
requests.memory: 16Gi
limits.cpu: "16"
limits.memory: 32Gi

# Pod limits
default.cpu: "1"
default.memory: "2Gi"
```

#### Application Deployment
- **3 replicas** for high availability
- **Rolling updates** with zero downtime
- **Resource requests/limits** for proper scheduling
- **Health checks** (liveness, readiness, startup)
- **Persistent volumes** for data and logs

#### Database Configuration
- **StatefulSet** for PostgreSQL with persistent storage
- **Connection pooling** and optimization
- **Automated backups** with encryption
- **Point-in-time recovery** capability

### Auto-scaling Configuration

#### Horizontal Pod Autoscaler (HPA)
```yaml
# Application scaling
minReplicas: 3
maxReplicas: 20
targetCPUUtilization: 70%
targetMemoryUtilization: 80%

# Worker scaling
minReplicas: 2
maxReplicas: 10
targetCPUUtilization: 75%
```

#### Vertical Pod Autoscaler (VPA)
- **Automatic resource optimization** based on usage patterns
- **Recommendation mode** for manual adjustments
- **Resource boundaries** to prevent over-allocation

#### Pod Disruption Budgets (PDB)
- **Minimum availability** during cluster maintenance
- **Graceful handling** of node updates and failures

### Monitoring and Observability

#### Metrics Collection
- **Prometheus** for metrics aggregation
- **Custom metrics** for voice processing performance
- **Application-specific dashboards** in Grafana
- **Alert rules** for proactive monitoring

#### Log Aggregation
- **Loki** for centralized log storage
- **Promtail** for log shipping
- **Structured logging** with correlation IDs
- **Log retention policies** for compliance

#### Key Metrics Monitored
- **Response times** (95th percentile < 2s for simple queries)
- **Error rates** (< 1% for critical endpoints)
- **Voice processing latency** (< 5s for complex multilingual queries)
- **Resource utilization** (CPU, memory, storage)
- **Database performance** (connection pool, query times)

### Load Balancing and Ingress

#### Nginx Configuration
- **SSL termination** with TLS 1.2/1.3
- **Rate limiting** (10 req/s general, 1 req/s auth)
- **Compression** (gzip) for better performance
- **Security headers** (HSTS, CSP, X-Frame-Options)
- **WebSocket support** for real-time features

#### Traffic Routing
- **Health check endpoints** (no rate limiting)
- **API endpoints** with standard rate limiting
- **Voice processing** with extended timeouts and larger body sizes
- **Static assets** with long-term caching

### Backup and Disaster Recovery

#### Automated Backups
```bash
# Database backup (daily at 2 AM IST)
- PostgreSQL dumps with compression and encryption
- S3 upload with lifecycle policies
- 30-day retention with automated cleanup

# Redis backup (daily at 3 AM IST)
- RDB snapshots
- 7-day retention

# Application data (weekly)
- User data and models
- Configuration backups
```

#### Disaster Recovery Procedures
```bash
# Check system status
./scripts/disaster-recovery.sh status

# Create emergency backup
./scripts/disaster-recovery.sh backup

# Full system restore
./scripts/disaster-recovery.sh restore --mode full

# Database-only restore
./scripts/disaster-recovery.sh restore --mode database-only
```

## Security Considerations

### Container Security
- **Non-root containers** with minimal privileges
- **Read-only root filesystems** where possible
- **Security scanning** of base images
- **Regular updates** of dependencies

### Network Security
- **Network policies** for pod-to-pod communication
- **TLS encryption** for all external traffic
- **Service mesh** (optional) for internal encryption
- **Firewall rules** for database access

### Data Protection
- **Encryption at rest** for persistent volumes
- **Encryption in transit** for all communications
- **Secret management** with Kubernetes secrets
- **Backup encryption** with GPG

## Performance Optimization

### Resource Allocation
```yaml
# Application pods
requests:
  memory: "1Gi"
  cpu: "500m"
limits:
  memory: "2Gi"
  cpu: "1000m"

# Database
requests:
  memory: "512Mi"
  cpu: "250m"
limits:
  memory: "1Gi"
  cpu: "500m"
```

### Caching Strategy
- **Redis** for session and API response caching
- **Application-level caching** for ML models
- **CDN integration** for static assets
- **Database query optimization** with connection pooling

### Voice Processing Optimization
- **Model caching** in persistent volumes
- **Batch processing** for multiple requests
- **GPU acceleration** (optional) for ML workloads
- **Streaming responses** for real-time interaction

## Troubleshooting

### Common Issues

#### Pod Startup Failures
```bash
# Check pod status
kubectl get pods -n bharatvoice

# View pod logs
kubectl logs -n bharatvoice <pod-name>

# Describe pod for events
kubectl describe pod -n bharatvoice <pod-name>
```

#### Database Connection Issues
```bash
# Check database pod
kubectl get pods -n bharatvoice -l app=postgres

# Test database connectivity
kubectl exec -n bharatvoice deployment/bharatvoice-app -- \
  pg_isready -h postgres-service -p 5432
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n bharatvoice

# View HPA status
kubectl get hpa -n bharatvoice

# Check application metrics
curl http://<external-ip>/metrics
```

### Log Analysis
```bash
# Application logs
kubectl logs -n bharatvoice deployment/bharatvoice-app --tail=100

# Worker logs
kubectl logs -n bharatvoice deployment/bharatvoice-worker --tail=100

# Database logs
kubectl logs -n bharatvoice statefulset/postgres --tail=100
```

## Maintenance Procedures

### Regular Updates
```bash
# Update application
./scripts/deploy.sh update --tag v1.2.0

# Rollback if needed
./scripts/deploy.sh rollback
```

### Database Maintenance
```bash
# Manual backup
./scripts/backup-database.sh

# Restore from backup
./scripts/restore-database.sh backup_file.sql.gpg

# Database migrations
kubectl exec -n bharatvoice deployment/bharatvoice-app -- \
  python -m alembic upgrade head
```

### Monitoring Maintenance
```bash
# Restart monitoring stack
kubectl rollout restart deployment/prometheus -n bharatvoice
kubectl rollout restart deployment/grafana -n bharatvoice

# Clear old metrics
kubectl exec -n bharatvoice deployment/prometheus -- \
  promtool tsdb delete-series --match='{__name__=~".*"}'
```

## Cost Optimization

### Resource Right-sizing
- **VPA recommendations** for optimal resource allocation
- **Spot instances** for non-critical workloads
- **Cluster autoscaling** to match demand
- **Reserved instances** for predictable workloads

### Storage Optimization
- **Lifecycle policies** for backup retention
- **Compression** for logs and backups
- **Tiered storage** (Standard → IA → Glacier)
- **Unused volume cleanup** automation

## Compliance and Governance

### Data Residency
- **Indian data centers** for user data
- **Cross-border data transfer** controls
- **Audit logging** for compliance
- **Data retention policies** per regulations

### Security Compliance
- **Regular security scans** of containers
- **Vulnerability assessments** of infrastructure
- **Access control** with RBAC
- **Audit trails** for all administrative actions

## Support and Monitoring

### Health Checks
- **Application health** endpoints
- **Database connectivity** checks
- **External service** availability
- **Voice processing** functionality tests

### Alerting
- **Critical alerts** (application down, database failure)
- **Warning alerts** (high latency, resource usage)
- **Performance alerts** (voice processing failures)
- **Security alerts** (authentication failures, rate limit breaches)

### Dashboards
- **Application overview** with key metrics
- **Infrastructure monitoring** (CPU, memory, disk)
- **Voice processing performance** metrics
- **User experience** monitoring (response times, error rates)

This comprehensive deployment setup ensures high availability, scalability, and maintainability of the BharatVoice assistant in production environments.