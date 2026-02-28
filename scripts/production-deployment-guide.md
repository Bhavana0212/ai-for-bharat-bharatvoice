<<<<<<< HEAD
# BharatVoice Assistant - Production Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the BharatVoice Assistant to a production Kubernetes environment with all necessary configurations for external API integrations, security hardening, monitoring, and performance optimization.

## Prerequisites

### Infrastructure Requirements

- **Kubernetes Cluster**: v1.24+ with at least 3 worker nodes
- **Node Specifications**: 
  - Minimum: 4 vCPU, 16GB RAM per node
  - Recommended: 8 vCPU, 32GB RAM per node
- **Storage**: 
  - Fast SSD storage class for databases
  - Shared storage class for application data
- **Network**: Load balancer support, ingress controller
- **DNS**: Domain name with DNS management access

### Required Tools

- `kubectl` v1.24+
- `helm` v3.8+
- `docker` v20.10+
- `python` 3.9+ (for configuration scripts)

### External Services

- **SSL Certificates**: Let's Encrypt or commercial SSL provider
- **Container Registry**: Docker Hub, AWS ECR, or similar
- **Monitoring**: Prometheus, Grafana (included in deployment)
- **Backup Storage**: AWS S3 or compatible object storage

## Deployment Steps

### Step 1: Configure External API Integrations

1. **Run the API configuration script**:
   ```bash
   python scripts/configure-external-apis.py setup
   ```

2. **Follow the interactive prompts to configure**:
   - Indian Railways API keys
   - Weather service APIs (OpenWeatherMap, IMD)
   - Digital India platform access
   - UPI payment gateways (Razorpay, PayU)
   - Platform integrations (Swiggy, Ola, Uber)
   - Entertainment APIs (Cricket, Bollywood news)

3. **Validate API connections**:
   ```bash
   python scripts/configure-external-apis.py validate
   ```

### Step 2: Prepare Production Infrastructure

1. **Set up Kubernetes context**:
   ```bash
   kubectl config use-context production
   ```

2. **Create necessary storage classes**:
   ```bash
   kubectl apply -f k8s/production/storage-classes.yaml
   ```

3. **Install required operators**:
   ```bash
   # Cert-manager for SSL certificates
   kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
   
   # Nginx ingress controller
   helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
   helm install ingress-nginx ingress-nginx/ingress-nginx --create-namespace --namespace ingress-nginx
   ```

### Step 3: Deploy Core Infrastructure

1. **Deploy PostgreSQL with high availability**:
   ```bash
   kubectl apply -f k8s/production/postgres-ha.yaml
   ```

2. **Deploy Redis cluster**:
   ```bash
   kubectl apply -f k8s/production/redis-cluster.yaml
   ```

3. **Wait for databases to be ready**:
   ```bash
   kubectl wait --for=condition=ready pod -l app=postgres,role=primary -n bharatvoice --timeout=300s
   kubectl wait --for=condition=ready pod -l app=redis,role=master -n bharatvoice --timeout=300s
   ```

### Step 4: Deploy Application

1. **Build and push Docker image**:
   ```bash
   docker build -t bharatvoice/assistant:v1.0.0 .
   docker push bharatvoice/assistant:v1.0.0
   ```

2. **Deploy application with high availability**:
   ```bash
   kubectl apply -f k8s/production/app-deployment-ha.yaml
   ```

3. **Wait for application to be ready**:
   ```bash
   kubectl wait --for=condition=ready pod -l app=bharatvoice-app -n bharatvoice --timeout=600s
   ```

### Step 5: Configure SSL and Ingress

1. **Deploy SSL certificates and ingress**:
   ```bash
   kubectl apply -f k8s/production/ssl-certificates.yaml
   ```

2. **Update DNS records** to point to the load balancer IP:
   ```bash
   kubectl get service bharatvoice-app-service -n bharatvoice
   ```

### Step 6: Deploy Security and Monitoring

1. **Deploy security hardening**:
   ```bash
   kubectl apply -f k8s/production/security-hardening.yaml
   ```

2. **Deploy secrets management (HashiCorp Vault)**:
   ```bash
   kubectl apply -f k8s/production/secrets-management.yaml
   ```

3. **Deploy rate limiting and DDoS protection**:
   ```bash
   kubectl apply -f k8s/production/rate-limiting-ddos.yaml
   ```

4. **Deploy monitoring stack**:
   ```bash
   kubectl apply -f k8s/production/monitoring-production.yaml
   ```

### Step 7: Configure Performance and Scaling

1. **Deploy performance optimization**:
   ```bash
   kubectl apply -f k8s/production/performance-optimization.yaml
   ```

2. **Configure auto-scaling policies**:
   ```bash
   # HPA and VPA are included in performance-optimization.yaml
   kubectl get hpa -n bharatvoice
   ```

### Step 8: Set Up Backup and Disaster Recovery

1. **Deploy backup and DR configuration**:
   ```bash
   kubectl apply -f k8s/production/backup-disaster-recovery.yaml
   ```

2. **Configure backup storage** (update with your S3 bucket):
   ```bash
   kubectl patch configmap backup-config -n bharatvoice -p '{"data":{"S3_BACKUP_BUCKET":"your-backup-bucket"}}'
   ```

### Step 9: Run Database Migrations

1. **Execute database migrations**:
   ```bash
   kubectl exec -n bharatvoice deployment/bharatvoice-app -- python -m alembic upgrade head
   ```

### Step 10: Verify Deployment

1. **Check all pods are running**:
   ```bash
   kubectl get pods -n bharatvoice
   kubectl get pods -n monitoring
   ```

2. **Test application endpoints**:
   ```bash
   # Health check
   curl https://api.bharatvoice.com/health/ready
   
   # Voice synthesis test
   curl -X POST https://api.bharatvoice.com/api/voice/synthesize \
     -H "Content-Type: application/json" \
     -d '{"text": "नमस्ते", "language": "hi-IN"}'
   ```

3. **Run performance tests**:
   ```bash
   python scripts/performance-testing.py --url https://api.bharatvoice.com --test-type load --duration 300 --rps 10
   ```

## Post-Deployment Configuration

### Monitoring Setup

1. **Access Grafana dashboard**:
   ```bash
   kubectl port-forward -n monitoring svc/grafana 3000:3000
   # Access at http://localhost:3000
   # Get admin password: kubectl get secret grafana-secret -n monitoring -o jsonpath='{.data.admin-password}' | base64 -d
   ```

2. **Configure alerting** in Grafana for:
   - Application health and performance
   - Database and Redis health
   - External API failures
   - Security incidents

### Security Configuration

1. **Review and update security policies**:
   ```bash
   kubectl get networkpolicies -n bharatvoice
   kubectl get podsecuritypolicies
   ```

2. **Configure Vault for secrets management**:
   ```bash
   kubectl exec -n bharatvoice vault-0 -- vault status
   ```

3. **Set up regular security scans**:
   ```bash
   kubectl get cronjob security-scan -n bharatvoice
   ```

### Performance Optimization

1. **Monitor resource usage**:
   ```bash
   kubectl top pods -n bharatvoice
   kubectl top nodes
   ```

2. **Adjust auto-scaling parameters** based on usage patterns:
   ```bash
   kubectl edit hpa bharatvoice-app-hpa -n bharatvoice
   ```

3. **Configure CDN** (CloudFlare or AWS CloudFront) using the configuration templates in `k8s/production/performance-optimization.yaml`

## Maintenance and Operations

### Regular Tasks

1. **Weekly**:
   - Review monitoring dashboards and alerts
   - Check backup job status
   - Review security scan results
   - Monitor external API usage and costs

2. **Monthly**:
   - Update SSL certificates (if not using Let's Encrypt auto-renewal)
   - Review and rotate API keys
   - Analyze performance metrics and optimize
   - Update dependencies and security patches

3. **Quarterly**:
   - Conduct disaster recovery testing
   - Review and update security policies
   - Performance testing and capacity planning
   - Compliance audit and documentation update

### Troubleshooting

1. **Application Issues**:
   ```bash
   # Check application logs
   kubectl logs -f deployment/bharatvoice-app -n bharatvoice
   
   # Check application metrics
   kubectl port-forward -n bharatvoice svc/bharatvoice-app-service 8001:8001
   curl http://localhost:8001/metrics
   ```

2. **Database Issues**:
   ```bash
   # Check PostgreSQL status
   kubectl exec -n bharatvoice postgres-primary-0 -- pg_isready
   
   # Check replication status
   kubectl exec -n bharatvoice postgres-primary-0 -- psql -c "SELECT * FROM pg_stat_replication;"
   ```

3. **Performance Issues**:
   ```bash
   # Run performance diagnostics
   python scripts/performance-testing.py --url https://api.bharatvoice.com --test-type load --duration 60 --rps 5
   
   # Check resource usage
   kubectl top pods -n bharatvoice --sort-by=cpu
   kubectl top pods -n bharatvoice --sort-by=memory
   ```

### Scaling Operations

1. **Manual Scaling**:
   ```bash
   # Scale application pods
   kubectl scale deployment bharatvoice-app --replicas=10 -n bharatvoice
   
   # Scale database replicas
   kubectl scale statefulset postgres-replica --replicas=3 -n bharatvoice
   ```

2. **Update Auto-scaling**:
   ```bash
   # Update HPA thresholds
   kubectl patch hpa bharatvoice-app-hpa -n bharatvoice -p '{"spec":{"maxReplicas":50}}'
   ```

## Security Best Practices

1. **Regular Security Updates**:
   - Keep Kubernetes cluster updated
   - Update container images regularly
   - Monitor security advisories for dependencies

2. **Access Control**:
   - Use RBAC for all service accounts
   - Implement network policies
   - Regular access review and cleanup

3. **Data Protection**:
   - Encrypt data at rest and in transit
   - Regular backup testing
   - Implement data retention policies

4. **Monitoring and Alerting**:
   - Monitor for security incidents
   - Set up alerts for unusual activity
   - Regular security audit logs review

## Support and Documentation

- **API Documentation**: Available at `/docs` endpoint
- **Monitoring Dashboards**: Grafana at monitoring URL
- **Log Aggregation**: Centralized logging with ELK stack
- **Issue Tracking**: Use GitHub issues for bug reports
- **Performance Reports**: Generated weekly by automated tests

## Emergency Procedures

### Disaster Recovery

1. **Database Recovery**:
   ```bash
   # Restore from latest backup
   kubectl exec -n bharatvoice backup-pod -- /scripts/restore-database.sh /backups/database/latest_backup.sql.gz
   ```

2. **Full System Recovery**:
   ```bash
   # Run disaster recovery procedure
   kubectl exec -n bharatvoice backup-pod -- /scripts/disaster-recovery.sh --confirm-disaster-recovery
   ```

### Incident Response

1. **High Error Rate**:
   - Check application logs and metrics
   - Scale up application pods if needed
   - Investigate external API failures
   - Activate backup/fallback services

2. **Performance Degradation**:
   - Check resource utilization
   - Scale up infrastructure if needed
   - Review recent changes
   - Implement traffic throttling if necessary

3. **Security Incident**:
   - Isolate affected components
   - Review security logs
   - Update security policies
   - Notify stakeholders

=======
# BharatVoice Assistant - Production Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the BharatVoice Assistant to a production Kubernetes environment with all necessary configurations for external API integrations, security hardening, monitoring, and performance optimization.

## Prerequisites

### Infrastructure Requirements

- **Kubernetes Cluster**: v1.24+ with at least 3 worker nodes
- **Node Specifications**: 
  - Minimum: 4 vCPU, 16GB RAM per node
  - Recommended: 8 vCPU, 32GB RAM per node
- **Storage**: 
  - Fast SSD storage class for databases
  - Shared storage class for application data
- **Network**: Load balancer support, ingress controller
- **DNS**: Domain name with DNS management access

### Required Tools

- `kubectl` v1.24+
- `helm` v3.8+
- `docker` v20.10+
- `python` 3.9+ (for configuration scripts)

### External Services

- **SSL Certificates**: Let's Encrypt or commercial SSL provider
- **Container Registry**: Docker Hub, AWS ECR, or similar
- **Monitoring**: Prometheus, Grafana (included in deployment)
- **Backup Storage**: AWS S3 or compatible object storage

## Deployment Steps

### Step 1: Configure External API Integrations

1. **Run the API configuration script**:
   ```bash
   python scripts/configure-external-apis.py setup
   ```

2. **Follow the interactive prompts to configure**:
   - Indian Railways API keys
   - Weather service APIs (OpenWeatherMap, IMD)
   - Digital India platform access
   - UPI payment gateways (Razorpay, PayU)
   - Platform integrations (Swiggy, Ola, Uber)
   - Entertainment APIs (Cricket, Bollywood news)

3. **Validate API connections**:
   ```bash
   python scripts/configure-external-apis.py validate
   ```

### Step 2: Prepare Production Infrastructure

1. **Set up Kubernetes context**:
   ```bash
   kubectl config use-context production
   ```

2. **Create necessary storage classes**:
   ```bash
   kubectl apply -f k8s/production/storage-classes.yaml
   ```

3. **Install required operators**:
   ```bash
   # Cert-manager for SSL certificates
   kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
   
   # Nginx ingress controller
   helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
   helm install ingress-nginx ingress-nginx/ingress-nginx --create-namespace --namespace ingress-nginx
   ```

### Step 3: Deploy Core Infrastructure

1. **Deploy PostgreSQL with high availability**:
   ```bash
   kubectl apply -f k8s/production/postgres-ha.yaml
   ```

2. **Deploy Redis cluster**:
   ```bash
   kubectl apply -f k8s/production/redis-cluster.yaml
   ```

3. **Wait for databases to be ready**:
   ```bash
   kubectl wait --for=condition=ready pod -l app=postgres,role=primary -n bharatvoice --timeout=300s
   kubectl wait --for=condition=ready pod -l app=redis,role=master -n bharatvoice --timeout=300s
   ```

### Step 4: Deploy Application

1. **Build and push Docker image**:
   ```bash
   docker build -t bharatvoice/assistant:v1.0.0 .
   docker push bharatvoice/assistant:v1.0.0
   ```

2. **Deploy application with high availability**:
   ```bash
   kubectl apply -f k8s/production/app-deployment-ha.yaml
   ```

3. **Wait for application to be ready**:
   ```bash
   kubectl wait --for=condition=ready pod -l app=bharatvoice-app -n bharatvoice --timeout=600s
   ```

### Step 5: Configure SSL and Ingress

1. **Deploy SSL certificates and ingress**:
   ```bash
   kubectl apply -f k8s/production/ssl-certificates.yaml
   ```

2. **Update DNS records** to point to the load balancer IP:
   ```bash
   kubectl get service bharatvoice-app-service -n bharatvoice
   ```

### Step 6: Deploy Security and Monitoring

1. **Deploy security hardening**:
   ```bash
   kubectl apply -f k8s/production/security-hardening.yaml
   ```

2. **Deploy secrets management (HashiCorp Vault)**:
   ```bash
   kubectl apply -f k8s/production/secrets-management.yaml
   ```

3. **Deploy rate limiting and DDoS protection**:
   ```bash
   kubectl apply -f k8s/production/rate-limiting-ddos.yaml
   ```

4. **Deploy monitoring stack**:
   ```bash
   kubectl apply -f k8s/production/monitoring-production.yaml
   ```

### Step 7: Configure Performance and Scaling

1. **Deploy performance optimization**:
   ```bash
   kubectl apply -f k8s/production/performance-optimization.yaml
   ```

2. **Configure auto-scaling policies**:
   ```bash
   # HPA and VPA are included in performance-optimization.yaml
   kubectl get hpa -n bharatvoice
   ```

### Step 8: Set Up Backup and Disaster Recovery

1. **Deploy backup and DR configuration**:
   ```bash
   kubectl apply -f k8s/production/backup-disaster-recovery.yaml
   ```

2. **Configure backup storage** (update with your S3 bucket):
   ```bash
   kubectl patch configmap backup-config -n bharatvoice -p '{"data":{"S3_BACKUP_BUCKET":"your-backup-bucket"}}'
   ```

### Step 9: Run Database Migrations

1. **Execute database migrations**:
   ```bash
   kubectl exec -n bharatvoice deployment/bharatvoice-app -- python -m alembic upgrade head
   ```

### Step 10: Verify Deployment

1. **Check all pods are running**:
   ```bash
   kubectl get pods -n bharatvoice
   kubectl get pods -n monitoring
   ```

2. **Test application endpoints**:
   ```bash
   # Health check
   curl https://api.bharatvoice.com/health/ready
   
   # Voice synthesis test
   curl -X POST https://api.bharatvoice.com/api/voice/synthesize \
     -H "Content-Type: application/json" \
     -d '{"text": "नमस्ते", "language": "hi-IN"}'
   ```

3. **Run performance tests**:
   ```bash
   python scripts/performance-testing.py --url https://api.bharatvoice.com --test-type load --duration 300 --rps 10
   ```

## Post-Deployment Configuration

### Monitoring Setup

1. **Access Grafana dashboard**:
   ```bash
   kubectl port-forward -n monitoring svc/grafana 3000:3000
   # Access at http://localhost:3000
   # Get admin password: kubectl get secret grafana-secret -n monitoring -o jsonpath='{.data.admin-password}' | base64 -d
   ```

2. **Configure alerting** in Grafana for:
   - Application health and performance
   - Database and Redis health
   - External API failures
   - Security incidents

### Security Configuration

1. **Review and update security policies**:
   ```bash
   kubectl get networkpolicies -n bharatvoice
   kubectl get podsecuritypolicies
   ```

2. **Configure Vault for secrets management**:
   ```bash
   kubectl exec -n bharatvoice vault-0 -- vault status
   ```

3. **Set up regular security scans**:
   ```bash
   kubectl get cronjob security-scan -n bharatvoice
   ```

### Performance Optimization

1. **Monitor resource usage**:
   ```bash
   kubectl top pods -n bharatvoice
   kubectl top nodes
   ```

2. **Adjust auto-scaling parameters** based on usage patterns:
   ```bash
   kubectl edit hpa bharatvoice-app-hpa -n bharatvoice
   ```

3. **Configure CDN** (CloudFlare or AWS CloudFront) using the configuration templates in `k8s/production/performance-optimization.yaml`

## Maintenance and Operations

### Regular Tasks

1. **Weekly**:
   - Review monitoring dashboards and alerts
   - Check backup job status
   - Review security scan results
   - Monitor external API usage and costs

2. **Monthly**:
   - Update SSL certificates (if not using Let's Encrypt auto-renewal)
   - Review and rotate API keys
   - Analyze performance metrics and optimize
   - Update dependencies and security patches

3. **Quarterly**:
   - Conduct disaster recovery testing
   - Review and update security policies
   - Performance testing and capacity planning
   - Compliance audit and documentation update

### Troubleshooting

1. **Application Issues**:
   ```bash
   # Check application logs
   kubectl logs -f deployment/bharatvoice-app -n bharatvoice
   
   # Check application metrics
   kubectl port-forward -n bharatvoice svc/bharatvoice-app-service 8001:8001
   curl http://localhost:8001/metrics
   ```

2. **Database Issues**:
   ```bash
   # Check PostgreSQL status
   kubectl exec -n bharatvoice postgres-primary-0 -- pg_isready
   
   # Check replication status
   kubectl exec -n bharatvoice postgres-primary-0 -- psql -c "SELECT * FROM pg_stat_replication;"
   ```

3. **Performance Issues**:
   ```bash
   # Run performance diagnostics
   python scripts/performance-testing.py --url https://api.bharatvoice.com --test-type load --duration 60 --rps 5
   
   # Check resource usage
   kubectl top pods -n bharatvoice --sort-by=cpu
   kubectl top pods -n bharatvoice --sort-by=memory
   ```

### Scaling Operations

1. **Manual Scaling**:
   ```bash
   # Scale application pods
   kubectl scale deployment bharatvoice-app --replicas=10 -n bharatvoice
   
   # Scale database replicas
   kubectl scale statefulset postgres-replica --replicas=3 -n bharatvoice
   ```

2. **Update Auto-scaling**:
   ```bash
   # Update HPA thresholds
   kubectl patch hpa bharatvoice-app-hpa -n bharatvoice -p '{"spec":{"maxReplicas":50}}'
   ```

## Security Best Practices

1. **Regular Security Updates**:
   - Keep Kubernetes cluster updated
   - Update container images regularly
   - Monitor security advisories for dependencies

2. **Access Control**:
   - Use RBAC for all service accounts
   - Implement network policies
   - Regular access review and cleanup

3. **Data Protection**:
   - Encrypt data at rest and in transit
   - Regular backup testing
   - Implement data retention policies

4. **Monitoring and Alerting**:
   - Monitor for security incidents
   - Set up alerts for unusual activity
   - Regular security audit logs review

## Support and Documentation

- **API Documentation**: Available at `/docs` endpoint
- **Monitoring Dashboards**: Grafana at monitoring URL
- **Log Aggregation**: Centralized logging with ELK stack
- **Issue Tracking**: Use GitHub issues for bug reports
- **Performance Reports**: Generated weekly by automated tests

## Emergency Procedures

### Disaster Recovery

1. **Database Recovery**:
   ```bash
   # Restore from latest backup
   kubectl exec -n bharatvoice backup-pod -- /scripts/restore-database.sh /backups/database/latest_backup.sql.gz
   ```

2. **Full System Recovery**:
   ```bash
   # Run disaster recovery procedure
   kubectl exec -n bharatvoice backup-pod -- /scripts/disaster-recovery.sh --confirm-disaster-recovery
   ```

### Incident Response

1. **High Error Rate**:
   - Check application logs and metrics
   - Scale up application pods if needed
   - Investigate external API failures
   - Activate backup/fallback services

2. **Performance Degradation**:
   - Check resource utilization
   - Scale up infrastructure if needed
   - Review recent changes
   - Implement traffic throttling if necessary

3. **Security Incident**:
   - Isolate affected components
   - Review security logs
   - Update security policies
   - Notify stakeholders

>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
This production deployment guide ensures a robust, secure, and scalable deployment of the BharatVoice Assistant with comprehensive monitoring, security, and operational procedures.