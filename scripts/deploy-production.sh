#!/bin/bash
set -e

# Production Deployment Script for BharatVoice Assistant
# This script deploys the complete production infrastructure to Kubernetes

# Configuration
NAMESPACE="bharatvoice"
MONITORING_NAMESPACE="monitoring"
DOCKER_REGISTRY="bharatvoice"
IMAGE_TAG="${IMAGE_TAG:-latest}"
KUBECTL_CONTEXT="${KUBECTL_CONTEXT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed or not in PATH"
        exit 1
    fi
    
    # Check docker
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check kubectl context
    if ! kubectl config get-contexts | grep -q "${KUBECTL_CONTEXT}"; then
        log_error "Kubectl context '${KUBECTL_CONTEXT}' not found"
        log_info "Available contexts:"
        kubectl config get-contexts
        exit 1
    fi
    
    # Set kubectl context
    kubectl config use-context "${KUBECTL_CONTEXT}"
    
    log_success "Prerequisites check passed"
}

# Create namespaces
create_namespaces() {
    log_info "Creating namespaces..."
    
    kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    kubectl create namespace "${MONITORING_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    
    # Label namespaces
    kubectl label namespace "${NAMESPACE}" name="${NAMESPACE}" --overwrite
    kubectl label namespace "${MONITORING_NAMESPACE}" name="${MONITORING_NAMESPACE}" --overwrite
    
    log_success "Namespaces created"
}

# Create secrets
create_secrets() {
    log_info "Creating secrets..."
    
    # Check if .env.production exists
    if [ ! -f "config/production/.env.production" ]; then
        log_error ".env.production file not found"
        log_info "Please run 'python scripts/configure-external-apis.py setup' first"
        exit 1
    fi
    
    # Source environment variables
    set -a
    source config/production/.env.production
    set +a
    
    # Create application secrets
    kubectl create secret generic bharatvoice-secrets \
        --from-literal=SECRET_KEY="${SECRET_KEY}" \
        --from-literal=ENCRYPTION_KEY="${ENCRYPTION_KEY}" \
        --from-literal=JWT_SECRET_KEY="${SECRET_KEY}" \
        --namespace="${NAMESPACE}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Create database secrets
    kubectl create secret generic postgres-secret \
        --from-literal=POSTGRES_USER="${POSTGRES_USER:-bharatvoice}" \
        --from-literal=POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$(openssl rand -base64 32)}" \
        --from-literal=POSTGRES_DB="${POSTGRES_DB:-bharatvoice}" \
        --from-literal=POSTGRES_REPLICATION_PASSWORD="$(openssl rand -base64 32)" \
        --namespace="${NAMESPACE}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Create Redis secrets
    kubectl create secret generic redis-secret \
        --from-literal=REDIS_PASSWORD="$(openssl rand -base64 32)" \
        --namespace="${NAMESPACE}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Create external API secrets
    kubectl create secret generic external-api-secrets \
        --from-literal=INDIAN_RAILWAYS_API_KEY="${INDIAN_RAILWAYS_API_KEY:-}" \
        --from-literal=OPENWEATHERMAP_API_KEY="${OPENWEATHERMAP_API_KEY:-}" \
        --from-literal=RAZORPAY_KEY_ID="${RAZORPAY_KEY_ID:-}" \
        --from-literal=RAZORPAY_KEY_SECRET="${RAZORPAY_KEY_SECRET:-}" \
        --from-literal=SWIGGY_API_KEY="${SWIGGY_API_KEY:-}" \
        --from-literal=OLA_CLIENT_ID="${OLA_CLIENT_ID:-}" \
        --from-literal=OLA_CLIENT_SECRET="${OLA_CLIENT_SECRET:-}" \
        --namespace="${NAMESPACE}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Create monitoring secrets
    kubectl create secret generic grafana-secret \
        --from-literal=admin-password="$(openssl rand -base64 32)" \
        --namespace="${MONITORING_NAMESPACE}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Create SSL certificate secrets (placeholder - replace with actual certificates)
    kubectl create secret tls bharatvoice-ssl-certs \
        --cert=certs/bharatvoice.crt \
        --key=certs/bharatvoice.key \
        --namespace="${NAMESPACE}" \
        --dry-run=client -o yaml | kubectl apply -f - || log_warning "SSL certificates not found, skipping"
    
    log_success "Secrets created"
}

# Deploy storage classes
deploy_storage_classes() {
    log_info "Deploying storage classes..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  encrypted: "true"
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: shared-storage
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: efs-ap
  fileSystemId: fs-xxxxxxxxx
  directoryPerms: "700"
allowVolumeExpansion: true
volumeBindingMode: Immediate
EOF
    
    log_success "Storage classes deployed"
}

# Deploy PostgreSQL
deploy_postgresql() {
    log_info "Deploying PostgreSQL with high availability..."
    
    kubectl apply -f k8s/production/postgres-ha.yaml
    
    # Wait for PostgreSQL to be ready
    log_info "Waiting for PostgreSQL to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres,role=primary -n "${NAMESPACE}" --timeout=300s
    
    log_success "PostgreSQL deployed successfully"
}

# Deploy Redis
deploy_redis() {
    log_info "Deploying Redis cluster..."
    
    kubectl apply -f k8s/production/redis-cluster.yaml
    
    # Wait for Redis to be ready
    log_info "Waiting for Redis to be ready..."
    kubectl wait --for=condition=ready pod -l app=redis,role=master -n "${NAMESPACE}" --timeout=300s
    
    log_success "Redis deployed successfully"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    # Build image
    docker build -t "${DOCKER_REGISTRY}/assistant:${IMAGE_TAG}" .
    
    # Push image
    docker push "${DOCKER_REGISTRY}/assistant:${IMAGE_TAG}"
    
    log_success "Docker image built and pushed"
}

# Deploy application
deploy_application() {
    log_info "Deploying BharatVoice application..."
    
    # Update image tag in deployment
    sed -i.bak "s|bharatvoice/assistant:latest|${DOCKER_REGISTRY}/assistant:${IMAGE_TAG}|g" k8s/production/app-deployment-ha.yaml
    
    kubectl apply -f k8s/production/app-deployment-ha.yaml
    
    # Restore original file
    mv k8s/production/app-deployment-ha.yaml.bak k8s/production/app-deployment-ha.yaml
    
    # Wait for application to be ready
    log_info "Waiting for application to be ready..."
    kubectl wait --for=condition=ready pod -l app=bharatvoice-app -n "${NAMESPACE}" --timeout=600s
    
    log_success "Application deployed successfully"
}

# Deploy SSL certificates
deploy_ssl_certificates() {
    log_info "Deploying SSL certificates and ingress..."
    
    # Install cert-manager if not present
    if ! kubectl get namespace cert-manager &> /dev/null; then
        log_info "Installing cert-manager..."
        kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
        kubectl wait --for=condition=ready pod -l app=cert-manager -n cert-manager --timeout=300s
    fi
    
    # Install nginx-ingress if not present
    if ! kubectl get namespace ingress-nginx &> /dev/null; then
        log_info "Installing nginx-ingress..."
        helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
        helm repo update
        helm install ingress-nginx ingress-nginx/ingress-nginx \
            --create-namespace \
            --namespace ingress-nginx \
            --set controller.service.type=LoadBalancer
        kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=ingress-nginx -n ingress-nginx --timeout=300s
    fi
    
    kubectl apply -f k8s/production/ssl-certificates.yaml
    
    log_success "SSL certificates and ingress deployed"
}

# Deploy monitoring
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    kubectl apply -f k8s/production/monitoring-production.yaml
    
    # Wait for monitoring to be ready
    log_info "Waiting for monitoring stack to be ready..."
    kubectl wait --for=condition=ready pod -l app=prometheus -n "${MONITORING_NAMESPACE}" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=grafana -n "${MONITORING_NAMESPACE}" --timeout=300s
    
    log_success "Monitoring stack deployed successfully"
}

# Deploy backup and disaster recovery
deploy_backup_dr() {
    log_info "Deploying backup and disaster recovery..."
    
    kubectl apply -f k8s/production/backup-disaster-recovery.yaml
    
    log_success "Backup and disaster recovery deployed"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Get a running application pod
    APP_POD=$(kubectl get pods -l app=bharatvoice-app -n "${NAMESPACE}" -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "${APP_POD}" ]; then
        log_error "No application pods found"
        exit 1
    fi
    
    # Run migrations
    kubectl exec -n "${NAMESPACE}" "${APP_POD}" -- python -m alembic upgrade head
    
    log_success "Database migrations completed"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pod status
    log_info "Checking pod status..."
    kubectl get pods -n "${NAMESPACE}"
    kubectl get pods -n "${MONITORING_NAMESPACE}"
    
    # Check services
    log_info "Checking services..."
    kubectl get services -n "${NAMESPACE}"
    
    # Check ingress
    log_info "Checking ingress..."
    kubectl get ingress -n "${NAMESPACE}"
    
    # Test application health
    APP_POD=$(kubectl get pods -l app=bharatvoice-app -n "${NAMESPACE}" -o jsonpath='{.items[0].metadata.name}')
    if kubectl exec -n "${NAMESPACE}" "${APP_POD}" -- curl -f http://localhost:8000/health/ready > /dev/null 2>&1; then
        log_success "Application health check passed"
    else
        log_error "Application health check failed"
        exit 1
    fi
    
    # Get external IP
    EXTERNAL_IP=$(kubectl get service bharatvoice-app-service -n "${NAMESPACE}" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -n "${EXTERNAL_IP}" ]; then
        log_success "Application is accessible at: http://${EXTERNAL_IP}"
    else
        log_warning "External IP not yet assigned, check service status"
    fi
    
    # Get Grafana admin password
    GRAFANA_PASSWORD=$(kubectl get secret grafana-secret -n "${MONITORING_NAMESPACE}" -o jsonpath='{.data.admin-password}' | base64 -d)
    log_info "Grafana admin password: ${GRAFANA_PASSWORD}"
    
    log_success "Deployment verification completed"
}

# Cleanup function
cleanup() {
    if [ $? -ne 0 ]; then
        log_error "Deployment failed. Check the logs above for details."
        log_info "To cleanup partial deployment, run: kubectl delete namespace ${NAMESPACE} ${MONITORING_NAMESPACE}"
    fi
}

# Main deployment function
main() {
    log_info "Starting BharatVoice production deployment..."
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-migrations)
                SKIP_MIGRATIONS=true
                shift
                ;;
            --image-tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            --context)
                KUBECTL_CONTEXT="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-build        Skip Docker image build and push"
                echo "  --skip-migrations   Skip database migrations"
                echo "  --image-tag TAG     Docker image tag (default: latest)"
                echo "  --context CONTEXT   Kubectl context (default: production)"
                echo "  --help              Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute deployment steps
    check_prerequisites
    create_namespaces
    create_secrets
    deploy_storage_classes
    deploy_postgresql
    deploy_redis
    
    if [ "${SKIP_BUILD}" != "true" ]; then
        build_and_push_image
    fi
    
    deploy_application
    deploy_ssl_certificates
    deploy_monitoring
    deploy_backup_dr
    
    if [ "${SKIP_MIGRATIONS}" != "true" ]; then
        run_migrations
    fi
    
    verify_deployment
    
    log_success "🎉 BharatVoice production deployment completed successfully!"
    log_info "Next steps:"
    log_info "1. Configure DNS to point to the load balancer IP"
    log_info "2. Set up external API integrations using the configuration script"
    log_info "3. Configure monitoring alerts and notifications"
    log_info "4. Test disaster recovery procedures"
}

# Run main function
main "$@"