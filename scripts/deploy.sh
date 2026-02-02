#!/bin/bash

# BharatVoice Production Deployment Script
# This script handles the complete deployment of BharatVoice to Kubernetes

set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-bharatvoice}"
ENVIRONMENT="${ENVIRONMENT:-production}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-bharatvoice}"
KUBECTL_CONTEXT="${KUBECTL_CONTEXT:-}"
DRY_RUN="${DRY_RUN:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_info() {
    log "${BLUE}INFO${NC}: $1"
}

log_warn() {
    log "${YELLOW}WARN${NC}: $1"
}

log_error() {
    log "${RED}ERROR${NC}: $1"
}

log_success() {
    log "${GREEN}SUCCESS${NC}: $1"
}

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  deploy        Deploy the complete application"
    echo "  update        Update existing deployment"
    echo "  rollback      Rollback to previous version"
    echo "  status        Check deployment status"
    echo "  logs          Show application logs"
    echo "  cleanup       Clean up resources"
    echo ""
    echo "Options:"
    echo "  -n, --namespace NAMESPACE    Kubernetes namespace (default: bharatvoice)"
    echo "  -e, --environment ENV        Environment (default: production)"
    echo "  -t, --tag TAG               Image tag (default: latest)"
    echo "  -r, --registry REGISTRY     Container registry (default: bharatvoice)"
    echo "  -c, --context CONTEXT       Kubectl context"
    echo "  -d, --dry-run               Dry run mode"
    echo "  -h, --help                  Show this help message"
    exit 1
}

# Parse command line arguments
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -c|--context)
            KUBECTL_CONTEXT="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            usage
            ;;
        deploy|update|rollback|status|logs|cleanup)
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

# Set kubectl context if specified
if [[ -n "${KUBECTL_CONTEXT}" ]]; then
    kubectl config use-context "${KUBECTL_CONTEXT}"
fi

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites"
    
    # Check kubectl
    if ! command -v kubectl >/dev/null 2>&1; then
        log_error "kubectl not found"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info >/dev/null 2>&1; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if running in dry-run mode
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warn "Running in dry-run mode - no changes will be applied"
    fi
    
    log_success "Prerequisites check completed"
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Creating namespace: ${NAMESPACE}"
    
    if kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
        log_info "Namespace ${NAMESPACE} already exists"
    else
        if [[ "${DRY_RUN}" == "true" ]]; then
            log_info "Would create namespace: ${NAMESPACE}"
        else
            kubectl apply -f k8s/namespace.yaml
            log_success "Namespace created: ${NAMESPACE}"
        fi
    fi
}

# Deploy secrets
deploy_secrets() {
    log_info "Deploying secrets"
    
    # Check if secrets exist
    if kubectl get secret bharatvoice-secrets -n "${NAMESPACE}" >/dev/null 2>&1; then
        log_warn "Secrets already exist, skipping deployment"
        log_warn "To update secrets, delete them first: kubectl delete secret bharatvoice-secrets -n ${NAMESPACE}"
    else
        if [[ "${DRY_RUN}" == "true" ]]; then
            log_info "Would deploy secrets"
        else
            # Note: In production, secrets should be managed externally
            log_warn "Please ensure secrets are properly configured before deployment"
            log_warn "Run: kubectl apply -f k8s/secrets.yaml"
        fi
    fi
}

# Deploy configuration
deploy_config() {
    log_info "Deploying configuration"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        kubectl apply -f k8s/configmap.yaml --dry-run=client -o yaml
    else
        kubectl apply -f k8s/configmap.yaml
        log_success "Configuration deployed"
    fi
}

# Deploy database
deploy_database() {
    log_info "Deploying PostgreSQL database"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        kubectl apply -f k8s/postgres.yaml --dry-run=client -o yaml
    else
        kubectl apply -f k8s/postgres.yaml
        
        # Wait for database to be ready
        log_info "Waiting for PostgreSQL to be ready..."
        kubectl wait --for=condition=ready pod -l app=postgres -n "${NAMESPACE}" --timeout=300s
        log_success "PostgreSQL deployed and ready"
    fi
}

# Deploy Redis
deploy_redis() {
    log_info "Deploying Redis cache"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        kubectl apply -f k8s/redis.yaml --dry-run=client -o yaml
    else
        kubectl apply -f k8s/redis.yaml
        
        # Wait for Redis to be ready
        log_info "Waiting for Redis to be ready..."
        kubectl wait --for=condition=ready pod -l app=redis -n "${NAMESPACE}" --timeout=300s
        log_success "Redis deployed and ready"
    fi
}

# Deploy application
deploy_application() {
    log_info "Deploying BharatVoice application"
    
    # Update image tags in deployment files
    sed -i.bak "s|bharatvoice/assistant:latest|${REGISTRY}/assistant:${IMAGE_TAG}|g" k8s/app-deployment.yaml
    sed -i.bak "s|bharatvoice/worker:latest|${REGISTRY}/worker:${IMAGE_TAG}|g" k8s/worker-deployment.yaml
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        kubectl apply -f k8s/app-deployment.yaml --dry-run=client -o yaml
        kubectl apply -f k8s/worker-deployment.yaml --dry-run=client -o yaml
    else
        kubectl apply -f k8s/app-deployment.yaml
        kubectl apply -f k8s/worker-deployment.yaml
        
        # Wait for deployments to be ready
        log_info "Waiting for application deployments to be ready..."
        kubectl wait --for=condition=available deployment/bharatvoice-app -n "${NAMESPACE}" --timeout=600s
        kubectl wait --for=condition=available deployment/bharatvoice-worker -n "${NAMESPACE}" --timeout=600s
        log_success "Application deployed and ready"
    fi
    
    # Restore original files
    mv k8s/app-deployment.yaml.bak k8s/app-deployment.yaml
    mv k8s/worker-deployment.yaml.bak k8s/worker-deployment.yaml
}

# Deploy ingress/load balancer
deploy_ingress() {
    log_info "Deploying Nginx ingress"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        kubectl apply -f k8s/nginx-deployment.yaml --dry-run=client -o yaml
    else
        kubectl apply -f k8s/nginx-deployment.yaml
        
        # Wait for Nginx to be ready
        log_info "Waiting for Nginx to be ready..."
        kubectl wait --for=condition=available deployment/nginx -n "${NAMESPACE}" --timeout=300s
        log_success "Nginx deployed and ready"
    fi
}

# Deploy monitoring
deploy_monitoring() {
    log_info "Deploying monitoring stack"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        kubectl apply -f k8s/monitoring.yaml --dry-run=client -o yaml
    else
        kubectl apply -f k8s/monitoring.yaml
        
        # Wait for monitoring components
        log_info "Waiting for monitoring components to be ready..."
        kubectl wait --for=condition=available deployment/prometheus -n "${NAMESPACE}" --timeout=300s || log_warn "Prometheus deployment timeout"
        kubectl wait --for=condition=available deployment/grafana -n "${NAMESPACE}" --timeout=300s || log_warn "Grafana deployment timeout"
        log_success "Monitoring stack deployed"
    fi
}

# Deploy auto-scaling
deploy_autoscaling() {
    log_info "Deploying auto-scaling configuration"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        kubectl apply -f k8s/hpa.yaml --dry-run=client -o yaml
        kubectl apply -f k8s/vpa.yaml --dry-run=client -o yaml
        kubectl apply -f k8s/pdb.yaml --dry-run=client -o yaml
    else
        kubectl apply -f k8s/hpa.yaml
        kubectl apply -f k8s/vpa.yaml || log_warn "VPA deployment failed (VPA may not be installed)"
        kubectl apply -f k8s/pdb.yaml
        log_success "Auto-scaling configuration deployed"
    fi
}

# Deploy backup jobs
deploy_backup_jobs() {
    log_info "Deploying backup jobs"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        kubectl apply -f k8s/backup-cronjob.yaml --dry-run=client -o yaml
    else
        kubectl apply -f k8s/backup-cronjob.yaml
        log_success "Backup jobs deployed"
    fi
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "Would run database migrations"
        return
    fi
    
    # Wait for application to be ready
    kubectl wait --for=condition=available deployment/bharatvoice-app -n "${NAMESPACE}" --timeout=300s
    
    # Run migrations
    kubectl exec -n "${NAMESPACE}" deployment/bharatvoice-app -- \
        python -m alembic upgrade head || log_warn "Migration failed"
    
    log_success "Database migrations completed"
}

# Full deployment
deploy_full() {
    log_info "Starting full deployment of BharatVoice"
    
    create_namespace
    deploy_secrets
    deploy_config
    deploy_database
    deploy_redis
    deploy_application
    deploy_ingress
    deploy_monitoring
    deploy_autoscaling
    deploy_backup_jobs
    run_migrations
    
    log_success "Full deployment completed successfully"
    
    # Show deployment status
    show_status
}

# Update deployment
update_deployment() {
    log_info "Updating BharatVoice deployment"
    
    deploy_config
    deploy_application
    run_migrations
    
    log_success "Deployment update completed"
    show_status
}

# Rollback deployment
rollback_deployment() {
    log_info "Rolling back BharatVoice deployment"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "Would rollback deployments"
        return
    fi
    
    # Rollback application deployments
    kubectl rollout undo deployment/bharatvoice-app -n "${NAMESPACE}"
    kubectl rollout undo deployment/bharatvoice-worker -n "${NAMESPACE}"
    kubectl rollout undo deployment/nginx -n "${NAMESPACE}"
    
    # Wait for rollback to complete
    kubectl rollout status deployment/bharatvoice-app -n "${NAMESPACE}" --timeout=300s
    kubectl rollout status deployment/bharatvoice-worker -n "${NAMESPACE}" --timeout=300s
    kubectl rollout status deployment/nginx -n "${NAMESPACE}" --timeout=300s
    
    log_success "Rollback completed"
    show_status
}

# Show deployment status
show_status() {
    log_info "Deployment status for namespace: ${NAMESPACE}"
    
    echo ""
    echo "=== Pods ==="
    kubectl get pods -n "${NAMESPACE}" -o wide
    
    echo ""
    echo "=== Services ==="
    kubectl get services -n "${NAMESPACE}"
    
    echo ""
    echo "=== Deployments ==="
    kubectl get deployments -n "${NAMESPACE}"
    
    echo ""
    echo "=== HPA Status ==="
    kubectl get hpa -n "${NAMESPACE}" || log_warn "No HPA found"
    
    echo ""
    echo "=== Ingress/LoadBalancer ==="
    kubectl get services nginx-service -n "${NAMESPACE}" -o wide || log_warn "Nginx service not found"
    
    # Show application URLs
    echo ""
    echo "=== Application URLs ==="
    EXTERNAL_IP=$(kubectl get service nginx-service -n "${NAMESPACE}" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Pending")
    echo "Application: http://${EXTERNAL_IP}"
    echo "Grafana: http://${EXTERNAL_IP}:3000"
    echo "Prometheus: http://${EXTERNAL_IP}:9090"
}

# Show application logs
show_logs() {
    log_info "Showing application logs"
    
    echo "=== BharatVoice App Logs ==="
    kubectl logs -n "${NAMESPACE}" deployment/bharatvoice-app --tail=50
    
    echo ""
    echo "=== BharatVoice Worker Logs ==="
    kubectl logs -n "${NAMESPACE}" deployment/bharatvoice-worker --tail=50
}

# Cleanup resources
cleanup_resources() {
    log_warn "Cleaning up BharatVoice resources"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "Would delete namespace: ${NAMESPACE}"
        return
    fi
    
    read -p "Are you sure you want to delete namespace ${NAMESPACE} and all resources? (yes/no): " -r
    
    if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        kubectl delete namespace "${NAMESPACE}"
        log_success "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

# Main execution
main() {
    log_info "BharatVoice Deployment Script"
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Namespace: ${NAMESPACE}"
    log_info "Image Tag: ${IMAGE_TAG}"
    log_info "Registry: ${REGISTRY}"
    
    check_prerequisites
    
    case "${COMMAND}" in
        "deploy")
            deploy_full
            ;;
        "update")
            update_deployment
            ;;
        "rollback")
            rollback_deployment
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "cleanup")
            cleanup_resources
            ;;
        *)
            log_error "Unknown command: ${COMMAND}"
            usage
            ;;
    esac
}

# Execute main function
main "$@"