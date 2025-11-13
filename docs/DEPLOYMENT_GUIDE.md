# TracingRAG Deployment Guide

This guide covers deploying TracingRAG to production environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Security](#security)
- [Scaling](#scaling)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Services

TracingRAG requires the following services:

1. **PostgreSQL 16+** with TimescaleDB extension
2. **Redis 7+** for caching
3. **Qdrant** for vector storage
4. **Neo4j 5+** for graph database

### Environment Variables

Required environment variables:

```bash
# Database
DATABASE_URL=postgresql://user:password@host:5432/tracingrag
REDIS_URL=redis://host:6379/0

# Vector & Graph Databases
QDRANT_URL=http://qdrant:6333
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=changeme

# LLM API
OPENROUTER_API_KEY=your-api-key

# Security
JWT_SECRET_KEY=your-secret-key-change-in-production
API_KEYS=key1,key2,key3  # Optional: comma-separated API keys

# Embedding Model
EMBEDDING_MODEL=all-mpnet-base-v2
```

## Docker Deployment

### Single Container (Development)

```bash
# Build
docker build -t tracingrag:latest .

# Run
docker run -d \
  --name tracingrag \
  -p 8000:8000 \
  -e DATABASE_URL="postgresql://..." \
  -e REDIS_URL="redis://..." \
  -e QDRANT_URL="http://qdrant:6333" \
  -e NEO4J_URI="bolt://neo4j:7687" \
  -e NEO4J_USERNAME="neo4j" \
  -e NEO4J_PASSWORD="changeme" \
  -e OPENROUTER_API_KEY="your-key" \
  -e JWT_SECRET_KEY="your-secret" \
  tracingrag:latest
```

### Docker Compose (Full Stack)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f tracingrag-api

# Stop all services
docker-compose down
```

### Production Build

Use the multi-stage Dockerfile for optimized production builds:

```bash
# Build production image
docker build -f Dockerfile.prod -t tracingrag:prod .

# Image is optimized for size and security:
# - Multi-stage build (smaller final image)
# - Non-root user
# - No dev dependencies
# - Security best practices
```

## Kubernetes Deployment

### Quick Start

```bash
# Create namespace
kubectl create namespace tracingrag

# Apply manifests
kubectl apply -f k8s/ -n tracingrag

# Check deployment status
kubectl get pods -n tracingrag
kubectl get svc -n tracingrag
```

### Step-by-Step Deployment

#### 1. Create Secrets

```bash
# Create secrets from file
kubectl create secret generic tracingrag-secrets \
  --from-literal=database-url="postgresql://..." \
  --from-literal=redis-url="redis://..." \
  --from-literal=neo4j-username="neo4j" \
  --from-literal=neo4j-password="changeme" \
  --from-literal=openrouter-api-key="your-key" \
  --from-literal=jwt-secret-key="your-secret" \
  -n tracingrag
```

#### 2. Apply ConfigMap

```bash
kubectl apply -f k8s/configmap.yaml -n tracingrag
```

#### 3. Deploy Application

```bash
kubectl apply -f k8s/deployment.yaml -n tracingrag
kubectl apply -f k8s/service.yaml -n tracingrag
```

#### 4. Configure Ingress (Optional)

```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# Apply ingress
kubectl apply -f k8s/ingress.yaml -n tracingrag
```

#### 5. Enable Autoscaling (Optional)

```bash
# Install Metrics Server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Apply HPA
kubectl apply -f k8s/hpa.yaml -n tracingrag
```

### Verify Deployment

```bash
# Check pods
kubectl get pods -n tracingrag

# Check services
kubectl get svc -n tracingrag

# Check ingress
kubectl get ingress -n tracingrag

# View logs
kubectl logs -f deployment/tracingrag-api -n tracingrag

# Check health
kubectl port-forward svc/tracingrag-api 8000:80 -n tracingrag
curl http://localhost:8000/health
```

## Configuration

### Resource Limits

Default resource limits in Kubernetes:

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

Adjust based on your workload:
- **Light load (<100 req/min)**: 1Gi RAM, 500m CPU
- **Medium load (<1000 req/min)**: 2Gi RAM, 1000m CPU
- **Heavy load (>1000 req/min)**: 4Gi+ RAM, 2000m+ CPU

### Autoscaling

HPA (Horizontal Pod Autoscaler) configuration:

```yaml
minReplicas: 3
maxReplicas: 10
targetCPUUtilizationPercentage: 70
targetMemoryUtilizationPercentage: 80
```

### Cache Configuration

Configure cache TTLs via environment variables:

```bash
CACHE_TTL_EMBEDDINGS=604800  # 7 days
CACHE_TTL_QUERIES=3600       # 1 hour
CACHE_TTL_WORKING_MEMORY=1800  # 30 minutes
CACHE_TTL_LATEST_STATE=86400   # 24 hours
```

### Rate Limiting

Configure rate limits:

```bash
RATE_LIMIT_REQUESTS_PER_MINUTE=60  # Per user/IP
```

## Monitoring

### Prometheus Metrics

TracingRAG exposes Prometheus metrics at `/metrics`:

```bash
# Scrape configuration
scrape_configs:
  - job_name: 'tracingrag'
    static_configs:
      - targets: ['tracingrag-api:8000']
    metrics_path: '/metrics'
```

### Key Metrics

- `tracingrag_api_requests_total`: Total API requests
- `tracingrag_api_request_duration_seconds`: Request latency
- `tracingrag_query_duration_seconds`: Query performance
- `tracingrag_embedding_cache_hits`: Cache hit rate
- `tracingrag_llm_tokens_total`: LLM token usage
- `tracingrag_promotion_success_total`: Successful promotions

### Grafana Dashboards

Import the provided Grafana dashboards:

```bash
# Dashboards located in monitoring/grafana/dashboards/
- tracingrag-overview.json
- tracingrag-performance.json
- tracingrag-errors.json
```

### Logs

Structured JSON logs are emitted to stdout:

```json
{
  "timestamp": "2025-01-15T10:30:00Z",
  "level": "INFO",
  "component": "api",
  "request_id": "abc123",
  "event": "api_request",
  "method": "POST",
  "endpoint": "/api/v1/query",
  "status": 200,
  "duration_ms": 450
}
```

Configure log aggregation (ELK, Loki, etc.) to collect logs.

### Health Checks

Health check endpoints:

- `/health`: Application health
- `/metrics`: Prometheus metrics

Kubernetes probes:

```yaml
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
  initialDelaySeconds: 10
  periodSeconds: 5
```

## Security

### Authentication

TracingRAG supports two authentication methods:

#### 1. JWT Tokens

```bash
# Generate access token
POST /api/v1/auth/login
{
  "username": "user",
  "password": "password"
}

# Returns JWT token
{
  "access_token": "eyJ...",
  "token_type": "bearer"
}

# Use token in requests
Authorization: Bearer eyJ...
```

#### 2. API Keys

```bash
# Set API keys via environment
API_KEYS=key1,key2,key3

# Use in requests
Authorization: Bearer your-api-key
```

### TLS/SSL

#### Docker

Use a reverse proxy (nginx, Traefik) with Let's Encrypt:

```bash
# docker-compose.yml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
```

#### Kubernetes

Use cert-manager for automatic TLS:

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
      - http01:
          ingress:
            class: nginx
EOF

# Ingress will automatically get TLS certificate
```

### Network Security

- Use network policies to restrict traffic
- Enable firewall rules
- Use private networks for database connections
- Rotate secrets regularly

### Input Validation

TracingRAG includes built-in input validation:
- SQL injection prevention
- XSS prevention
- Input length limits
- Dangerous pattern detection

## Scaling

### Horizontal Scaling

Scale API pods:

```bash
# Manual scaling
kubectl scale deployment tracingrag-api --replicas=5 -n tracingrag

# Autoscaling (HPA)
kubectl apply -f k8s/hpa.yaml -n tracingrag
```

### Database Scaling

#### PostgreSQL

- Use read replicas for read-heavy workloads
- Enable connection pooling (PgBouncer)
- Partition large tables by date

#### Redis

- Use Redis Cluster for high availability
- Enable persistence (AOF + RDB)
- Configure eviction policy

#### Qdrant

- Shard collections by topic
- Use quantization for memory efficiency
- Enable replication

#### Neo4j

- Use Neo4j Causal Cluster for HA
- Optimize indexes
- Use APOC procedures

### Performance Tuning

1. **Cache warming**: Pre-populate caches on startup
2. **Batch operations**: Use batch embedding generation
3. **Connection pooling**: Configure appropriate pool sizes
4. **Query optimization**: Index frequently queried fields
5. **LLM optimization**: Use cheaper models for non-critical operations

## Troubleshooting

### Common Issues

#### 1. Pod not starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n tracingrag

# Common causes:
# - Image pull errors
# - Missing secrets
# - Resource limits too low
# - Health check failing
```

#### 2. Database connection errors

```bash
# Verify connection string
kubectl exec -it <pod-name> -n tracingrag -- env | grep DATABASE_URL

# Test connection
kubectl exec -it <pod-name> -n tracingrag -- python -c "
from tracingrag.storage.database import get_engine
engine = get_engine()
print('Connection successful')
"
```

#### 3. High memory usage

```bash
# Check memory usage
kubectl top pods -n tracingrag

# Solutions:
# - Increase memory limits
# - Enable cache eviction
# - Reduce cache TTLs
# - Scale horizontally
```

#### 4. Slow query performance

```bash
# Check metrics
curl http://localhost:8000/metrics | grep query_duration

# Solutions:
# - Enable query result caching
# - Optimize database indexes
# - Reduce graph traversal depth
# - Use agent-based retrieval
```

### Debug Mode

Enable debug logging:

```bash
# Set log level
LOG_LEVEL=DEBUG

# View detailed logs
kubectl logs -f deployment/tracingrag-api -n tracingrag
```

### Support

For issues and questions:
- GitHub Issues: https://github.com/your-org/tracingrag/issues
- Documentation: https://docs.tracingrag.example.com
- Slack/Discord: [community link]

## Backup & Recovery

### Database Backups

#### PostgreSQL

```bash
# Daily backup
pg_dump -h postgres -U tracingrag tracingrag > backup-$(date +%Y%m%d).sql

# Restore
psql -h postgres -U tracingrag tracingrag < backup-20250115.sql
```

#### Neo4j

```bash
# Backup
neo4j-admin database dump neo4j --to-path=/backups

# Restore
neo4j-admin database load neo4j --from-path=/backups
```

#### Qdrant

```bash
# Backup collections
curl -X POST http://qdrant:6333/collections/memory_states/snapshots

# Restore
curl -X PUT http://qdrant:6333/collections/memory_states/snapshots/upload \
  -H "Content-Type: multipart/form-data" \
  -F "snapshot=@snapshot.tar"
```

### Disaster Recovery

1. **Regular backups**: Daily database backups, weekly full backups
2. **Multi-region**: Deploy to multiple regions
3. **Monitoring**: Alert on anomalies
4. **Runbooks**: Document recovery procedures

## Production Checklist

Before going to production:

- [ ] All secrets configured
- [ ] TLS/SSL enabled
- [ ] Authentication enabled
- [ ] Rate limiting configured
- [ ] Monitoring set up (Prometheus + Grafana)
- [ ] Log aggregation configured
- [ ] Backup strategy implemented
- [ ] Disaster recovery plan documented
- [ ] Load testing completed
- [ ] Security scan passed
- [ ] Health checks passing
- [ ] Autoscaling configured
- [ ] Documentation updated
- [ ] Team trained on operations

## Summary

TracingRAG is production-ready with:
- ✅ Docker and Kubernetes support
- ✅ Horizontal autoscaling
- ✅ Comprehensive monitoring
- ✅ Security best practices
- ✅ High availability options
- ✅ Backup and recovery procedures

For questions or issues, refer to the [documentation](../README.md) or open an issue on GitHub.
