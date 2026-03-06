#!/bin/bash

##############################################
# Chess Bot Deployment Script for GCP
# This script pulls latest images and deploys
##############################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Start deployment
log_info "Starting Chess Bot deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed!"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    log_error "Docker Compose is not installed!"
    exit 1
fi

# Set Docker Compose command
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

log_info "Using: $DOCKER_COMPOSE"

# Load environment variables
if [ -f .env.prod ]; then
    log_info "Loading environment variables from .env.prod"
    export $(cat .env.prod | grep -v '^#' | xargs)
elif [ -f .env ]; then
    log_info "Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
else
    log_warn "No .env.prod or .env file found, using default values"
fi

# Pull latest images from Docker Hub
log_info "Pulling latest images from Docker Hub..."
docker pull ${DOCKERHUB_USERNAME}/${IMAGE_BACKEND}:latest || log_error "Failed to pull backend image"
docker pull ${DOCKERHUB_USERNAME}/${IMAGE_FRONTEND}:latest || log_error "Failed to pull frontend image"

# Stop existing containers
log_info "Stopping existing containers..."
$DOCKER_COMPOSE -f docker-compose.prod.yml down || log_warn "No containers to stop"

# Remove old containers and images (optional - cleanup)
log_info "Cleaning up old containers and images..."
docker system prune -af --volumes || log_warn "Cleanup failed, continuing anyway"

# Start new containers
log_info "Starting new containers..."
$DOCKER_COMPOSE -f docker-compose.prod.yml up -d

# Wait for services to be healthy
log_info "Waiting for services to be healthy..."
sleep 10

# Check if containers are running
if docker ps | grep -q chess-backend && docker ps | grep -q chess-frontend; then
    log_info "✅ All containers are running!"
else
    log_error "❌ Some containers failed to start!"
    $DOCKER_COMPOSE -f docker-compose.prod.yml logs
    exit 1
fi

# Display container status
log_info "Container status:"
docker ps --filter "name=chess-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Test backend health
log_info "Testing backend health..."
sleep 5
if curl -f http://localhost:8000/docs &> /dev/null; then
    log_info "✅ Backend is healthy!"
else
    log_warn "⚠️ Backend health check failed"
fi

# Test frontend health
log_info "Testing frontend health..."
if curl -f http://localhost:80 &> /dev/null; then
    log_info "✅ Frontend is healthy!"
else
    log_warn "⚠️ Frontend health check failed"
fi

# Display logs (last 20 lines)
log_info "Recent logs:"
$DOCKER_COMPOSE -f docker-compose.prod.yml logs --tail=20

log_info "🎉 Deployment completed successfully!"
log_info "Backend API: http://localhost:8000/docs"
log_info "Frontend: http://localhost:80"

exit 0
