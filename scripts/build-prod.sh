#!/bin/bash

##############################################
# Build Production Docker Images
# This script builds images for production
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

# Start build
log_info "Starting production build..."

# Load environment variables
if [ -f .env.prod ]; then
    log_info "Loading environment variables from .env.prod"
    export $(cat .env.prod | grep -v '^#' | xargs)
elif [ -f .env ]; then
    log_info "Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
else
    log_warn "No .env.prod or .env file found"
    log_error "Please create .env.prod with required variables"
    exit 1
fi

# Check required variables
if [ -z "$DOCKERHUB_USERNAME" ]; then
    log_error "DOCKERHUB_USERNAME is not set in .env.prod"
    exit 1
fi

# Set defaults
VITE_API_URL=${VITE_API_URL:-/api}
IMAGE_BACKEND=${IMAGE_BACKEND:-chess-bot-backend}
IMAGE_FRONTEND=${IMAGE_FRONTEND:-chess-bot-frontend}

log_info "Configuration:"
log_info "  Docker Hub: $DOCKERHUB_USERNAME"
log_info "  Backend Image: $IMAGE_BACKEND"
log_info "  Frontend Image: $IMAGE_FRONTEND"
log_info "  API URL: $VITE_API_URL"

# Build backend
log_info "Building backend image..."
docker build -t $DOCKERHUB_USERNAME/$IMAGE_BACKEND:latest ./Backend

if [ $? -eq 0 ]; then
    log_info "✅ Backend image built successfully"
else
    log_error "❌ Backend build failed"
    exit 1
fi

# Build frontend with API URL
log_info "Building frontend image with VITE_API_URL=$VITE_API_URL..."
docker build \
    --build-arg VITE_API_URL=$VITE_API_URL \
    -t $DOCKERHUB_USERNAME/$IMAGE_FRONTEND:latest \
    ./Frontend

if [ $? -eq 0 ]; then
    log_info "✅ Frontend image built successfully"
else
    log_error "❌ Frontend build failed"
    exit 1
fi

# Display image info
log_info "Docker images:"
docker images | grep -E "$IMAGE_BACKEND|$IMAGE_FRONTEND"

# Ask to push to Docker Hub
read -p "Push images to Docker Hub? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Pushing images to Docker Hub..."
    docker push $DOCKERHUB_USERNAME/$IMAGE_BACKEND:latest
    docker push $DOCKERHUB_USERNAME/$IMAGE_FRONTEND:latest
    log_info "🎉 Images pushed successfully!"
else
    log_info "Skipping push. Build completed."
fi

log_info "Done!"
exit 0
