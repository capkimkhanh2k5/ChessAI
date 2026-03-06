# Multi-stage build for optimized React Frontend

# Stage 1: Build stage
FROM node:20-alpine as builder

# Set work directory
WORKDIR /app

# Copy package files first (better caching)
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci --only=production=false

# Copy source code
COPY . .

# Build argument for API URL
# Default to /api for production (nginx proxy)
# Use http://localhost:8000 for local dev
ARG VITE_API_URL=/api
ENV VITE_API_URL=$VITE_API_URL

# Build application
RUN npm run build

# Stage 2: Production stage with nginx
FROM nginx:alpine as production

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built files from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
