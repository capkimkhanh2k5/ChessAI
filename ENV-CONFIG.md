# Environment Variables Configuration

## Overview

This project uses different environment configurations for development and production:

- **Development**: Full URLs with ports (e.g., `http://localhost:8000`)
- **Production**: Relative paths for nginx reverse proxy (e.g., `/api`)

## Frontend Environment Variables

### `VITE_API_URL`

The API endpoint that the frontend uses to communicate with the backend.

**Local Development** (`Frontend/.env`):
```bash
VITE_API_URL=http://localhost:8000
```

**Production** (Docker build):
```bash
VITE_API_URL=/api
```

### Why Different Values?

#### Development
- Frontend runs on `localhost:5173` (Vite dev server)
- Backend runs on `localhost:8000` (FastAPI)
- Need full URL with port to make API calls

#### Production
- Both services behind nginx reverse proxy
- Nginx routes `/api/*` to backend
- Frontend can use relative path `/api`
- Nginx handles domain and SSL

## Nginx Configuration

The nginx setup in `nginx/chess-ai.conf`:
```nginx
# Backend API
location /api/ {
    rewrite ^/api(.*)$ $1 break;
    proxy_pass http://chess_backend;
}

# Frontend
location / {
    proxy_pass http://chess_frontend;
}
```

## Docker Build

### Important: Build-Time vs Runtime

**VITE_API_URL is a BUILD-TIME variable!**

When Vite builds the React app, it replaces `import.meta.env.VITE_API_URL` with the actual value. This means:

✅ **Correct**: Set VITE_API_URL when building the Docker image
```bash
docker build --build-arg VITE_API_URL=/api -t image:tag ./Frontend
```

❌ **Wrong**: Try to change VITE_API_URL at runtime
```bash
# This won't work - the value is already baked into the JS bundle!
docker run -e VITE_API_URL=/api image:tag
```

### Building Production Images

**Option 1: Use the build script (Recommended)**
```bash
cp .env.prod.example .env.prod
# Edit .env.prod with your values
./scripts/build-prod.sh
```

**Option 2: Use docker-compose**
```bash
export VITE_API_URL=/api
docker-compose -f docker-compose.prod.yml build
```

**Option 3: Manual docker build**
```bash
docker build --build-arg VITE_API_URL=/api -t username/chess-bot-frontend:latest ./Frontend
```

## Troubleshooting

### Problem: API calls fail with 404 or CORS errors

**Check 1**: Inspect the built JavaScript bundle
```bash
# Extract files from Docker image
docker create --name temp username/chess-bot-frontend:latest
docker cp temp:/usr/share/nginx/html/assets/index-*.js bundle.js
docker rm temp

# Search for API URL
grep -o 'baseURL:"[^"]*"' bundle.js
```

**Expected**: Should show `baseURL:"/api"` not `baseURL:"http://localhost:8000"`

**Fix**: Rebuild the image with correct VITE_API_URL:
```bash
docker build --build-arg VITE_API_URL=/api -t image:tag ./Frontend
```

### Problem: Changes to .env don't take effect

Remember: `.env` is only used during development. For production:
1. Set environment variables BEFORE building
2. Rebuild the Docker image
3. Deploy the new image

### Problem: Hard refresh shows old values

This can happen if:
1. Browser cached old JS bundle
2. Docker image wasn't actually rebuilt
3. Old image was deployed instead of new one

**Fix**:
```bash
# Force rebuild without cache
docker build --no-cache --build-arg VITE_API_URL=/api -t image:tag ./Frontend

# Push to Docker Hub
docker push image:tag

# On server, pull and restart
docker pull image:tag
docker-compose -f docker-compose.prod.yml up -d --force-recreate
```

## Best Practices

1. **Always use relative paths in production** (`/api`) when using nginx reverse proxy
2. **Use the build script** `./scripts/build-prod.sh` to ensure consistency
3. **Never commit `.env` or `.env.prod`** files - use `.env.example` instead
4. **Tag images with versions** for easier rollback
5. **Test locally first** with `docker-compose up` before pushing to production

## Default Values

If VITE_API_URL is not set, the application uses these defaults:

- **Dockerfile**: `/api` (for production builds)
- **Source code fallback**: `/api` (for when env var is missing)

This means you can safely build without specifying VITE_API_URL and it will work with nginx reverse proxy by default.
