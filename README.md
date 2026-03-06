# Chess Bot API

A chess engine application with three different bot difficulty levels and a React-based frontend interface.

## Project Structure

- **Backend**: FastAPI-based REST API with three chess engines
- **Frontend**: React + Vite with chess.js and react-chessboard

## Features

### Chess Engines

1. **Bot V1**: Simple alpha-beta pruning engine
2. **Bot V2**: Enhanced with iterative deepening, transposition tables, and late move reduction
3. **Bot VIP**: Advanced engine with static exchange evaluation, aspiration windows, and pawn hash tables

## Quick Start with Docker

### Local Development

```bash
docker-compose up -d
```

Services will be available at:
- Backend API: http://localhost:8000/docs
- Frontend: http://localhost:5000

### Production Deployment

1. **Create production environment file**:
```bash
cp .env.prod.example .env.prod
```

2. **Configure environment variables** in `.env.prod`:
```bash
# Docker Hub credentials
DOCKERHUB_USERNAME=your_dockerhub_username

# Frontend API URL (used during Docker build)
VITE_API_URL=https://your-domain.com/api
```

3. **Build and deploy**:
```bash
# Build images with production config
docker-compose -f docker-compose.prod.yml build
# Or use the deployment script
./scripts/deploy.sh
```

> **Note**: `VITE_API_URL` is a build-time variable for Vite. If you change it, you must rebuild the frontend image.

### Stop Services

```bash
docker-compose down
```

### View Logs

```bash
docker-compose logs -f
```

## Manual Setup

### Backend

1. Navigate to Backend folder:
```bash
cd Backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API documentation: http://localhost:8000/docs

### Frontend

1. Navigate to Frontend folder:
```bash
cd Frontend
```

2. **Setup environment variables**:
```bash
cp .env.example .env
```

Edit `.env` to configure API URL:
```
VITE_API_URL=http://localhost:8000
```

3. Install dependencies:
```bash
5pm install
```

4. Run development server:
```bash
npm run dev
```

Application: http://localhost:5173

4. Build for production:
```bash
npm run build
```

## API Endpoints

### Get Best Move

```
POST /api/v1/best-move
POST /api/v2/best-move
POST /api/vip/best-move
```

Request body:
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "depth": 4
}
```

Response:
```json
{
  "move": "e2e4",
  "eval": 0.5
}
```

## Development

### Run Tests

Backend:
```bash
cd Backend
pytest
```

Frontend:
```bash
cd Frontend
npm run lint
```

## Technology Stack

### Backend
- FastAPI
- python-chess
- uvicorn
- numpy
- pydantic-settings

### Frontend
- React 19
- Vite
- chess.js
- react-chessboard
- axios

## Docker Images

- Backend: Python 3.11-slim based (approximately 150-200MB)
- Frontend: Multi-stage Nginx Alpine based (approximately 25-35MB)

## 🚀 CI/CD Deployment

This project includes automated CI/CD pipeline for deploying to Google Cloud Platform.

### Quick Setup

1. **Docker Hub**: Create account and access token
2. **GCP VM**: Setup Ubuntu VM with Docker installed
3. **GitHub Secrets**: Configure deployment secrets
4. **Nginx**: Setup reverse proxy on GCP
5. **Deploy**: Push to GitHub → Auto deploy!

### Documentation

- **[CI/CD Quick Start](./CI-CD-SETUP.md)** - 10-step setup guide
- **[Complete Deployment Guide](./DEPLOY.md)** - Detailed documentation with monitoring and troubleshooting

### GitHub Actions Workflow

Pipeline automatically:
1. ✅ Builds Docker images
2. ✅ Pushes to Docker Hub with version tags
3. ✅ Deploys to GCP via SSH
4. ✅ Runs health checks

Triggers on:
- Push to `main` or `prod` branch
- Manual workflow dispatch

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `GCP_HOST` | GCP VM public IP |
| `GCP_USERNAME` | SSH username |
| `GCP_SSH_KEY` | SSH private key |

### Production URLs

After deployment, access your application:
- Frontend: `http://YOUR_GCP_IP/`
- API Docs: `http://YOUR_GCP_IP/docs`
- API Endpoint: `http://YOUR_GCP_IP/api/`

## License

[Add your license here]

