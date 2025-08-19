# Docker Setup for Wealth Potential Estimator API with FaceNet

This project includes Docker configuration for easy deployment and development with integrated FaceNet model for facial embeddings extraction.

## Prerequisites

- Docker Engine (20.10+)
- Docker Compose (2.0+)

## Quick Start

### Production Mode

1. Build and start the container:
```bash
docker-compose up --build
```

2. The API will be available at: http://localhost:8000

3. View API documentation at: http://localhost:8000/docs

### Development Mode (with hot reload)

For development with automatic code reloading:

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

## Docker Commands

### Start the application
```bash
# Start in background
docker-compose up -d

# Start with build
docker-compose up --build

# Start and see logs
docker-compose up
```

### Stop the application
```bash
# Stop containers
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove containers, volumes, and images
docker-compose down -v --rmi all
```

### View logs
```bash
# View all logs
docker-compose logs

# View logs for specific service
docker-compose logs wealth-estimator-api

# Follow logs in real-time
docker-compose logs -f
```

### Access the container
```bash
# Execute bash shell in running container
docker-compose exec wealth-estimator-api bash

# Run a command in the container
docker-compose exec wealth-estimator-api python --version
```

### Rebuild the image
```bash
# Rebuild without cache
docker-compose build --no-cache

# Rebuild and start
docker-compose up --build
```

## File Structure

- `Dockerfile` - Defines the container image for the FastAPI application
- `docker-compose.yml` - Main orchestration file for production
- `docker-compose.dev.yml` - Override configuration for development
- `.dockerignore` - Specifies files to exclude from the Docker build context

## Volume Mounts

The following directories are mounted as volumes:

- `./uploads:/app/uploads` - Persists uploaded images
- `./dataset:/app/dataset` - Provides access to dataset files

In development mode, the source code is also mounted for hot reload:
- `./main.py:/app/main.py`

## Environment Variables

You can customize the application by setting environment variables in the docker-compose files:

- `PYTHONUNBUFFERED=1` - Ensures Python output is sent straight to terminal
- `ENV=development` - Set in dev mode for development-specific configurations

## Health Check

The container includes a health check that pings the `/health` endpoint every 30 seconds. You can check the health status:

```bash
docker-compose ps
```

## Troubleshooting

### Port already in use
If port 8000 is already in use, you can change it in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Change 8001 to your desired port
```

### Permission issues with uploads
If you encounter permission issues with the uploads directory:
```bash
# Create uploads directory with proper permissions
mkdir -p uploads
chmod 755 uploads
```

### Container won't start
Check the logs for errors:
```bash
docker-compose logs wealth-estimator-api
```

### Clean restart
For a completely fresh start:
```bash
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

## Testing the API with Docker

Once the container is running, you can test the endpoints:

### Using the test script:
```bash
# Run comprehensive tests
python test_api.py
```

### Manual testing with curl:

```bash
# Check health endpoint
curl http://localhost:8000/health

# Check model status
curl http://localhost:8000/model-status

# Upload an image
curl -X POST "http://localhost:8000/wealth-potential" \
     -H "accept: application/json" \
     -F "file=@dataset/selfie1.png"

# Extract facial embeddings
curl -X POST "http://localhost:8000/extract-embeddings" \
     -H "accept: application/json" \
     -F "file=@dataset/selfie1.png"
```

## FaceNet Model Integration

The application includes FaceNet model integration for facial embeddings extraction:

### Model Download Strategies:

1. **Production (Recommended)**: Model is downloaded during Docker build
   - Model is cached in the Docker image
   - No download needed on startup
   - Larger image size (~500MB more)
   - Use: `docker-compose up --build`

2. **Development**: Model downloaded on first startup
   - Smaller Docker image
   - Model downloaded to mounted volume
   - Persists between container restarts
   - Use: `docker-compose -f docker-compose.yml -f docker-compose.dev.yml up`

3. **Manual Pre-download**: Download model before building
   ```bash
   # Create models directory
   mkdir -p models
   
   # Download model manually
   python download_model.py
   
   # Then build and run
   docker-compose up --build
   ```

### Model Cache Location:
- Inside container: `/app/models/`
- Host (development): `./models/`
- Model file: `facenet_20180402_114759_vggface2.pth` (~107MB)

### API Endpoints:

- `POST /wealth-potential` - Upload image for processing
- `POST /extract-embeddings` - Extract 512-dimensional facial embeddings
- `GET /model-status` - Check if model is loaded
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

## Production Considerations

For production deployment:

1. Use environment-specific configuration files
2. Set up proper logging volumes
3. Configure SSL/TLS termination (use a reverse proxy like nginx)
4. Set resource limits in docker-compose.yml
5. Use Docker secrets for sensitive data
6. Consider using Docker Swarm or Kubernetes for orchestration
