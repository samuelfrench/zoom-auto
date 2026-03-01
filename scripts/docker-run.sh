#!/usr/bin/env bash
# Convenience script to build and run zoom-auto with GPU support.
# Usage:
#   ./scripts/docker-run.sh          # build + run
#   ./scripts/docker-run.sh --build  # force rebuild
#   ./scripts/docker-run.sh --down   # stop and remove container

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker/docker-compose.yml"

# Ensure .env exists (docker-compose needs it even if empty)
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "Warning: .env file not found at $PROJECT_ROOT/.env"
    echo "Create one from .env.example or provide required environment variables."
    exit 1
fi

# Ensure data directory exists for volume mount
mkdir -p "$PROJECT_ROOT/data"

case "${1:-}" in
    --down)
        echo "Stopping zoom-auto..."
        docker compose -f "$COMPOSE_FILE" down
        ;;
    --build)
        echo "Rebuilding and starting zoom-auto..."
        docker compose -f "$COMPOSE_FILE" up --build -d
        echo "Container started. Logs: docker compose -f $COMPOSE_FILE logs -f"
        ;;
    --logs)
        docker compose -f "$COMPOSE_FILE" logs -f
        ;;
    *)
        echo "Starting zoom-auto..."
        docker compose -f "$COMPOSE_FILE" up -d
        echo "Container started. Logs: docker compose -f $COMPOSE_FILE logs -f"
        ;;
esac
