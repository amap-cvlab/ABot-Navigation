#!/usr/bin/env bash
# Start a local HTTP server for ABot-Navigation static site.
# Default: http://localhost:8080

set -e
cd "$(dirname "$0")"

PORT="${1:-8080}"

echo "Starting local server at http://localhost:${PORT}"
echo "Press Ctrl+C to stop"

python3 -m http.server "${PORT}"
