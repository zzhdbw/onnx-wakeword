#!/bin/bash
echo "Voicute Wake Word Demo"
echo "Open http://localhost:8080/web/"
echo "Press Ctrl+C to stop"
cd "$(dirname "$0")/.."
python3 -m http.server 8080
